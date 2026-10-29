"""收益归因：持仓 -> 组合因子暴露 -> 暴露 × 当日因子收益 -> 各项贡献。

约定：weights 的第 t 行是 t 日开盘前（= t-1 收盘后）的持仓权重，它赚取 t 日收益，
使用 t-1 的因子暴露。特异收益按 r_i - X_i f 当场计算，保证单日恒等式严格成立：
    Σ_i w_i r_i = Σ_k b_k f_k + Σ_i w_i u_i
停牌（收益缺失）的股票当日收益按 0 计。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .model import design_matrix

SPECIFIC = "Specific"


def drifted_weights(snapshots: pd.DataFrame, returns: pd.DataFrame,
                    dates: list[str]) -> pd.DataFrame:
    """snapshots 第 s 行 = s 日收盘后的目标权重；两次快照之间按个股收益漂移。

    返回每个 date 的开盘前权重。快照日若非交易日，视作其之前最近交易日的收盘权重。
    """
    tdays = returns.index
    snap = snapshots.copy()
    snap.index = tdays[tdays.searchsorted(snap.index.astype(str), side="right") - 1]
    snap = snap[~snap.index.duplicated(keep="last")]
    cols = returns.columns.union(snap.columns)
    rets = returns.reindex(columns=cols).fillna(0.0)
    snap = snap.reindex(columns=cols).fillna(0.0)
    want, out, cur = set(dates), {}, None
    start = tdays.searchsorted(snap.index.min())
    for d in tdays[start:]:
        if cur is not None:
            if d in want:
                out[d] = cur
            cur = cur * (1.0 + rets.loc[d].to_numpy())
            total = cur.sum()
            cur = cur / total if total != 0 else cur
        if d in snap.index:
            cur = snap.loc[d].to_numpy(float)
    res = pd.DataFrame(out, index=cols).T
    return res.loc[:, (res != 0).any(axis=0)]


def attribute(weights: pd.DataFrame, exposures: pd.DataFrame, factor_returns: pd.DataFrame,
              returns: pd.DataFrame, styles: list[str]):
    """返回 (贡献 日期×[因子..., Specific, Total], 组合暴露 日期×因子)。

    weights 可以是组合权重，也可以是主动权重（组合 - 基准），后者得到超额收益归因。
    """
    industries = [c for c in factor_returns.columns if c not in styles and c != config.MARKET]
    exp_dates = exposures.index.get_level_values(0).unique().sort_values()
    contrib, expo = {}, {}
    for d in weights.index:
        if d not in factor_returns.index:
            continue
        k = exp_dates.searchsorted(d) - 1
        if k < 0:
            continue
        w = weights.loc[d]
        w = w[w != 0]
        exp = exposures.loc[exp_dates[k]].reindex(w.index)
        X = design_matrix(exp, styles, industries)
        f = factor_returns.loc[d].reindex(X.columns).fillna(0.0)
        r = returns.loc[d].reindex(w.index).fillna(0.0)
        b = X.T @ w
        c = b * f
        c[SPECIFIC] = float((w * (r - X @ f)).sum())
        c["Total"] = float((w * r).sum())
        contrib[d], expo[d] = c, b
    return pd.DataFrame(contrib).T, pd.DataFrame(expo).T


def carino_link(contrib: pd.DataFrame, port: pd.Series, bench: pd.Series | None = None) -> pd.Series:
    """Carino 多期链接：各项链接后之和 = 累计（超额）收益。

    bench 为空时 port 是组合日收益；否则 port/bench 分别是组合与基准日收益，contrib 是超额贡献。
    """
    rp = port.reindex(contrib.index).astype(float)
    rb = pd.Series(0.0, index=contrib.index) if bench is None else bench.reindex(contrib.index)

    def k(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        diff = a - b
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (np.log1p(a) - np.log1p(b)) / diff
        return np.where(np.abs(diff) < 1e-12, 1.0 / (1.0 + a), out)

    Rp, Rb = (1 + rp).prod() - 1, (1 + rb).prod() - 1
    kt = k(rp, rb)
    K = float(k(Rp, Rb))
    return (contrib.mul(kt / K, axis=0)).sum()


def group(linked: pd.Series, styles: list[str]) -> pd.Series:
    """把各因子贡献归为 市场 / 行业 / 风格 / 特异 四大类。"""
    skip = {config.MARKET, SPECIFIC, "Total"}
    industry = [i for i in linked.index if i not in skip and i not in styles]
    return pd.Series({
        "Market": linked.get(config.MARKET, 0.0),
        "Industry": linked[industry].sum(),
        "Style": linked[[s for s in styles if s in linked.index]].sum(),
        "Specific": linked.get(SPECIFIC, 0.0),
    })
