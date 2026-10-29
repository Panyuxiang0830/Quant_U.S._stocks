"""逐日横截面回归估计因子收益（市场 + 行业 + 风格合并在同一个回归里）。

r_t = 1·f_country + D_{t-1} f_ind + X_{t-1} f_style + u_t
- 权重：sqrt(期初市值)；
- 约束：Σ_j s_j f_ind_j = 0，s_j 为行业市值占比（解决“行业哑变量之和 = 市场列”的共线性）；
- 暴露一律用 t-1 收盘值，解释 t 日收益。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class CrossSection:
    factor_returns: pd.Series  # Market + 行业 + 风格
    specific_returns: pd.Series  # 所有有行业的股票
    r2: float
    n_obs: int


def design_matrix(exp: pd.DataFrame, styles: list[str], industries: list[str]) -> pd.DataFrame:
    """把 industry 列展开为哑变量，风格缺失填 0（= 市场平均）。"""
    dummies = pd.get_dummies(exp["industry"]).reindex(columns=industries, fill_value=False)
    X = pd.concat([pd.Series(1.0, index=exp.index, name=config.MARKET),
                   dummies.astype(float), exp[styles].astype(float).fillna(0.0)], axis=1)
    return X


def cross_section(exp: pd.DataFrame, ret: pd.Series, styles: list[str]) -> CrossSection:
    """exp: t-1 暴露（index=ts_code，含 industry/cap/estu/风格列）；ret: t 日收益。"""
    exp = exp[exp["industry"].notna()]
    r = ret.reindex(exp.index)
    fit = (exp["estu"].astype(bool) & r.notna() & exp["cap"].gt(0)
           & exp[styles].notna().all(axis=1) & r.abs().lt(config.REGRESSION_MAX_ABS_RETURN))
    industries = sorted(exp.loc[fit, "industry"].unique())
    X = design_matrix(exp, styles, industries)
    Xf, rf, cap = X[fit].to_numpy(), r[fit].to_numpy(), exp.loc[fit, "cap"].to_numpy(float)

    k, p = X.shape[1], len(industries)
    s = pd.Series(cap, index=exp.index[fit]).groupby(exp.loc[fit, "industry"]).sum()
    s = (s / s.sum()).reindex(industries).to_numpy()
    # f = C g：最后一个行业由约束表示为其他行业的线性组合
    C = np.delete(np.eye(k), p, axis=1)
    C[p, 1:p] = -s[:-1] / s[-1]
    sw = np.sqrt(np.sqrt(cap))  # 回归权重 w = sqrt(cap)，乘 sqrt(w) 后用 OLS
    g, *_ = np.linalg.lstsq((Xf @ C) * sw[:, None], rf * sw, rcond=None)
    f = pd.Series(C @ g, index=X.columns)

    has_r = r.notna()
    u = (r[has_r] - X[has_r].to_numpy() @ f.to_numpy())
    w = sw ** 2
    resid = rf - Xf @ f.to_numpy()
    r2 = 1.0 - (w * resid ** 2).sum() / (w * (rf - np.average(rf, weights=w)) ** 2).sum()
    return CrossSection(f, u, float(r2), int(fit.sum()))


def estimate(exposures: pd.DataFrame, returns: pd.DataFrame, styles: list[str]):
    """exposures: 长表（trade_date, ts_code）；returns: 日期×股票。

    返回 (因子收益 日期×因子, 特异收益 日期×股票, 诊断 日期×[r2, n_obs])。
    t 日使用 exposures 中 t 之前最近一个交易日的暴露。
    """
    exp_dates = exposures.index.get_level_values(0).unique().sort_values()
    rows, specs, diag = {}, {}, {}
    for prev, d in zip(exp_dates[:-1], exp_dates[1:]):
        if d not in returns.index:
            continue
        cs = cross_section(exposures.loc[prev], returns.loc[d], styles)
        rows[d], specs[d] = cs.factor_returns, cs.specific_returns
        diag[d] = {"r2": cs.r2, "n_obs": cs.n_obs}
    f = pd.DataFrame(rows).T.sort_index()
    u = pd.DataFrame(specs).T.sort_index().astype(np.float32)
    return f, u, pd.DataFrame(diag).T.sort_index()
