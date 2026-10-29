"""横截面预处理：去极值、标准化、细项合成、正交化。输入均为 日期×股票 宽表。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def ewm_weights(length: int, half_life: float) -> np.ndarray:
    """长度为 length 的指数权重，最后一个（最近）权重最大，和为 1。"""
    lag = np.arange(length)[::-1]
    w = 0.5 ** (lag / half_life)
    return w / w.sum()


def winsorize_mad(df: pd.DataFrame, universe: pd.DataFrame | None = None,
                  n: float = config.WINSOR_MAD) -> pd.DataFrame:
    """按 median ± n × 1.4826 × MAD 截断，统计量只用 universe 内股票。"""
    base = df if universe is None else df.where(universe)
    med = base.median(axis=1)
    mad = base.sub(med, axis=0).abs().median(axis=1) * 1.4826
    return df.clip(lower=med - n * mad, upper=med + n * mad, axis=0)


def standardize(df: pd.DataFrame, cap: pd.DataFrame,
                universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """市值加权去均值、等权标准差缩放（Barra 口径），统计量只用 universe 内股票。"""
    mask = df.notna() & cap.notna()
    if universe is not None:
        mask &= universe
    w = cap.where(mask)
    mu = (df.where(mask) * w).sum(axis=1) / w.sum(axis=1)
    sd = df.where(mask).std(axis=1, ddof=0)
    return df.sub(mu, axis=0).div(sd.replace(0.0, np.nan), axis=0)


def combine(parts: list[tuple[pd.DataFrame, float]]) -> pd.DataFrame:
    """加权合成细项；某只股票缺失的细项不计入，并对剩余权重重新归一。"""
    num, den = None, None
    for df, w in parts:
        valid = df.notna()
        term = df.fillna(0.0) * w
        wt = valid * w
        num = term if num is None else num + term
        den = wt if den is None else den + wt
    return num / den.where(den > 0)


def orthogonalize(y: pd.DataFrame, xs: list[pd.DataFrame], weight: pd.DataFrame,
                  universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """逐日用 weight 加权回归 y ~ 1 + xs，返回残差（所有有数据的股票都给出）。"""
    out = pd.DataFrame(np.nan, index=y.index, columns=y.columns)
    for d in y.index:
        yd = y.loc[d].to_numpy(float)
        X = np.column_stack([np.ones_like(yd)] + [x.loc[d].to_numpy(float) for x in xs])
        c = weight.loc[d].to_numpy(float)
        has = ~np.isnan(yd) & ~np.isnan(X).any(axis=1)
        fit = has & ~np.isnan(c) & (c > 0)
        if universe is not None:
            fit &= universe.loc[d].to_numpy(bool)
        if fit.sum() <= X.shape[1]:
            continue
        sw = np.sqrt(c[fit])
        coef, *_ = np.linalg.lstsq(X[fit] * sw[:, None], yd[fit] * sw, rcond=None)
        res = np.full_like(yd, np.nan)
        res[has] = yd[has] - X[has] @ coef
        out.loc[d] = res
    return out
