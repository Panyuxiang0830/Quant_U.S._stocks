"""量价类细项：LNCAP、BETA、HSIGMA、RSTR、DASTD、CMRA、STOM、STOQ、STOA。

只计算 dates 指定的截面，每个截面只使用当日及之前的数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .preprocess import ewm_weights


def market_return(ret: pd.DataFrame, cap: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """估计池内、按前一日市值加权的市场收益。"""
    w = cap.shift(1).where(universe.shift(1, fill_value=False) & ret.notna())
    return (ret * w).sum(axis=1) / w.sum(axis=1)


def _weighted_beta(y: np.ndarray, x: np.ndarray, w: np.ndarray, min_obs: int):
    valid = ~np.isnan(y) & ~np.isnan(x)[:, None]
    W = np.where(valid, w[:, None], 0.0)
    sw = W.sum(axis=0)
    y0 = np.where(valid, y, 0.0)
    x0 = np.where(valid, x[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xb = (W * x0).sum(axis=0) / sw
        yb = (W * y0).sum(axis=0) / sw
        dx, dy = x0 - xb, y0 - yb
        beta = (W * dx * dy).sum(axis=0) / (W * dx * dx).sum(axis=0)
        resid = np.where(valid, dy - beta * dx, 0.0)
        hsigma = np.sqrt((W * resid ** 2).sum(axis=0) / sw)
    bad = valid.sum(axis=0) < min_obs
    beta[bad] = np.nan
    hsigma[bad] = np.nan
    return beta, hsigma


def _weighted_std(y: np.ndarray, w: np.ndarray, min_obs: int) -> np.ndarray:
    valid = ~np.isnan(y)
    W = np.where(valid, w[:, None], 0.0)
    sw = W.sum(axis=0)
    y0 = np.where(valid, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (W * y0).sum(axis=0) / sw
        out = np.sqrt((W * np.where(valid, y0 - mu, 0.0) ** 2).sum(axis=0) / sw)
    out[valid.sum(axis=0) < min_obs] = np.nan
    return out


def _log_turnover(to: np.ndarray, months: int, min_obs: int) -> np.ndarray:
    valid = ~np.isnan(to)
    total = np.where(valid, to, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(total / months)
    out[(valid.sum(axis=0) < min_obs) | ~np.isfinite(out)] = np.nan
    return out


def price_descriptors(ret: pd.DataFrame, cap: pd.DataFrame, turnover: pd.DataFrame,
                      mkt: pd.Series, dates: list[str]) -> dict[str, pd.DataFrame]:
    """turnover 为百分比换手率（Tushare turnover_rate）。"""
    idx, cols = ret.index, ret.columns
    rf = (1.0 + config.RF_ANNUAL) ** (1 / 252) - 1.0
    R = ret.to_numpy(float) - rf
    M = mkt.reindex(idx).to_numpy(float) - rf
    LR = np.log1p(ret.to_numpy(float)) - np.log1p(rf)
    TO = turnover.reindex(index=idx, columns=cols).to_numpy(float) / 100.0
    names = ["BETA", "HSIGMA", "RSTR", "DASTD", "CMRA", "STOM", "STOQ", "STOA"]
    out = {k: np.full((len(dates), len(cols)), np.nan) for k in names}
    w_beta = ewm_weights(config.BETA_WINDOW, config.BETA_HALF_LIFE)
    w_dastd = ewm_weights(config.DASTD_WINDOW, config.DASTD_HALF_LIFE)
    w_rstr = ewm_weights(config.RSTR_WINDOW, config.RSTR_HALF_LIFE)
    m, n_month = config.MONTH_DAYS, config.CMRA_MONTHS
    for i, t in enumerate(idx.get_indexer(dates)):
        if t < 0:
            continue
        if t + 1 >= config.BETA_WINDOW:
            s = slice(t + 1 - config.BETA_WINDOW, t + 1)
            out["BETA"][i], out["HSIGMA"][i] = _weighted_beta(R[s], M[s], w_beta, config.BETA_MIN_OBS)
        if t + 1 >= config.DASTD_WINDOW:
            out["DASTD"][i] = _weighted_std(R[t + 1 - config.DASTD_WINDOW:t + 1], w_dastd,
                                            config.DASTD_WINDOW // 2)
        end = t + 1 - config.RSTR_LAG
        if end - config.RSTR_WINDOW >= 0:
            block = LR[end - config.RSTR_WINDOW:end]
            valid = ~np.isnan(block)
            rstr = (np.where(valid, block, 0.0) * w_rstr[:, None]).sum(axis=0)
            rstr[valid.sum(axis=0) < config.RSTR_MIN_OBS] = np.nan
            out["RSTR"][i] = rstr
        if t + 1 >= m * n_month:
            block = LR[t + 1 - m * n_month:t + 1]
            valid = ~np.isnan(block)
            monthly = np.where(valid, block, 0.0).reshape(n_month, m, -1).sum(axis=1)
            z = np.cumsum(monthly[::-1], axis=0)  # Z(T)：最近 T 个月累计对数超额收益
            cmra = z.max(axis=0) - z.min(axis=0)
            cmra[valid.sum(axis=0) < m * n_month * 3 // 4] = np.nan
            out["CMRA"][i] = cmra
        for name, months in (("STOM", 1), ("STOQ", 3), ("STOA", 12)):
            n = m * months
            if t + 1 >= n:
                out[name][i] = _log_turnover(TO[t + 1 - n:t + 1], months, n // 2)
    res = {k: pd.DataFrame(v, index=dates, columns=cols) for k, v in out.items()}
    res["LNCAP"] = np.log(cap.reindex(index=dates, columns=cols).where(lambda x: x > 0))
    return res
