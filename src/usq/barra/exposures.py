"""Transparent US research risk proxies, not licensed MSCI USE4 exposures."""
from __future__ import annotations
import numpy as np
import pandas as pd
from usq.utils import zscore


def robust_standardize(x, universe):
    x = x.where(universe.eq(1)).replace([np.inf, -np.inf], np.nan)
    med = x.median(axis=1)
    mad = x.sub(med, axis=0).abs().median(axis=1) * 1.4826
    clean = x.clip(med - 5 * mad, med + 5 * mad, axis=0)
    return zscore(clean)


def build_exposures(store):
    price = store.load("close_adj")
    cap = store.load("market_cap", price)
    universe = store.load("universe", price)
    ret = price / price.shift(1) - 1
    # Market proxy uses lagged signal-date cap; no future cap weighting.
    w = cap.shift(1).where(universe.shift(1).eq(1) & ret.notna())
    w = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    market = (ret * w).sum(axis=1, min_count=1)
    var = market.rolling(252, min_periods=126).var()
    beta = ret.rolling(252, min_periods=126).cov(market).div(var.replace(0, np.nan), axis=0)
    residual = ret - beta.mul(market, axis=0)
    vol = residual.rolling(126, min_periods=60).std()
    size = robust_standardize(np.log(cap.where(cap > 0)), universe)
    cube = size ** 3
    nonlin = cube.copy() * np.nan
    for d in size.index:
        valid = size.loc[d].notna() & cube.loc[d].notna()
        if valid.sum() < 3:
            continue
        s = size.loc[d, valid].to_numpy()
        X = np.column_stack([np.ones(len(s)), s])
        y = cube.loc[d, valid].to_numpy()
        nonlin.loc[d, valid] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    volume = store.load("volume", price)
    rawclose = store.load("close", price)
    liquidity = np.log((volume * rawclose / cap.where(cap > 0)).rolling(21, min_periods=15).mean().replace(0, np.nan))
    exposures = {"Size": size, "Beta": beta,
                 "Momentum": np.log(price.shift(21) / price.shift(252)),
                 "ResidualVolatility": vol, "NonLinearSize": nonlin, "Liquidity": liquidity}
    optional = {
        "BookToPrice": ("book_equity",), "EarningsYield": ("earnings_ttm",),
        "Growth": ("revenue_ttm",), "Leverage": ("total_liabilities", "total_assets")}
    missing = []
    for name, fields in optional.items():
        if not all(store.path(f).exists() for f in fields):
            missing.append(name)
            continue
        a = store.load(fields[0], price)
        if name in ("BookToPrice", "EarningsYield"):
            exposures[name] = a / cap.where(cap > 0)
        elif name == "Growth":
            exposures[name] = (a - a.shift(252)) / a.shift(252).abs().replace(0, np.nan)
        else:
            exposures[name] = a / store.load(fields[1], price).replace(0, np.nan)
    exposures = {k: robust_standardize(v, universe).astype("float32") for k, v in exposures.items()}
    exposures["industry"] = store.load("industry", price, numeric=False)
    return exposures, {"model": "US research proxy v1; NOT MSCI USE4/CNE5 replication",
                       "styles": [k for k in exposures if k != "industry"],
                       "missing_optional_styles": missing,
                       "return_basis_for_attribution": "same adjusted VWAP interval as portfolio",
                       "exposure_timing": "signal date T, explains VWAP T+1 to T+2",
                       "covariance_window": 252, "covariance_min_obs": 60}
