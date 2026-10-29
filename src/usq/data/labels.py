from __future__ import annotations

import numpy as np
import pandas as pd
from .panels import PanelStore
from usq.utils import zscore


def forward_returns(price, horizon=10, delay=1):
    """Signal t close -> enter t+delay VWAP -> exit t+delay+horizon VWAP.

    Require every mark in the window, and retain label_end for exact split purging.
    Prices must use a consistently adjusted basis; no ffill/pct_change defaults.
    """
    if horizon < 1 or delay < 1:
        raise ValueError("horizon and delay must be positive")
    raw = price.shift(-(delay + horizon)) / price.shift(-delay) - 1
    good = pd.DataFrame(True, index=price.index, columns=price.columns)
    for offset in range(delay, delay + horizon + 1):
        good &= price.shift(-offset).gt(0) & price.shift(-offset).notna()
    end = pd.Series(price.index, index=price.index, name="label_end").shift(-(delay + horizon))
    return raw.where(good), end


def neutralize(target, cap, industry, mask, standardize=True, min_obs=20):
    """Daily WLS residual versus log cap + industries, weights=sqrt(cap).

    Controls use SIGNAL-date information. Full industry dummies span the intercept.
    Residuals are not portfolio returns. Missing controls remain missing.
    """
    out = target * np.nan
    for d in target.index:
        valid = mask.loc[d].eq(1) & target.loc[d].notna() & cap.loc[d].gt(0) & industry.loc[d].notna()
        ids = target.columns[valid]
        if len(ids) < min_obs:
            continue
        c = cap.loc[d, ids].to_numpy(float)
        logc = np.log(c)
        X = np.column_stack([logc - logc.mean(), pd.get_dummies(industry.loc[d, ids], dtype=float).to_numpy()])
        if len(ids) <= np.linalg.matrix_rank(X) + 2:
            continue
        y = target.loc[d, ids].to_numpy(float)
        sw = (c / np.median(c)) ** .25
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        out.loc[d, ids] = y - X @ beta
    return zscore(out) if standardize else out


def prepare(store: PanelStore, horizon=10, standardize=True, mode="vwap_compound"):
    if horizon != 10:
        raise ValueError("this project fixes the label horizon at ten trading days")
    close = store.load("close")
    volume = store.load("volume", close)
    if store.path("vwap").exists():
        vwap = store.load("vwap", close)
    else:
        amount = store.load("amount", close)
        vwap = (amount / volume.where(volume > 0)).where(amount > 0)
        store.save("vwap", vwap)
    adj = store.load("adj_factor", close)
    if (adj.dropna() <= 0).any().any():
        raise ValueError("adj_factor must be positive; formula is adjusted_price=raw_price*factor")
    cap = store.load("market_cap", close)
    ind = store.load("industry", close, numeric=False)
    mask = store.load("universe", close).eq(1)
    for field in ("open", "high", "low"):
        raw_field = store.load(field, close)
        store.save(f"{field}_adj", raw_field * adj)
    adjusted = (vwap * adj).where(vwap > 0)
    if mode == "vwap_compound":
        raw, end = forward_returns(adjusted, horizon)
    elif mode == "close_sum":
        # Alternative target convention: sum future daily close returns.
        # Still require all future marks; never reproduce pandas sum(skipna=True).
        cp = close * adj
        daily = cp / cp.shift(1) - 1
        raw = sum(daily.shift(-k) for k in range(1, horizon + 1))
        end = pd.Series(close.index, index=close.index, name="label_end").shift(-horizon)
    else:
        raise ValueError("label mode must be vwap_compound or close_sum")
    label = neutralize(raw, cap, ind, mask, standardize)
    store.save("vwap_adj", adjusted)
    store.save("close_adj", close * adj)
    store.save("forward_10d_raw", raw)
    store.save("label_10", label)
    end.to_csv(store.root / "label_end.csv", index_label="date")
    from usq.utils import json_dump
    json_dump({"mode": mode, "horizon": horizon, "standardize": standardize,
               "neutralization": "signal-date log cap + full industry dummies; WLS weights=sqrt(cap)",
               "price_adjustment": "raw_price * adj_factor"}, store.root / "label_metadata.json")
    return store.manifest()
