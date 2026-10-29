from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from usq.backtest.engine import run_backtest, group_backtests, score_groups


def evaluate_complex(factor, vwap_adj, universe, direction, groups=10,
                     cost_bps=5, barra_context=None, **kwargs):
    """Layer 2. Fixed IS direction; executable one-day portfolios, not label PnL.

    Optional barra_context contains a persisted model; when absent the
    result explicitly marks Barra as unavailable. CLI --barra computes it.
    """
    if direction not in (-1, 1):
        raise ValueError("pass the direction frozen on the mining period")
    signal = factor * direction
    bt = run_backtest(signal, vwap_adj, universe, mode="long_short", cost_bps=cost_bps, **kwargs)
    group_returns = group_backtests(signal, vwap_adj, universe, groups=groups, cost_bps=cost_bps, **kwargs)
    means = group_returns.mean()
    good = means.notna()
    monotonicity = float(spearmanr(np.arange(1, groups + 1)[good], means[good]).statistic) if good.sum() > 2 else np.nan
    summary = {**bt.summary, "direction": direction, "group_monotonicity_spearman": monotonicity,
               "adjacent_increasing_rate": float((means.diff().dropna() > 0).mean()),
               "barra_status": "not_requested"}
    attribution = None
    if barra_context is not None:
        from usq.barra import analyze
        attribution = analyze(bt, vwap_adj, universe, **barra_context)
        summary["barra_status"] = "computed_reference_model"
    return summary, bt, group_returns, attribution
