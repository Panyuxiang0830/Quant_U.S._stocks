"""Daily target-weight research simulator, T signal -> T+1/T+2 adjusted VWAP.

Return rows carry EXIT dates. Target weights chosen at T earn r[T+2]. No
future return or future tradability enters stock selection. Missing held returns
are fatal. Cost is bps per dollar bought/sold; this is not an account simulator.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .metrics import performance


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    weights: pd.DataFrame  # beginning of interval, indexed by exit date
    summary: dict


def score_groups(score, universe, groups=50):
    """Tie-aware percentile bins. Equal signals stay together; empty bins allowed."""
    if groups < 2:
        raise ValueError("groups >= 2 required")
    x = score.where(universe.eq(1))
    rank = x.rank(axis=1, method="average")
    count = x.count(axis=1)
    pct = rank.sub(.5).div(count.replace(0, np.nan), axis=0)
    return (np.floor(pct * groups) + 1).clip(1, groups).where(x.notna())


def target_weights(score, universe, top_n=100, mode="long", group=None, groups=50):
    x = score.where(universe.eq(1)).replace([np.inf, -np.inf], np.nan)
    if group is not None:
        chosen = score_groups(x, universe, groups).eq(group)
        return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if top_n < 1:
        raise ValueError("top_n >= 1 required")
    if mode == "long":
        # Stable sid order for ties, recorded in the input schema.
        chosen = x.rank(axis=1, ascending=False, method="first").le(top_n) & x.notna()
        return chosen.div(chosen.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if mode == "long_short":
        r = x.rank(axis=1, pct=True)
        r = r.sub(r.mean(axis=1), axis=0)
        return r.div(r.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raise ValueError("mode must be long or long_short")


def run_backtest(score, vwap_adj, universe, top_n=100, mode="long", cost_bps=5.,
                 borrow_bps_annual=300., group=None, groups=50, benchmark_returns=None,
                 start=None, end=None):
    if cost_bps < 0 or borrow_bps_annual < 0:
        raise ValueError("costs cannot be negative")
    cal = vwap_adj.index
    if not score.columns.equals(vwap_adj.columns) or not universe.index.equals(cal) or not universe.columns.equals(vwap_adj.columns):
        raise ValueError("align ordered sids and full calendar before backtesting")
    if not score.index.isin(cal).all():
        raise ValueError("score contains dates outside the price calendar")
    x = score.reindex(cal)
    active_signal = x.notna().any(axis=1)
    if active_signal.any():
        observed = np.flatnonzero(active_signal.to_numpy())
        if not active_signal.iloc[observed[0]:observed[-1] + 1].all():
            raise ValueError("internal missing score day; explicit cash/hold policy is required")
    target = target_weights(x, universe, top_n, mode, group, groups)
    interval = vwap_adj / vwap_adj.shift(1) - 1
    weights = target.shift(2).fillna(0)
    # Explicitly distinguish a zero-signal CASH day from a day without a prediction.
    has_signal = active_signal.shift(2, fill_value=False)
    want = has_signal & (cal >= (start or cal[0])) & (cal <= (end or cal[-1]))
    weights = weights.loc[want]
    ret = interval.loc[want]
    if weights.empty:
        raise ValueError("no score intervals in requested window")
    missing = weights.ne(0) & (ret.isna() | ~np.isfinite(ret))
    if missing.any().any():
        d, sid = np.argwhere(missing.to_numpy())[0]
        raise ValueError(f"missing held return: {weights.index[d]} {weights.columns[sid]}; supply corporate-action/delisting marks")
    gross = (weights * ret.fillna(0)).sum(axis=1)
    # Compare new target against previous holdings drifted by the preceding interval.
    previous = weights.shift(1).fillna(0)
    previous_gross = gross.shift(1).fillna(0)
    drift = previous.mul(1 + ret.shift(1).fillna(0)).div(1 + previous_gross, axis=0)
    traded = (weights - drift).abs().sum(axis=1)
    turn = traded / 2  # one-way turnover; initial all-cash entry has 0.5 turnover.
    cost = traded * cost_bps / 1e4
    borrow = weights.clip(upper=0).abs().sum(axis=1) * borrow_bps_annual / 1e4 / 252
    net = gross - cost - borrow
    if (net <= -1).any():
        raise ValueError("portfolio NAV exhausted")
    daily = pd.DataFrame({"gross_return": gross, "cost": cost, "borrow_cost": borrow,
                          "net_return": net, "turnover": turn, "gross_exposure": weights.abs().sum(axis=1)})
    for name in ("gross", "net"):
        daily[f"{name}_nav"] = (1 + daily[f"{name}_return"]).cumprod()
    daily["drawdown"] = daily.net_nav / daily.net_nav.cummax().clip(lower=1) - 1
    if benchmark_returns is not None:
        b = benchmark_returns.reindex(daily.index)
        if b.isna().any():
            raise ValueError("benchmark missing on a return date")
        daily["benchmark_return"] = b
        daily["benchmark_nav"] = (1 + b).cumprod()
        daily["relative_nav"] = daily.net_nav / daily.benchmark_nav
    summary = {"net": performance(net, turn), "gross": performance(gross, turn),
               "timing": "T close signal; T+1 VWAP entry; T+2 VWAP exit; rows=exit date",
               "mode": mode, "cost_bps_per_traded_dollar": cost_bps,
               "borrow_bps_annual": borrow_bps_annual,
               "return_basis": "adjusted-VWAP price return; dividends only if embedded in adj_factor"}
    return BacktestResult(daily, weights, summary)


def group_backtests(score, vwap_adj, universe, groups=50, **kwargs):
    series = {}
    counts = score_groups(score.reindex(vwap_adj.index), universe, groups)
    for g in range(1, groups + 1):
        if not counts.eq(g).any().any():
            series[f"G{g:02d}"] = pd.Series(dtype=float)
            continue
        result = run_backtest(score, vwap_adj, universe, group=g, groups=groups, **kwargs)
        # A zero-stock group is undefined rather than a synthetic zero-return group.
        series[f"G{g:02d}"] = result.daily.net_return.where(result.daily.gross_exposure > 0)
    return pd.DataFrame(series)
