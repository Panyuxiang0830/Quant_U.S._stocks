from __future__ import annotations
import numpy as np
import pandas as pd
from .direction_cv import direction_cv


def evaluate_simple(factor, label, universe, min_stocks=30, direction=None, horizon=10, rankic_history=None):
    """Layer 1. direction=None learns a sign ONLY on a caller-supplied IS slice.

    CV is the user's sign/extreme ratio on daily RankIC over the PAST two years.
    Optional rankic_history is already oriented with the same fixed IS sign.
    """
    if not factor.index.equals(label.index) or not factor.columns.equals(label.columns):
        raise ValueError("factor and label axes must match")
    valid = universe.eq(1) & factor.notna() & label.notna() & np.isfinite(factor) & np.isfinite(label)
    f, y = factor.where(valid), label.where(valid)
    count = valid.sum(axis=1)
    daily = pd.DataFrame({"ic": f.corrwith(y, axis=1),
                          "rankic": f.rank(axis=1).corrwith(y.rank(axis=1), axis=1),
                          "n_stocks": count})
    daily.loc[count < min_stocks, ["ic", "rankic"]] = np.nan
    raw_mean = daily.rankic.mean()
    if direction is None:
        direction = 1 if pd.isna(raw_mean) or raw_mean >= 0 else -1
    if direction not in (-1, 1):
        raise ValueError("direction must be fixed at +1 or -1")
    daily[["ic", "rankic"]] *= direction
    summary = {"direction": direction, "days": len(daily), "valid_ic_days": int(daily.ic.count()),
               "coverage": float(valid.to_numpy().sum() / max(1, (universe.eq(1) & label.notna()).to_numpy().sum()))}
    for key in ("ic", "rankic"):
        mu, sd = daily[key].mean(), daily[key].std(ddof=1)
        summary[f"{key}_mean"] = float(mu)
        summary[f"{key}_std"] = float(sd)
        summary[f"{key}ir"] = float(mu / sd) if sd > 0 else np.nan
        summary[f"{key}_positive_rate"] = float((daily[key].dropna() > 0).mean())
    history = daily.rankic if rankic_history is None else pd.concat([rankic_history, daily.rankic]).sort_index()
    history = history[~history.index.duplicated(keep="last")]
    cv = direction_cv(history).reindex(daily.index)
    daily["cv"] = cv.cv
    daily["cv_history_min"] = cv.history_min
    daily["cv_history_max"] = cv.history_max
    daily["cv_denominator"] = cv.denominator
    daily["cv_history_count"] = cv.history_count
    summary["cv_mean"] = float(daily.cv.mean())
    summary["cv_valid_days"] = int(daily.cv.count())
    summary["cv_negative_rate"] = float((daily.cv.dropna() < 0).mean())
    summary["cv_definition"] = "daily RankIC>0 / past-two-calendar-year minimum; RankIC<0 / maximum; exclude current day"
    # These are diagnostics, with no refitting/reselection. Drop horizon days at
    # each right boundary to avoid label windows crossing the next block.
    blocks = np.array_split(np.arange(len(daily)), 4)
    summary["temporal_blocks"] = []
    for b in blocks:
        keep = b[:max(0, len(b) - horizon - 1)]
        summary["temporal_blocks"].append({"start": str(daily.index[b[0]]) if len(b) else None,
            "end": str(daily.index[b[-1]]) if len(b) else None,
            "rankic_mean": float(daily.iloc[keep].rankic.mean()), "valid_days": int(daily.iloc[keep].rankic.count())})
    return summary, daily
