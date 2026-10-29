"""User-defined CV: daily RankIC divided by sign-selected past-two-year extreme.

This is NOT coefficient of variation and NOT model cross-validation.
"""
import numpy as np
import pandas as pd
from collections import deque


def direction_cv(rankic, years=2, min_obs=252):
    s = pd.Series(rankic).astype(float)
    dates = pd.to_datetime(s.index.astype(str), format="%Y%m%d")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("RankIC dates must be unique and sorted")
    lower_dates = dates - pd.DateOffset(years=years)
    valid_count = np.r_[0, np.cumsum(s.notna().to_numpy())]
    minimum, maximum = deque(), deque()
    rows = []
    for i, (day, value) in enumerate(zip(dates, s)):
        lower = lower_dates[i]
        left = int(dates.searchsorted(lower))
        # Insert yesterday, never current day. Monotone queues make this O(T).
        if i and np.isfinite(s.iloc[i - 1]):
            previous = float(s.iloc[i - 1])
            while minimum and minimum[-1][1] >= previous: minimum.pop()
            while maximum and maximum[-1][1] <= previous: maximum.pop()
            minimum.append((i - 1, previous)); maximum.append((i - 1, previous))
        while minimum and minimum[0][0] < left: minimum.popleft()
        while maximum and maximum[0][0] < left: maximum.popleft()
        count = int(valid_count[i] - valid_count[left])
        complete = bool(len(dates) and dates[0] <= lower and count >= min_obs)
        lo = minimum[0][1] if minimum else np.nan
        hi = maximum[0][1] if maximum else np.nan
        denom = lo if value > 0 else hi if value < 0 else np.nan
        valid = complete and np.isfinite(value) and np.isfinite(denom) and abs(denom) > 1e-12
        rows.append({"rankic": value, "history_min": lo, "history_max": hi,
                     "denominator": denom, "history_count": count,
                     "full_two_year_history": complete,
                     "cv": float(value / denom) if valid else np.nan})
    return pd.DataFrame(rows, index=s.index)
