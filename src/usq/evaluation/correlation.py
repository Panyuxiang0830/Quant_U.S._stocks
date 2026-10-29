from __future__ import annotations
import numpy as np
import pandas as pd


def evaluate_correlation(factors, quality, universe, date_step=10, stock_fraction=.25,
                         threshold=.7, min_pairs=100, seed=2964):
    """Layer 3. Daily ranks, deterministic date/stock downsample, pairwise Pearson.

    Streaming callers can load only sampled panels. Correlation NaN is rejected
    conservatively, not interpreted as independence. quality must be IS-only.
    """
    if not factors or date_step < 1 or not 0 < stock_fraction <= 1 or not 0 <= threshold <= 1:
        raise ValueError("invalid sampling/threshold or empty factor set")
    ref = next(iter(factors.values()))
    rng = np.random.default_rng(seed)
    cols = np.sort(rng.choice(ref.shape[1], max(1, int(ref.shape[1] * stock_fraction)), replace=False))
    dates = ref.index[::date_step]
    ids = ref.columns[cols]
    sampled = {}
    for name, f in factors.items():
        x = f.reindex(index=dates, columns=ids).where(universe.reindex(index=dates, columns=ids).eq(1))
        # Rank/standardize within each sampled date, preventing cross-day level effects.
        x = x.rank(axis=1, pct=True)
        x = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1).replace(0, np.nan), axis=0)
        sampled[name] = x.to_numpy().ravel()
    sample = pd.DataFrame(sampled)
    corr = sample.corr(method="pearson", min_periods=min_pairs)
    overlap = sample.notna().astype("int64").T @ sample.notna().astype("int64")
    order = sorted(factors, key=lambda n: (-abs(float(quality.get(n, 0))), n))
    kept, audit = [], []
    for name in order:
        if not np.isfinite(quality.get(name, np.nan)) or not np.isfinite(corr.loc[name, name]):
            audit.append({"factor": name, "kept": False, "reason": "invalid quality or insufficient variable observations"})
            continue
        conflict = next((k for k in kept if pd.isna(corr.loc[name, k]) or abs(corr.loc[name, k]) >= threshold), None)
        audit.append({"factor": name, "kept": conflict is None, "conflict": conflict,
                      "correlation": float(corr.loc[name, conflict]) if conflict else None})
        if conflict is None:
            kept.append(name)
    meta = {"dates": dates.tolist(), "sids": ids.tolist(), "seed": seed,
            "date_step": date_step, "stock_fraction": stock_fraction, "threshold_abs": threshold,
            "min_pairs": min_pairs, "kept": kept, "audit": audit}
    return corr, overlap, meta
