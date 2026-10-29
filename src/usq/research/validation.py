"""Conditional candidate-set null study. Does not claim to rerun adaptive mining."""
import numpy as np
import pandas as pd


def block_resample(array, n_days, rng, block=20):
    """Joint date blocks retain cross-factor dependence; circular within source."""
    a = np.asarray(array, float)
    if a.ndim != 2 or len(a) < block or n_days < 1:
        raise ValueError("need 2D daily IC matrix and enough days for block bootstrap")
    starts = rng.integers(0, len(a), size=int(np.ceil(n_days / block)))
    idx = np.concatenate([(s + np.arange(block)) % len(a) for s in starts])[:n_days]
    return a[idx]


def validation_experiment(is_ic, val_ic, k=20, repeats=500, block=20, seed=2964):
    """All candidate columns, not only surviving factors. No OOS inputs accepted.

    Sign is learned from IS and frozen. Null centers each candidate's joint daily
    ICs and independently samples IS and validation blocks. Reports selection
    optimism under zero skill, not proof that validation is useful/useless.
    """
    if is_ic.empty or val_ic.empty or not is_ic.columns.equals(val_ic.columns):
        raise ValueError("supply all-candidate IS and VAL matrices with identical columns")
    if str(is_ic.index.max()) > "20211231" or str(val_ic.index.max()) > "20221231" or str(val_ic.index.min()) < "20220101":
        raise ValueError("research design uses IS<=2021 and validation=2022 only")
    if not 1 <= k <= len(is_ic.columns) or repeats < 10:
        raise ValueError("invalid top K or too few null repetitions")
    train, val = is_ic.mean(), val_ic.mean()
    direction = np.sign(train).replace(0, 1)
    train, val = train * direction, val * direction
    is_top = train.nlargest(k).index
    # Matched budget: select K by VAL from a fixed 3K IS shortlist.
    shortlist = train.nlargest(min(len(train), 3 * k)).index
    validate_top = val.loc[shortlist].nlargest(k).index
    observed = {"n_candidates": len(train), "K": k,
                "is_val_spearman": float(train.corr(val, method="spearman")),
                "val_mean_of_is_topk": float(val.loc[is_top].mean()),
                "val_mean_after_val_selection": float(val.loc[validate_top].mean()),
                "warning": "last metric reuses validation for selection and is optimistically biased; not OOS evidence"}
    pooled = pd.concat([is_ic, val_ic]).to_numpy(float)
    means = np.nanmean(pooled, axis=0)
    centered = pooled - means
    rng = np.random.default_rng(seed)
    null = []
    for rep in range(repeats):
        a = np.nanmean(block_resample(centered, len(is_ic), rng, block), axis=0)
        b = np.nanmean(block_resample(centered, len(val_ic), rng, block), axis=0)
        sign = np.where(a >= 0, 1, -1)
        a, b = a * sign, b * sign
        ids = np.argsort(np.nan_to_num(a, nan=-np.inf))[::-1][:k]
        short = np.argsort(np.nan_to_num(a, nan=-np.inf))[::-1][:min(len(a), 3 * k)]
        selected = short[np.argsort(np.nan_to_num(b[short], nan=-np.inf))[::-1][:k]]
        null.append({"repeat": rep, "val_mean_of_is_topk": float(np.nanmean(b[ids])),
                     "val_mean_after_val_selection": float(np.nanmean(b[selected])),
                     "is_val_spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman"))})
    null = pd.DataFrame(null)
    for key in ("val_mean_of_is_topk", "is_val_spearman"):
        observed[f"{key}_conditional_null_p"] = float((1 + (null[key] >= observed[key]).sum()) / (repeats + 1))
    observed.update({"seed": seed, "block": block, "repeats": repeats,
                     "scope": "conditional candidate-set bootstrap, not full adaptive-search null",
                     "is_only_selection": is_top.tolist(), "is_plus_val_selection": validate_top.tolist()})
    return observed, null
