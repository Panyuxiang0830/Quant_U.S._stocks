"""由细项构建 10 个 CNE5 风格因子暴露。"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import config
from .preprocess import combine, orthogonalize, standardize, winsorize_mad


def style_exposures(descriptors: dict[str, pd.DataFrame], cap: pd.DataFrame,
                    universe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """descriptors: 细项名 -> 日期×股票。cap/universe 需与之对齐。

    流程：细项去极值+标准化 -> 按权重合成 -> 因子去极值+标准化 -> 正交化后再标准化。
    """
    dates, cols = next(iter(descriptors.values())).index, next(iter(descriptors.values())).columns
    cap = cap.reindex(index=dates, columns=cols)
    universe = universe.reindex(index=dates, columns=cols).fillna(False).astype(bool)

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        return standardize(winsorize_mad(df, universe), cap, universe)

    ortho_w = cap ** config.ORTHO_CAP_POWER
    std_desc = {k: clean(v) for k, v in descriptors.items()}
    factors: dict[str, pd.DataFrame] = {}
    for name, spec in config.STYLE_SPEC.items():
        parts = [(std_desc[d], w) for d, w in spec if d in std_desc]
        if not parts:
            warnings.warn(f"{name}: 没有可用细项，跳过", stacklevel=2)
            continue
        factors[name] = clean(combine(parts))
    for name, base in config.ORTHOGONALIZE.items():
        if name in factors and all(b in factors for b in base):
            resid = orthogonalize(factors[name], [factors[b] for b in base], ortho_w, universe)
            factors[name] = clean(resid)
    if "Size" in factors:
        cube = factors["Size"] ** 3
        factors["NonLinearSize"] = clean(orthogonalize(cube, [factors["Size"]], ortho_w, universe))
    return {k: factors[k] for k in config.STYLE_FACTORS if k in factors}


def to_long(factors: dict[str, pd.DataFrame], industry: pd.DataFrame, cap: pd.DataFrame,
            universe: pd.DataFrame) -> pd.DataFrame:
    """合成长表：index=(trade_date, ts_code)，列=风格因子 + industry/cap/estu。"""
    dates, cols = next(iter(factors.values())).index, next(iter(factors.values())).columns
    frame = {k: v.stack(future_stack=True) for k, v in factors.items()}
    frame["industry"] = industry.reindex(index=dates, columns=cols).stack(future_stack=True)
    frame["cap"] = cap.reindex(index=dates, columns=cols).stack(future_stack=True)
    frame["estu"] = universe.reindex(index=dates, columns=cols).fillna(False).astype(bool).stack(
        future_stack=True)
    out = pd.DataFrame(frame)
    out.index.names = ["trade_date", "ts_code"]
    out = out[out["industry"].notna() | out[list(factors)].notna().any(axis=1)]
    styles = list(factors)
    out[styles] = out[styles].astype(np.float32)
    return out
