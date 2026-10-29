from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def json_dump(value, path):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (np.integer, np.bool_)):
            return x.item()
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, (Path, pd.Timestamp)):
            return str(x)
        return x
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def zscore(frame):
    sd = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(frame.mean(axis=1), axis=0).div(sd, axis=0)


def preprocess(frame, mask):
    x = frame.where(mask).replace([np.inf, -np.inf], np.nan)
    lo, hi = x.quantile(.005, axis=1), x.quantile(.995, axis=1)
    return zscore(x.clip(lo, hi, axis=0)).astype("float32")
