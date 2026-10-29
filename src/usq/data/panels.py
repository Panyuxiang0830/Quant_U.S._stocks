"""gpushare-compatible trusted pickle wide panels: date strings x security IDs."""
from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd
from usq.utils import sha256, json_dump


def date_index(index):
    s = pd.Index(index).astype(str).str.replace("-", "", regex=False)
    if not s.str.fullmatch(r"\d{8}").all():
        # Permit a real DatetimeIndex, but do not guess arbitrary source strings.
        if isinstance(index, pd.DatetimeIndex):
            s = pd.Index(index.strftime("%Y%m%d"))
        else:
            raise ValueError("dates must be YYYYMMDD, YYYY-MM-DD or DatetimeIndex")
    pd.to_datetime(s, format="%Y%m%d", errors="raise")
    return pd.Index(s, name="date")


def validate_panel(frame, reference=None, numeric=True):
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("panel must be a non-empty DataFrame")
    out = frame.copy()
    out.index = date_index(out.index)
    out.columns = out.columns.astype(str).rename("sid")
    if out.index.has_duplicates or out.columns.has_duplicates:
        raise ValueError("duplicate date or sid")
    if not out.index.is_monotonic_increasing:
        raise ValueError("dates must be sorted; supply the complete exchange calendar")
    if reference is not None and (not out.index.equals(reference.index) or not out.columns.equals(reference.columns)):
        raise ValueError("all panels must have identical calendar and ordered security IDs")
    if numeric:
        out = out.astype(float)
        if np.isinf(out.to_numpy()).any():
            raise ValueError("infinity in input panel")
    return out


class PanelStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("unsafe panel name")
        return self.root / f"{name}.pkl"

    def load(self, name, reference=None, numeric=True):
        # Pickle is used for gpushare compatibility: only load trusted team artifacts.
        return validate_panel(pd.read_pickle(self.path(name)), reference, numeric)

    def save(self, name, frame, numeric=True):
        self.root.mkdir(parents=True, exist_ok=True)
        frame = validate_panel(frame, numeric=numeric)
        if numeric:
            frame = frame.astype("float32")
        temp = self.path(name).with_suffix(".tmp")
        frame.to_pickle(temp, protocol=4)
        temp.replace(self.path(name))

    def manifest(self):
        result = {}
        for p in sorted(self.root.glob("*.pkl")):
            x = self.load(p.stem, numeric=p.stem != "industry")
            result[p.name] = {"sha256": sha256(p), "shape": list(x.shape),
                              "start": x.index[0], "end": x.index[-1],
                              "coverage": float(x.notna().to_numpy().mean())}
        json_dump(result, self.root / "manifest.json")
        return result


def compare_sources(primary, secondary, rtol=.001, atol=.01):
    """Compare ONLY after identity, units, session and adjustment basis are matched."""
    a = validate_panel(primary)
    b = validate_panel(secondary).reindex(index=a.index, columns=a.columns)
    shared = a.notna() & b.notna()
    diff = (a - b).abs()
    bad = shared & (diff > atol + rtol * b.abs())
    return {"primary_observations": int(a.notna().sum().sum()),
            "matched_observations": int(shared.sum().sum()),
            "mismatch_count": int(bad.sum().sum()),
            "mismatch_rate": float(bad.sum().sum() / max(1, shared.sum().sum())),
            "secondary_missing_on_primary": int((a.notna() & b.isna()).sum().sum())}, diff.where(bad)


def pit_panel(records, dates, sids):
    """One field per call. records: sid, available_date, value (after session cutoff).

    available_date is the first signal date on which the released vintage is usable.
    Never replace earlier vintages with a later restatement. No backward fill.
    """
    rec = records.copy()
    required = {"sid", "available_date", "value"}
    if not required.issubset(rec):
        raise ValueError(f"required: {required}")
    rec["sid"] = rec.sid.astype(str)
    rec["available_date"] = date_index(rec.available_date).to_numpy()
    if rec.duplicated(["sid", "available_date"]).any():
        raise ValueError("resolve same-day vintages explicitly before PIT expansion")
    ix = date_index(dates)
    union = ix.union(pd.Index(rec.available_date)).sort_values()
    wide = rec.pivot(index="available_date", columns="sid", values="value")
    return wide.reindex(index=union, columns=pd.Index(sids).astype(str)).ffill().reindex(ix)
