"""Evaluate a full externally computed 101/191 library without pretending examples
are a complete canonical implementation. Every expected ID has an audit status.
"""
from pathlib import Path
import shutil
from usq.data import PanelStore
from usq.utils import json_dump, sha256


def import_library(source_dir, target_dir, family, reference, require_complete=False):
    if family not in ("wq101", "gtja191"):
        raise ValueError("family must be wq101 or gtja191")
    count = 101 if family == "wq101" else 191
    source = PanelStore(source_dir); target = PanelStore(target_dir)
    target.root.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in range(1, count + 1):
        name = f"{family}_{n:03d}"
        p = source.path(name)
        if not p.exists():
            rows.append({"name": name, "status": "missing_external_formula_output"})
            continue
        x = source.load(name, reference)
        if not x.notna().any().any():
            rows.append({"name": name, "status": "all_nan"}); continue
        target.save(name, x)
        rows.append({"name": name, "status": "imported", "source_sha256": sha256(p)})
    json_dump(rows, target.root / f"{family}_import_audit.json")
    if require_complete and any(r["status"] != "imported" for r in rows):
        raise ValueError("library incomplete; inspect import audit, do not claim 101/191 complete")
    return rows
