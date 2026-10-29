from __future__ import annotations
import hashlib
import heapq
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from .expressions import compute, enumerate_expressions
from usq.evaluation.simple import evaluate_simple
from usq.data import PanelStore
from usq.utils import json_dump, sha256


def search(store, out_dir, fields=("close", "volume", "returns", "vwap"), windows=(5, 10, 20, 60),
           seeds=(), max_trials=5000, top_k=100, min_abs_rankic=.01,
           min_coverage=.6, min_stocks=30, seed=2964):
    """Evaluate <=2021 only, retain bounded top K, journal ALL trials in SQLite.

    Resume requires identical data hashes and settings. Trial metrics and daily
    ICs survive resume; discarded full factor matrices do not consume disk.
    """
    if max_trials < 1 or top_k < 1:
        raise ValueError("positive max_trials/top_k required")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    factors = PanelStore(out / "factors")
    label = store.load("label_10")
    mask = store.load("universe", label).eq(1)
    label_end = pd.read_csv(store.root / "label_end.csv", dtype=str, index_col=0).iloc[:, 0].reindex(label.index)
    is_days = (label.index >= "20150101") & (label.index <= "20211231") & label_end.lt("20220101")
    # Load only fields demanded by the grammar; trailing calculations use warmup.
    panel = {}
    for name in fields:
        if name == "returns":
            close = store.load("close_adj", label)
            panel[name] = close / close.shift(1) - 1
        else:
            source = f"{name}_adj" if name in ("open", "high", "low", "close", "vwap") and store.path(f"{name}_adj").exists() else name
            panel[name] = store.load(source, label)
    metadata = {"fields": list(fields), "windows": list(windows), "seeds": list(seeds),
                "max_trials": max_trials, "top_k": top_k, "min_abs_rankic": min_abs_rankic,
                "min_coverage": min_coverage, "min_stocks": min_stocks, "seed": seed,
                "data_hashes": {p.name: sha256(p) for p in sorted(store.root.glob("*.pkl"))},
                "period": "2015-2021; label_end<20220101"}
    settings = out / "search_config.json"
    if settings.exists() and json.loads(settings.read_text()) != metadata:
        raise ValueError("resume settings/data differ; use a new output directory")
    json_dump(metadata, settings)
    db = sqlite3.connect(out / "trials.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS trials (name TEXT PRIMARY KEY, expression TEXT, status TEXT, quality REAL, summary TEXT)")
    daily_dir = out / "trial_rankic"; daily_dir.mkdir(exist_ok=True)
    heap = []
    for name, quality in db.execute("SELECT name,quality FROM trials WHERE status='ok' AND quality IS NOT NULL ORDER BY quality DESC"):
        if factors.path(name).exists(): heapq.heappush(heap, (quality, name))
    expressions = enumerate_expressions(fields, windows, seeds)
    for num, expression in enumerate(expressions):
        if num >= max_trials: break
        name = "f_" + hashlib.sha256(expression.encode()).hexdigest()[:16]
        if db.execute("SELECT 1 FROM trials WHERE name=?", (name,)).fetchone(): continue
        try:
            factor = compute(expression, panel, mask)
            summary, daily = evaluate_simple(factor.loc[is_days], label.loc[is_days], mask.loc[is_days], min_stocks=min_stocks)
            q = summary["rankic_mean"]
            summary["expression"] = expression
            eligible = np.isfinite(q) and q >= min_abs_rankic and summary["coverage"] >= min_coverage
            # The ledger always records low-quality successes too, avoiding survivor-only research.
            db.execute("INSERT INTO trials VALUES (?,?,?,?,?)", (name, expression, "ok", float(q) if np.isfinite(q) else None, json.dumps(summary)))
            daily[["rankic"]].to_parquet(daily_dir / f"{name}.parquet")
            if eligible and (len(heap) < top_k or q > heap[0][0]):
                factors.save(name, factor)
                heapq.heappush(heap, (q, name))
                if len(heap) > top_k:
                    _, removed = heapq.heappop(heap)
                    factors.path(removed).unlink(missing_ok=True)
        except (ValueError, KeyError, FloatingPointError) as e:
            db.execute("INSERT INTO trials VALUES (?,?,?,?,?)", (name, expression, "error", None, json.dumps({"error": str(e)})))
        db.commit()
    entries = []
    for q, name in sorted(heap, reverse=True):
        row = db.execute("SELECT expression,summary FROM trials WHERE name=?", (name,)).fetchone()
        s = json.loads(row[1])
        entries.append({"name": name, "direction": s["direction"], "is_rankic": q, "expression": row[0]})
    count = db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    db.close()
    json_dump({"frozen_through": "20211231", "factors": entries, "trials": count}, out / "candidates.json")
    return entries
