"""Recursive engine + batch process orchestration + the project's evaluation API."""
from __future__ import annotations
import json
import multiprocessing as mp
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .recursive import ENGINE_VERSION, compute, load_panels, generate_candidates, engine_hashes
from usq.evaluation.simple import evaluate_simple
from usq.data import PanelStore
from usq.utils import json_dump, sha256

_STATE = {}


def _initialize_worker(panels, label, mask, days, min_stocks):
    _STATE.clear()
    _STATE.update(panels=panels, label=label, mask=mask, days=days, min_stocks=min_stocks)


def _evaluate_candidate(item):
    try:
        s = _STATE
        f = compute(item["expression"], s["panels"], s["mask"])
        summary, daily = evaluate_simple(f.loc[s["days"]], s["label"].loc[s["days"]],
                                        s["mask"].loc[s["days"]], min_stocks=s["min_stocks"])
        raw = f.loc[s["days"]].where(s["mask"].loc[s["days"]])
        count = max(1, s["mask"].loc[s["days"]].to_numpy().sum())
        summary["inf_pct"] = float(np.isinf(raw.to_numpy()).sum() / count)
        summary["na_pct"] = float((raw.isna() & s["mask"].loc[s["days"]]).to_numpy().sum() / count)
        summary.update(source_name=item["source_name"], dimension=item["dimension"], engine=ENGINE_VERSION)
        return item, summary, daily[["rankic"]], None
    except Exception as e:
        return item, None, None, f"{type(e).__name__}: {e}"


def _eligible(summary, min_rankic, min_coverage):
    q = summary.get("rankic_mean")
    return q is not None and np.isfinite(q) and q >= min_rankic and summary["coverage"] >= min_coverage


def search(store, out_dir, fields=("open", "high", "low", "close", "volume", "returns", "vwap"),
           windows=(5, 10, 20, 60), time_periods=(1, 5, 10), seeds=(),
           complexity=3, dimension_filter=("unitless", "percentage"),
           required_fields=(),
           max_workers=1, batch_size=64, start_method="auto", operators=None,
           additional_field_dimensions=None, seed_factor_dir=None, seed_factors=(),
           max_trials=5000, top_k=100, min_abs_rankic=.01, min_coverage=.6,
           min_stocks=30, seed=2964):
    """Same recursive generator and Operators, with US inputs and new evaluation.

    Only <=2021 data enters candidate evaluation workers. Retained factor values
    are then recomputed over the full calendar for fixed-rule validation/modeling.
    Resume replays journaled eligibility and never requires discarded matrices.
    """
    if min(max_trials, top_k, max_workers, batch_size) < 1:
        raise ValueError("positive budgets, workers and batch_size required")
    if start_method == "auto": start_method = "fork" if sys.platform.startswith("linux") else "spawn"
    if start_method not in mp.get_all_start_methods(): raise ValueError("unsupported process start method")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fs = PanelStore(out / "factors")
    if seed_factor_dir and Path(seed_factor_dir).resolve() == fs.root.resolve():
        raise ValueError("seed inputs and retained-factor outputs must use different directories")
    label = store.load("label_10")
    mask = store.load("universe", label).eq(1)
    end = pd.read_csv(store.root / "label_end.csv", dtype=str, index_col=0).iloc[:, 0].reindex(label.index)
    days = np.asarray((label.index >= "20150101") & (label.index <= "20211231") & end.notna() & end.lt("20220101"))
    panels, dimensions = load_panels(store, label, fields, seeds, seed_factor_dir, seed_factors, additional_field_dimensions)
    settings = {"engine": ENGINE_VERSION, "engine_sha256": engine_hashes(), "fields": list(fields),
                "windows": list(windows), "time_periods": list(time_periods), "complexity": complexity,
                "dimension_filter": list(dimension_filter), "operators": operators,
                "required_fields": list(required_fields),
                "seeds": list(seeds), "seed_factors": list(seed_factors), "seed_factor_dir": str(seed_factor_dir) if seed_factor_dir else None,
                "additional_field_dimensions": additional_field_dimensions or {}, "resolved_dimensions": dimensions,
                "max_workers": max_workers, "batch_size": batch_size, "start_method": start_method,
                "max_trials": max_trials, "top_k": top_k, "min_abs_rankic": min_abs_rankic,
                "min_coverage": min_coverage, "min_stocks": min_stocks, "seed": seed,
                "data_hashes": {p.name: sha256(p) for p in sorted(store.root.glob("*.pkl"))},
                "label_end_sha256": sha256(store.root / "label_end.csv"),
                "seed_factor_hashes": {x["name"]: sha256(PanelStore(seed_factor_dir).path(x["name"])) for x in seed_factors},
                "period": "2015-2021; label_end<20220101"}
    path = out / "search_config.json"
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError("resume engine/settings/data differ; use a new output directory")
    json_dump(settings, path)
    catalog, gen_meta = generate_candidates(list(panels), dimensions, windows, time_periods,
        complexity, dimension_filter, max_trials, seed, operators, required_fields)
    json_dump({"engine": ENGINE_VERSION, **gen_meta, "factors": catalog}, out / "expression_catalog.json")
    # Identical expressions can have multiple source names. Evaluate each unique
    # tree once, matching the source-name first-occurrence ordering.
    unique = {}
    for item in catalog: unique.setdefault(item["name"], item)
    catalog = list(unique.values())
    db = sqlite3.connect(out / "trials.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS trials (name TEXT PRIMARY KEY, expression TEXT, status TEXT, quality REAL, summary TEXT)")
    done = {r[0] for r in db.execute("SELECT name FROM trials")}
    pending = [x for x in catalog if x["name"] not in done]
    train = label.index <= "20211231"
    train_panels = {n: p.loc[train] for n, p in panels.items()}
    initargs = (train_panels, label.loc[train], mask.loc[train], days[train], min_stocks)
    daily_dir = out / "trial_rankic"; daily_dir.mkdir(exist_ok=True)
    csv_dir = out / "batches"; csv_dir.mkdir(exist_ok=True)
    next_batch = len(list(csv_dir.glob("factor_evaluation_report_id_*.csv")))
    pool = mp.get_context(start_method).Pool(max_workers, initializer=_initialize_worker, initargs=initargs) if max_workers > 1 and pending else None
    if pool is None: _initialize_worker(*initargs)
    try:
        for lo in range(0, len(pending), batch_size):
            batch = pending[lo:lo + batch_size]
            results = pool.imap(_evaluate_candidate, batch, chunksize=1) if pool else map(_evaluate_candidate, batch)
            rows = []
            for item, summary, daily, error in results:
                if error:
                    db.execute("INSERT INTO trials VALUES (?,?,?,?,?)", (item["name"], item["expression"], "error", None, json.dumps({"error": error})))
                    rows.append({**item, "status": "error", "error": error})
                else:
                    q = summary["rankic_mean"]
                    db.execute("INSERT INTO trials VALUES (?,?,?,?,?)", (item["name"], item["expression"], "ok", float(q) if np.isfinite(q) else None, json.dumps(summary)))
                    daily.to_parquet(daily_dir / f"{item['name']}.parquet")
                    rows.append({**item, "status": "ok", **{k: v for k, v in summary.items() if not isinstance(v, (dict, list))}})
                db.commit()
            pd.DataFrame(rows).to_csv(csv_dir / f"factor_evaluation_report_id_{next_batch:05d}.csv", index=False)
            next_batch += 1
    finally:
        if pool: pool.close(); pool.join()
        _STATE.clear()
    # Choose from ALL successful eligible ledger rows, including pre-interruption
    # trials whose full factor file had not yet been materialized.
    selected = []
    for name, expr, serialized in db.execute("SELECT name,expression,summary FROM trials WHERE status='ok' ORDER BY quality DESC,name"):
        summary = json.loads(serialized)
        if not _eligible(summary, min_abs_rankic, min_coverage): continue
        selected.append({"name": name, "expression": json.loads(expr), "direction": summary["direction"],
                         "is_rankic": summary["rankic_mean"], "dimension": summary["dimension"],
                         "source_name": summary["source_name"], "engine": ENGINE_VERSION})
        if len(selected) >= top_k: break
    chosen = {x["name"] for x in selected}
    for p in fs.root.glob("f_*.pkl") if fs.root.exists() else ():
        if p.stem not in chosen: p.unlink()
    for item in selected:
        if not fs.path(item["name"]).exists():
            factor = compute(item["expression"], panels, mask).replace([np.inf, -np.inf], np.nan)
            fs.save(item["name"], factor)
    count = db.execute("SELECT COUNT(*) FROM trials").fetchone()[0]
    db.close()
    json_dump({"engine": ENGINE_VERSION, "frozen_through": "20211231", "factors": selected,
               "trials": count, "generation": gen_meta}, out / "candidates.json")
    return selected
