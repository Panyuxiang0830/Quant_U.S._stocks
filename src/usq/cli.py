from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from usq.data import PanelStore, prepare, compare_sources
from usq.utils import json_dump


def main(argv=None):
    p = argparse.ArgumentParser(description="US factor research. Read docs/HANDBOOK.md before real-data runs.")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("audit-data")
    sub.add_parser("mine")
    e = sub.add_parser("evaluate"); e.add_argument("--level", choices=["simple", "complex", "correlation"], required=True)
    e.add_argument("--stage", choices=["mining", "validation"], default="mining"); e.add_argument("--barra", action="store_true")
    sub.add_parser("select")
    sub.add_parser("train")
    b = sub.add_parser("backtest"); b.add_argument("--scores"); b.add_argument("--barra", action="store_true")
    sub.add_parser("barra")
    d = sub.add_parser("demo"); d.add_argument("--output", default="runs/demo"); d.add_argument("--stocks", type=int, default=60)
    d.add_argument("--full", action="store_true", help="also train all six CPU models, 500 trees each")
    imp = sub.add_parser("import-library"); imp.add_argument("--source", required=True); imp.add_argument("--family", choices=["wq101", "gtja191"], required=True)
    imp.add_argument("--require-complete", action="store_true")
    t = sub.add_parser("trial-ic"); t.add_argument("--stage", choices=["mining", "validation"], required=True)
    t.add_argument("--output", required=True)
    c = sub.add_parser("compare"); c.add_argument("--primary", required=True); c.add_argument("--secondary", required=True)
    c.add_argument("--output", required=True); c.add_argument("--rtol", type=float, default=.001); c.add_argument("--atol", type=float, default=.01)
    r = sub.add_parser("research"); r.add_argument("--is-ic", required=True); r.add_argument("--val-ic", required=True)
    r.add_argument("--output", required=True); r.add_argument("--k", type=int, default=20); r.add_argument("--repeats", type=int, default=500)
    r.add_argument("--block", type=int, default=20)
    a = p.parse_args(argv)
    if a.command == "demo":
        from usq.demo import demo
        print(demo(a.output, a.stocks, a.full)); return
    if a.command == "compare":
        s, d = compare_sources(pd.read_pickle(a.primary), pd.read_pickle(a.secondary), a.rtol, a.atol)
        Path(a.output).mkdir(parents=True, exist_ok=True)
        json_dump(s, Path(a.output) / "summary.json"); d.to_csv(Path(a.output) / "mismatches.csv"); print(s); return
    if a.command == "research":
        from usq.research import validation_experiment
        s, d = validation_experiment(pd.read_parquet(a.is_ic), pd.read_parquet(a.val_ic), k=a.k, repeats=a.repeats, block=a.block)
        Path(a.output).mkdir(parents=True, exist_ok=True)
        json_dump(s, Path(a.output) / "summary.json"); d.to_csv(Path(a.output) / "null.csv", index=False); print(s); return
    cfg = yaml.safe_load(Path(a.config).read_text())
    store = PanelStore(cfg["data_dir"])
    from usq import workflow
    if a.command == "prepare": result = prepare(store, **cfg.get("label", {}))
    elif a.command == "audit-data":
        from usq.data.audit import audit_data
        result = audit_data(store, Path(cfg["run_dir"]) / "data_audit.json")
    elif a.command == "mine":
        from usq.mining import search
        result = search(store, Path(cfg["run_dir"]) / "mining", **cfg["mining"])
    elif a.command == "evaluate": result = workflow.evaluate(cfg, a.level, a.stage, a.barra)
    elif a.command == "select": result = workflow.select(cfg)
    elif a.command == "train":
        from usq.model import train_walkforward
        _, result = train_walkforward(store, cfg["factor_dir"], Path(cfg["run_dir"]) / "selected.json", Path(cfg["run_dir"]) / "model", cfg["model"])
    elif a.command in ("backtest", "barra"):
        result = workflow.backtest(cfg, getattr(a, "scores", None), with_barra=a.command == "barra" or a.barra)
    elif a.command == "import-library":
        from usq.mining.libraries import import_library
        result = import_library(a.source, cfg["factor_dir"], a.family, store.load("label_10"), a.require_complete)
        # Register imported matrices for the same three evaluation layers.
        from usq.evaluation import evaluate_simple
        label = store.load("label_10"); mask = store.load("universe", label)
        days = workflow.stage_days(store, label, "mining")
        path = Path(cfg["run_dir"]) / "mining" / "candidates.json"
        existing = json.loads(path.read_text()) if path.exists() else {"frozen_through": "20211231", "factors": []}
        registered = {x["name"]: x for x in existing["factors"]}
        for row in result:
            if row["status"] != "imported": continue
            name = row["name"]
            f = PanelStore(cfg["factor_dir"]).load(name, label)
            summary, _ = evaluate_simple(f.loc[days], label.loc[days], mask.loc[days], min_stocks=cfg["evaluation"]["min_stocks"])
            registered[name] = {"name": name, "direction": summary["direction"], "is_rankic": summary["rankic_mean"], "source": a.family}
        existing["factors"] = list(registered.values()); json_dump(existing, path)
    elif a.command == "trial-ic":
        from usq.mining.trial_ic import export_trial_ic
        result = export_trial_ic(store, Path(cfg["run_dir"]) / "mining", a.stage, a.output)
    else: raise AssertionError(a.command)
    print(f"Completed {a.command}: {len(result) if isinstance(result, (list, dict)) else result}")


if __name__ == "__main__": main()
