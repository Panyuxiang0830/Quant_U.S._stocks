"""Export all evaluated candidate daily RankICs, including discarded trials."""
import json
import sqlite3
from pathlib import Path
import pandas as pd
from usq.mining.expressions import compute
from usq.evaluation import evaluate_simple
from usq.workflow import stage_days


def export_trial_ic(store, mining_dir, stage, output):
    root = Path(mining_dir)
    settings = json.loads((root / "search_config.json").read_text())
    label = store.load("label_10"); mask = store.load("universe", label).eq(1)
    days = stage_days(store, label, stage)
    panels = {}
    for name in settings["fields"]:
        if name == "returns":
            close = store.load("close_adj", label)
            panels[name] = close / close.shift(1) - 1
        else:
            source = f"{name}_adj" if name in ("open", "high", "low", "close", "vwap") else name
            panels[name] = store.load(source, label)
    db = sqlite3.connect(root / "trials.sqlite")
    result = {}
    for name, expression in db.execute("SELECT name, expression FROM trials WHERE status='ok' ORDER BY name"):
        factor = compute(expression, panels, mask)
        _, daily = evaluate_simple(factor.loc[days], label.loc[days], mask.loc[days], min_stocks=settings["min_stocks"], direction=1)
        result[name] = daily.rankic
    db.close()
    frame = pd.DataFrame(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    return {"path": str(output), "days": len(frame), "all_successful_candidates": len(frame.columns), "orientation": "raw; no factor sign flips"}
