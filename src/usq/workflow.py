from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from usq.data import PanelStore
from usq.evaluation import evaluate_simple, evaluate_complex, evaluate_correlation
from usq.backtest.engine import run_backtest, group_backtests
from usq.barra import build_model, analyze, attribute_positions
from usq.report import save_backtest, save_barra
from usq.utils import json_dump, sha256

STAGES = {"mining": ("20150101", "20220101"), "validation": ("20220101", "20230101")}


def stage_days(store, reference, stage):
    lo, hi = STAGES[stage]
    end = pd.read_csv(store.root / "label_end.csv", dtype=str, index_col=0).iloc[:, 0].reindex(reference.index)
    return (reference.index >= lo) & (reference.index < hi) & end.notna() & end.lt(hi)


def candidates(cfg):
    p = Path(cfg["run_dir"]) / "mining" / "candidates.json"
    return json.loads(p.read_text())["factors"]


def evaluate(cfg, level, stage="mining", with_barra=False):
    store = PanelStore(cfg["data_dir"])
    fs = PanelStore(cfg["factor_dir"])
    out = Path(cfg["run_dir"]) / "evaluation"; out.mkdir(parents=True, exist_ok=True)
    label = store.load("label_10")
    mask = store.load("universe", label).eq(1)
    selected_days = stage_days(store, label, stage)
    entries = candidates(cfg)
    if not entries:
        raise ValueError("no retained candidates; inspect trials, data quality and predeclared thresholds")
    if level == "simple":
        summaries = {}
        daily_dir = out / f"{stage}_daily"; daily_dir.mkdir(exist_ok=True)
        for item in entries:
            f = fs.load(item["name"], label)
            history = None
            if stage == "validation":
                history_path = out / "mining_daily" / f"{item['name']}.parquet"
                if not history_path.exists():
                    raise ValueError("run mining simple evaluation first to supply causal two-year RankIC history")
                history = pd.read_parquet(history_path).rankic
            s, daily = evaluate_simple(f.loc[selected_days], label.loc[selected_days], mask.loc[selected_days],
                min_stocks=cfg["evaluation"]["min_stocks"], direction=item["direction"], rankic_history=history)
            summaries[item["name"]] = s
            daily.to_parquet(daily_dir / f"{item['name']}.parquet")
        json_dump({"stage": stage, "period": STAGES[stage], "factors": summaries}, out / f"{stage}_simple.json")
        return summaries
    if level == "correlation":
        if stage != "mining":
            raise ValueError("default correlation policy is frozen on mining years only")
        ev = cfg["evaluation"]
        # Read/sample one at a time, so full T*N*F matrices are never resident.
        dates = label.index[selected_days][::ev["correlation_date_step"]]
        sampled = {x["name"]: fs.load(x["name"], label).loc[dates] for x in entries}
        q = {x["name"]: x["is_rankic"] for x in entries}
        corr, overlaps, meta = evaluate_correlation(sampled, q, mask.loc[dates], date_step=1,
            stock_fraction=ev["correlation_stock_fraction"], threshold=ev["correlation_threshold"],
            min_pairs=ev["correlation_min_pairs"], seed=2964)
        meta["full_calendar_date_step"] = ev["correlation_date_step"]
        corr.to_csv(out / "correlation.csv"); overlaps.to_csv(out / "correlation_overlap.csv")
        json_dump(meta, out / "correlation_meta.json")
        return meta
    if level != "complex": raise ValueError("unknown evaluation layer")
    vwap = store.load("vwap_adj", label)
    ctx, barra_meta = None, None
    if with_barra:
        model = barra_model(cfg, store)
        barra_meta = {**model.meta, "has_benchmark": True}
        ctx = {"model": model}
    summaries = {}
    for item in entries:
        f = fs.load(item["name"], label).where(pd.DataFrame(np.repeat(np.asarray(selected_days)[:, None], label.shape[1], axis=1), index=label.index, columns=label.columns))
        s, bt, groups, attr = evaluate_complex(f, vwap, mask, item["direction"],
            groups=cfg["evaluation"]["groups"], cost_bps=cfg["backtest"]["cost_bps"],
            borrow_bps_annual=cfg["backtest"]["borrow_bps_annual"], barra_context=ctx,
            start=STAGES[stage][0], end=(pd.Timestamp(STAGES[stage][1]) - pd.Timedelta(days=1)).strftime("%Y%m%d"))
        summaries[item["name"]] = s
        target = out / f"{stage}_complex" / item["name"]
        save_backtest(bt, groups, target, f"Single factor {item['name']} {stage}")
        if attr is not None: save_barra(attr, bt, target / "barra", barra_meta)
    json_dump({"stage": stage, "factors": summaries}, out / f"{stage}_complex.json")
    return summaries


def select(cfg):
    out = Path(cfg["run_dir"]); ev = out / "evaluation"
    val = json.loads((ev / "validation_simple.json").read_text())["factors"]
    complex_ = json.loads((ev / "validation_complex.json").read_text())["factors"]
    corr = pd.read_csv(ev / "correlation.csv", index_col=0)
    policy = cfg["selection"]
    kept, audit = [], []
    ordered = sorted(candidates(cfg), key=lambda x: (-abs(x["is_rankic"]), x["name"]))
    for item in ordered:
        name = item["name"]
        s, c = val[name], complex_[name]
        tests = {"validation_direction_and_rankic": s["rankic_mean"] is not None and s["rankic_mean"] > policy["min_validation_rankic"],
                 "validation_coverage": s["coverage"] >= policy["min_validation_coverage"],
                 "monotonicity": c["group_monotonicity_spearman"] is not None and c["group_monotonicity_spearman"] >= policy["min_monotonicity"],
                 "net_sharpe": c["net"]["sharpe"] is not None and c["net"]["sharpe"] >= policy["min_net_sharpe"]}
        conflicts = [k["name"] for k in kept if pd.isna(corr.loc[name, k["name"]]) or abs(corr.loc[name, k["name"]]) >= cfg["evaluation"]["correlation_threshold"]]
        tests["correlation"] = not conflicts and np.isfinite(corr.loc[name, name])
        passed = all(tests.values())
        audit.append({"name": name, "passed": passed, "checks": tests, "correlation_conflicts": conflicts})
        if passed: kept.append(item)
    manifest = {"frozen_through": "20221231", "factors": kept, "policy": policy,
                "correlation_threshold": cfg["evaluation"]["correlation_threshold"], "audit": audit,
                "evidence_sha256": {p.name: sha256(p) for p in (ev / "validation_simple.json", ev / "validation_complex.json", ev / "correlation.csv")}}
    json_dump(manifest, out / "selected.json")
    if not kept: raise ValueError("no factors pass the frozen policy; no OOS run produced")
    return manifest


def backtest(cfg, scores_path=None, with_barra=False):
    store = PanelStore(cfg["data_dir"])
    vwap = store.load("vwap_adj")
    mask = store.load("universe", vwap).eq(1)
    path = Path(scores_path or Path(cfg["run_dir"]) / "model" / "scores.pkl")
    score = pd.read_pickle(path)
    bc = cfg["backtest"]
    kwargs = dict(cost_bps=bc["cost_bps"], borrow_bps_annual=bc["borrow_bps_annual"], start="20230101", end="20251231")
    cap = store.load("market_cap", vwap)
    bw = cap.where(mask & cap.gt(0)).div(cap.where(mask & cap.gt(0)).sum(axis=1), axis=0).shift(2)
    interval = vwap / vwap.shift(1) - 1
    relevant = score.reindex(vwap.index).notna().any(axis=1).shift(2, fill_value=False) & (vwap.index >= "20230101") & (vwap.index <= "20251231")
    if (bw.loc[relevant].fillna(0).ne(0) & interval.loc[relevant].isna()).any().any():
        raise ValueError("missing market-cap benchmark held returns; resolve delisting/corporate-action marks")
    benchmark = (bw * interval.fillna(0)).sum(axis=1)
    # Score file may include warmup rows as all-NaN; calendar remains complete.
    bt = run_backtest(score, vwap, mask, top_n=bc["top_n"], benchmark_returns=benchmark, **kwargs)
    groups = group_backtests(score, vwap, mask, groups=bc["groups"], **kwargs)
    root = Path(cfg["run_dir"])
    path = save_backtest(bt, groups, root / "backtest", "2023-2025 score backtest")
    if with_barra:
        model = barra_model(cfg, store)
        attr = analyze(bt, model=model)
        save_barra(attr, bt, root / "barra", {**model.meta, "has_benchmark": True})
    return path


def barra_model(cfg, store=None, force=False):
    store = store or PanelStore(cfg["data_dir"])
    opts = cfg.get("barra", {})
    return build_model(store, opts.get("model_dir", str(Path(cfg["run_dir"])/"barra_model")),
        start=opts.get("start", "20220101"), end=opts.get("end", "20251231"),
        min_obs=opts.get("min_obs",30), descriptor_dir=opts.get("descriptor_dir"), force=force)


def barra(cfg, holdings=None, timing="signal_close", benchmark_weights=None,
          benchmark_timing="signal_close", no_benchmark=False, output=None, start=None, end=None):
    model = barra_model(cfg)
    root=Path(cfg["run_dir"])
    if holdings is None:
        from usq.backtest.engine import BacktestResult
        daily=pd.read_csv(root/"backtest/daily.csv",index_col=0,dtype={0:str})
        daily.index=daily.index.astype(str)
        weights=pd.read_pickle(root/"backtest/weights.pkl")
        if start or end:
            keep=(daily.index >= (start or daily.index[0])) & (daily.index <= (end or daily.index[-1]))
            daily=daily.loc[keep];weights=weights.reindex(daily.index)
        bt=BacktestResult(daily,weights,{})
        if benchmark_weights or no_benchmark:
            attr=attribute_positions(weights,model,benchmark_weights,timing="interval",benchmark_timing=benchmark_timing,
                benchmark=None if no_benchmark else "cap_weighted",costs=daily.cost+daily.borrow_cost)
            if not np.allclose(attr["daily_returns"].portfolio,daily.net_return,atol=1e-10): raise ValueError("strategy net returns differ")
        else: attr=analyze(bt,model=model)
        basis="回测净收益：已含交易与借券成本"
    else:
        bt=None
        attr=attribute_positions(holdings,model,benchmark_weights,timing,benchmark_timing,start,end,
            benchmark=None if no_benchmark else "cap_weighted")
        basis="持仓毛收益：未提供外部成本账本"
    return save_barra(attr,bt,output or root/"barra",{**model.meta,"has_benchmark":not no_benchmark,"portfolio_return_basis":basis})
