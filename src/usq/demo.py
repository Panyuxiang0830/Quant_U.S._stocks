"""Deterministic artificial-data integration exercise. Never US market evidence."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import yaml
from usq.data import PanelStore, prepare
from usq.mining import search
from usq.model import train_walkforward
from usq.workflow import evaluate, backtest
from usq.utils import json_dump


def synthetic_panels(root, stocks=60, seed=2964):
    if stocks < 40: raise ValueError("use at least 40 artificial stocks")
    rng = np.random.default_rng(seed)
    # Weekdays deliberately NOT claimed to be an exchange calendar.
    dates = pd.bdate_range("2014-01-01", "2026-01-20").strftime("%Y%m%d")
    ids = pd.Index([f"SYNTH_{i:04d}" for i in range(stocks)])
    T, N = len(dates), stocks
    state = np.zeros((T, N)); noise = rng.normal(size=(T, N))
    for t in range(1, T): state[t] = .92 * state[t - 1] + .35 * noise[t]
    rets = .0001 + .0007 * state + rng.normal(0, .009, (T, N)) + rng.normal(0, .003, (T, 1))
    close = 50 * np.exp(np.cumsum(rets, axis=0))
    volume = np.exp(14 + .15 * state + rng.normal(0, .4, (T, N)))
    op = close * np.exp(rng.normal(0, .003, (T, N)))
    vwap = (op + close) / 2
    shares = np.exp(rng.normal(18, 1, (1, N))) * np.exp(np.cumsum(rng.normal(0, .0001, (T, N)), axis=0))
    cap = close * shares
    fields = {"close": close, "open": op,
              "high": np.maximum(op, close) * (1 + rng.uniform(.001, .02, (T, N))),
              "low": np.minimum(op, close) * (1 - rng.uniform(.001, .02, (T, N))),
              "vwap": vwap, "amount": vwap * volume, "volume": volume,
              "adj_factor": np.ones((T, N)), "market_cap": cap,
              "universe": np.ones((T, N)), "shares_out": np.broadcast_to(shares, (T, N)),
              "book_equity": cap * np.exp(rng.normal(-.5, .5, (T, N))),
              "earnings_ttm": cap * rng.normal(.05, .02, (T, N)),
              "revenue_ttm": cap * np.exp(rng.normal(-.5, .5, (T, N))),
              "total_assets": cap * np.exp(rng.normal(0, .4, (T, N))),
              "total_liabilities": cap * np.exp(rng.normal(-1, .5, (T, N)))}
    store = PanelStore(root)
    for name, arr in fields.items(): store.save(name, pd.DataFrame(arr, index=dates, columns=ids))
    ind = np.tile(np.array([f"Sector_{i % 3}" for i in range(N)]), (T, 1))
    store.save("industry", pd.DataFrame(ind, index=dates, columns=ids), numeric=False)
    pd.DataFrame({"date": dates}).to_csv(store.root / "calendar.csv", index=False)
    prepare(store)
    return store


def demo(output="runs/demo", stocks=60, full=False):
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
    store = synthetic_panels(out / "data", stocks)
    cfg = {"data_dir": str(store.root), "factor_dir": str(out / "mining" / "factors"), "run_dir": str(out),
           "evaluation": {"min_stocks": 30, "groups": 10, "correlation_date_step": 10,
                          "correlation_stock_fraction": .8, "correlation_min_pairs": 100, "correlation_threshold": .9},
           "backtest": {"top_n": 10, "groups": 50, "cost_bps": 5., "borrow_bps_annual": 300.}}
    (out / "demo_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    entries = search(store, out / "mining", fields=("close", "volume", "returns", "vwap"), windows=(5, 20),
                     complexity=2, max_workers=1, batch_size=8,
                     max_trials=16, top_k=4, min_abs_rankic=0, min_coverage=.3)
    evaluate(cfg, "simple", "mining")
    evaluate(cfg, "simple", "validation")
    evaluate(cfg, "correlation")
    evaluate(cfg, "complex", "validation", with_barra=True)
    # Smoke selection is intentionally not a scientific feature screen.
    # Keep all synthetic candidates regardless of validation to guarantee coverage.
    json_dump({"frozen_through": "20221231", "factors": entries,
               "purpose": "ARTIFICIAL SMOKE TEST ONLY; no strategy validation"}, out / "selected.json")
    if full:
        train_walkforward(store, cfg["factor_dir"], out / "selected.json", out / "model",
                          params={"device": "cpu", "nthread": 4})
        backtest(cfg, with_barra=True)
    else:
        f = PanelStore(cfg["factor_dir"]).load(entries[0]["name"]) * entries[0]["direction"]
        f.loc[(f.index < "20230101") | (f.index > "20251231")] = np.nan
        f.to_pickle(out / "synthetic_scores.pkl", protocol=4)
        backtest(cfg, out / "synthetic_scores.pkl", with_barra=True)
    json_dump({"status": "passed", "artificial_data": True, "full_six_half_year_models": full,
               "stocks": stocks, "report": str(out / "backtest" / "report.html"),
               "barra_report": str(out / "barra" / "report.html")}, out / "demo_result.json")
    return out / "demo_result.json"
