import pandas as pd
from usq.data import PanelStore
import usq.mining.search as search_module
import importlib


def test_resume_keeps_eligible_factor_even_if_higher_quality_trial_failed_coverage(tmp_path, monkeypatch):
    mod = importlib.import_module("usq.mining.search")
    store = PanelStore(tmp_path / "panels")
    dates = pd.bdate_range("2015-01-01", periods=40).strftime("%Y%m%d")
    panel = pd.DataFrame(1., index=dates, columns=[str(i) for i in range(40)])
    for name in ("close_adj", "label_10", "universe"): store.save(name, panel)
    pd.Series(dates, index=dates, name="label_end").to_csv(store.root / "label_end.csv", index_label="date")
    calls = []
    def compute(expression, panels, mask):
        calls.append(expression)
        return panel * len(calls)
    def evaluate(factor, label, mask, **kwargs):
        n = int(factor.iloc[0, 0])
        summary = {"rankic_mean": .08 if n == 1 else .09, "direction": 1,
                   "coverage": .9 if n == 1 else .1}
        return summary, pd.DataFrame({"rankic": .02}, index=dates)
    monkeypatch.setattr(mod, "compute", compute)
    monkeypatch.setattr(mod, "evaluate_simple", evaluate)
    args = dict(fields=("close",), windows=(5,), max_trials=2, top_k=1,
                min_abs_rankic=.01, min_coverage=.6)
    first = mod.search(store, tmp_path / "mining", **args)
    second = mod.search(store, tmp_path / "mining", **args)
    assert len(first) == len(second) == 1
    assert first[0]["name"] == second[0]["name"]
    assert len(calls) == 2  # existing trials were not recomputed
