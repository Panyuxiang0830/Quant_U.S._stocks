import ast
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest
from usq.data import PanelStore
from usq.mining.engine import FactorExpressionGenerator, Operators
from usq.mining.recursive import compute, generate_candidates, load_panels, ENGINE_VERSION
from usq.mining.search import search

FIXTURE = json.loads((Path(__file__).parent / "fixtures/recursive_engine_reference.json").read_text())


def test_original_operator_source_and_generator_logic_are_preserved():
    import usq.mining.engine as engine
    root = Path(engine.__file__).parent
    assert hashlib.sha256((root / "operators.py").read_bytes()).hexdigest() == FIXTURE["source_sha256"]["operators.py"]
    tree = ast.parse((root / "expression_generator.py").read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    def normalize(node):
        if isinstance(node, ast.AST):
            return [type(node).__name__, {k: normalize(v) for k, v in ast.iter_fields(node) if k != "type_params"}]
        if isinstance(node, list): return [normalize(x) for x in node]
        return node
    digest = hashlib.sha256(json.dumps(normalize(tree), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == FIXTURE["generator_logic_ast_sha256"]


@pytest.mark.parametrize("profile", FIXTURE["profiles"], ids=["complexity1", "complexity2", "complexity3"])
def test_candidate_trees_and_order_match_original_reference(profile):
    p = profile
    selected, meta = generate_candidates(p["fields"], p["dimensions"],
        windows=p["params"]["stats_windows"], time_periods=p["params"]["time_periods"],
        complexity=p["complexity"], dimension_filter=p["dimension_filter"],
        max_trials=p["max_count"], seed=p["params"]["binary_random_seed"])
    canonical = [[x["source_name"], {"expression": json.loads(x["expression"]), "dimension": x["dimension"]}] for x in selected]
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(serialized.encode()).hexdigest() == p["selected_sha256"]
    assert len(selected) == p["selected_count"]
    assert meta["generated_before_selection"] == p["generated_count"]


def test_original_rank_ties_and_rolling_minimum_observations():
    x = pd.DataFrame([[5., 5., 9.], [6., 2., 4.], [7., 4., 1.], [8., 5., 3.], [9., 8., 5.]])
    rank = compute(("rank", "close"), {"close": x})
    np.testing.assert_allclose(rank.iloc[0], [1/3, 1/3, 1.])
    pd.testing.assert_frame_equal(compute(("mean", "close", 5), {"close": x}), x.rolling(5, min_periods=1).mean())
    std = compute(("std", "close", 5), {"close": x})
    assert std.iloc[:2].isna().all().all()
    pd.testing.assert_frame_equal(std, x.rolling(5, min_periods=3).std())


def test_base_field_candidates_survive_catalog_json_and_search_journal(tmp_path):
    store = make_store(tmp_path / "data")
    label = store.load("label_10")
    panels, _ = load_panels(store, label, ["returns"])
    pd.testing.assert_frame_equal(compute(json.dumps("returns"), panels), panels["returns"])
    search(store, tmp_path / "run", fields=("returns",), complexity=1,
           max_trials=1, top_k=1, min_abs_rankic=0, min_coverage=0)
    db = sqlite3.connect(tmp_path / "run/trials.sqlite")
    assert db.execute("SELECT expression,status FROM trials").fetchall() == [('"returns"', 'ok')]
    db.close()


def test_recursive_trees_reproduce_original_numerical_operator_dispatch():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(10, 2, (35, 8)))
    y = pd.DataFrame(rng.normal(20, 3, (35, 8)))
    expr = ("rank", ("/", ("delta", "close", 5), ("std", "volume", 10)))
    op = Operators()
    expected = op._rank(op._divide(op._delta(x, 5), op._std(y, 10)))
    actual = compute(expr, {"close": x, "volume": y})
    pd.testing.assert_frame_equal(actual, expected)
    changed = x.copy(); changed.iloc[25:] *= 100
    altered = compute(expr, {"close": changed, "volume": y})
    pd.testing.assert_frame_equal(actual.iloc[:25], altered.iloc[:25])
    with pytest.raises(ValueError): compute("__import__('os').system('echo unsafe')", {"close": x})
    with pytest.raises(ValueError): compute(("lag", "close", -1), {"close": x})


def test_dimension_registry_and_cross_operator_rules_match_generator():
    g = FactorExpressionGenerator(["close", "volume"], additional_field_dimensions={"volume": "volume"})
    candidates = g.generate_dual_field_operators(params={"stats_windows": [5]})
    assert not any(v["expression"] == ("+", "close", "volume") for v in candidates.values())
    assert g._cross_operators_regulate("rank", "zscore") is False
    assert g._cross_operators_regulate("mean", "std") is False
    assert g._cross_operators_regulate("delta", "rank") is True


def make_store(root):
    rng = np.random.default_rng(91)
    dates = pd.bdate_range("2014-12-01", periods=80).strftime("%Y%m%d")
    ids = [str(i) for i in range(40)]
    x = pd.DataFrame(rng.lognormal(4, .3, (80, 40)), index=dates, columns=ids)
    store = PanelStore(root)
    store.save("close_adj", x); store.save("label_10", x.rank(axis=1) + rng.normal(0, 2, x.shape))
    store.save("universe", x * 0 + 1)
    pd.Series(dates, index=dates, name="label_end").shift(-11).to_csv(store.root / "label_end.csv", index_label="date")
    return store


def test_seed_expressions_and_existing_factor_panels_become_base_fields(tmp_path):
    store = make_store(tmp_path / "data")
    ref = store.load("label_10")
    seed_store = PanelStore(tmp_path / "seeds"); seed_store.save("wq101_003", ref)
    panels, dimensions = load_panels(store, ref, ["close"],
        seeds=[{"name": "reversal", "expression": "rank(ts_delta(close,5))", "dimension": "unitless"}],
        seed_factor_dir=seed_store.root, seed_factors=[{"name": "wq101_003", "dimension": "unitless"}])
    assert {"close", "reversal", "wq101_003"} == set(panels)
    assert dimensions["wq101_003"] == dimensions["reversal"] == "unitless"
    items, _ = generate_candidates(list(panels), dimensions, complexity=2, max_trials=300)
    assert any("reversal" in x["expression"] for x in items)
    assert any("wq101_003" in x["expression"] for x in items)


def test_required_seed_field_filter_is_applied_to_every_selected_tree():
    fields = ["close", "volume", "alpha_seed"]
    dims = {"close": "price", "volume": "volume", "alpha_seed": "unitless"}
    items, _ = generate_candidates(fields, dims, complexity=2, max_trials=80, required_fields=["alpha_seed"])
    assert len(items) == 80
    assert all("alpha_seed" in x["expression"] for x in items)


def test_post_2021_data_does_not_change_mining_quality(tmp_path):
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2021-11-01", periods=100).strftime("%Y%m%d")
    x = pd.DataFrame(rng.lognormal(4, .2, (100, 40)), index=dates, columns=[str(i) for i in range(40)])
    store = PanelStore(tmp_path / "data")
    store.save("close_adj", x); store.save("label_10", x.rank(axis=1)); store.save("universe", x * 0 + 1)
    pd.Series(dates, index=dates, name="label_end").shift(-11).to_csv(store.root / "label_end.csv", index_label="date")
    kw = dict(fields=("close",), max_trials=3, top_k=3, complexity=1, max_workers=1, min_abs_rankic=0, min_coverage=0)
    a = search(store, tmp_path / "before", **kw)
    changed = x.copy(); changed.loc[changed.index >= "20220101"] *= rng.uniform(20, 100, changed.loc[changed.index >= "20220101"].shape)
    store.save("close_adj", changed)
    y = store.load("label_10"); y.loc[y.index >= "20220101"] *= -10
    store.save("label_10", y)
    b = search(store, tmp_path / "after", **kw)
    assert [(r["name"], r["is_rankic"]) for r in a] == [(r["name"], r["is_rankic"]) for r in b]


def test_spawn_parallel_batches_equal_serial_and_resume_preserves_ledger(tmp_path):
    store = make_store(tmp_path / "data")
    kw = dict(fields=("close",), windows=(5,), time_periods=(1,), complexity=2,
              max_trials=8, top_k=3, batch_size=3, min_abs_rankic=0, min_coverage=0,
              min_stocks=30, start_method="spawn")
    serial = search(store, tmp_path / "serial", max_workers=1, **kw)
    parallel = search(store, tmp_path / "parallel", max_workers=2, **kw)
    assert [(x["name"], x["is_rankic"]) for x in serial] == [(x["name"], x["is_rankic"]) for x in parallel]
    assert all(x["engine"] == ENGINE_VERSION for x in parallel)
    resumed = search(store, tmp_path / "parallel", max_workers=2, **kw)
    assert resumed == parallel
    db = sqlite3.connect(tmp_path / "parallel/trials.sqlite")
    assert db.execute("SELECT COUNT(*) FROM trials").fetchone()[0] == 8
    db.close()
    assert len(list((tmp_path / "parallel/batches").glob("*.csv"))) == 3
