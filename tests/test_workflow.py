import json
import numpy as np
import pandas as pd
import pytest
from usq.data import PanelStore
from usq.workflow import stage_days, select
from usq.model import half_years, XGB_PARAMS
from usq.backtest import run_backtest
from usq.mining.libraries import import_library


def test_split_purges_by_actual_label_exit_not_row_count(tmp_path):
    s = PanelStore(tmp_path)
    x = pd.DataFrame([[1], [2], [3]], index=["20211210", "20211220", "20220103"], columns=["A"])
    pd.Series(["20211228", "20220106", "20220120"], index=x.index, name="label_end").to_csv(tmp_path / "label_end.csv", index_label="date")
    assert stage_days(s, x, "mining").tolist() == [True, False, False]
    assert stage_days(s, x, "validation").tolist() == [False, False, True]


def test_model_policy_matches_image_and_six_half_years():
    periods = list(half_years())
    assert len(periods) == 6
    assert periods[0] == ("20230101", "20230701")
    assert periods[-1] == ("20250701", "20260101")
    assert XGB_PARAMS["n_estimators"] == 500
    assert XGB_PARAMS["max_depth"] == 7
    assert XGB_PARAMS["device"] == "cuda"
    assert XGB_PARAMS["missing"] is None
    assert "early_stopping_rounds" not in XGB_PARAMS


def test_internal_missing_score_day_is_not_silently_skipped():
    dates = pd.bdate_range("2020-01-01", periods=8).strftime("%Y%m%d")
    score = pd.DataFrame(1., index=dates, columns=["A"]); score.iloc[3] = np.nan
    price = score.copy().fillna(1)
    with pytest.raises(ValueError, match="internal missing score day"):
        run_backtest(score, price, price.notna())


def test_library_import_reports_each_expected_id_and_registers_no_fake_formulas(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    ref = pd.DataFrame([[1., 2.]], index=["20200102"], columns=["A", "B"])
    PanelStore(source).save("wq101_001", ref)
    result = import_library(source, target, "wq101", ref)
    assert len(result) == 101
    assert sum(x["status"] == "imported" for x in result) == 1
    with pytest.raises(ValueError, match="incomplete"):
        import_library(source, target, "wq101", ref, require_complete=True)


def test_validation_never_flips_is_direction_and_selection_does_not_use_oos(tmp_path):
    root = tmp_path; ev = root / "evaluation"; ev.mkdir(); (root / "mining").mkdir()
    (root / "mining" / "candidates.json").write_text(json.dumps({"factors": [
        {"name": "good", "direction": -1, "is_rankic": .03},
        {"name": "bad", "direction": 1, "is_rankic": .02}]}))
    (ev / "validation_simple.json").write_text(json.dumps({"factors": {
        "good": {"rankic_mean": .01, "coverage": 1}, "bad": {"rankic_mean": -.01, "coverage": 1}}}))
    (ev / "validation_complex.json").write_text(json.dumps({"factors": {
        k: {"group_monotonicity_spearman": .5, "net": {"sharpe": 1}} for k in ("good", "bad")}}))
    pd.DataFrame([[1., .1], [.1, 1.]], index=["good", "bad"], columns=["good", "bad"]).to_csv(ev / "correlation.csv")
    cfg = {"run_dir": str(root), "selection": {"min_validation_rankic": 0., "min_validation_coverage": .5,
           "min_monotonicity": 0., "min_net_sharpe": 0.}, "evaluation": {"correlation_threshold": .7}}
    m = select(cfg)
    assert [x["name"] for x in m["factors"]] == ["good"]
    assert m["factors"][0]["direction"] == -1
    assert m["frozen_through"] == "20221231"
