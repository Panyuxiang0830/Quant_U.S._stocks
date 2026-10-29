import numpy as np
import pandas as pd
import pytest
from usq.data import PanelStore, validate_panel, pit_panel, compare_sources
from usq.data.labels import forward_returns, neutralize
from usq.data.labels import prepare


def test_forward_ten_day_is_eleven_over_one_and_has_exact_end():
    dates = pd.bdate_range("2021-12-01", periods=30).strftime("%Y%m%d")
    p = pd.DataFrame({"A": 100 * 1.01 ** np.arange(30)}, index=dates)
    y, end = forward_returns(p)
    assert y.iloc[0, 0] == pytest.approx(1.01 ** 10 - 1)
    assert end.iloc[0] == dates[11]
    assert y.tail(11).isna().all().all()
    p.iloc[6, 0] = np.nan
    y, _ = forward_returns(p)
    assert pd.isna(y.iloc[0, 0])


def test_neutralization_removes_size_and_industry_wls():
    rng = np.random.default_rng(4)
    ids = [str(i) for i in range(100)]
    idx = ["20200102"]
    cap = pd.DataFrame([np.exp(rng.normal(20, 2, 100))], index=idx, columns=ids)
    ind = pd.DataFrame([["Tech" if i % 2 else "Other" for i in range(100)]], index=idx, columns=ids)
    y = .03 * np.log(cap) + .1 * ind.eq("Tech") + pd.DataFrame(rng.normal(0, .01, (1, 100)), index=idx, columns=ids)
    mask = y.notna()
    u = neutralize(y, cap, ind, mask, standardize=False)
    X = np.column_stack([np.log(cap.iloc[0]), pd.get_dummies(ind.iloc[0], dtype=float)])
    assert np.max(np.abs(X.T @ (u.iloc[0].to_numpy() * np.sqrt(cap.iloc[0].to_numpy())))) < 1e-6


def test_pit_does_not_backfill_or_use_future_restatement():
    r = pd.DataFrame({"sid": ["A", "A"], "available_date": ["20200103", "20200106"], "value": [1, 9]})
    x = pit_panel(r, ["20200102", "20200103", "20200106"], ["A"])
    assert pd.isna(x.iloc[0, 0])
    assert x.iloc[1, 0] == 1
    assert x.iloc[2, 0] == 9


def test_storage_schema_roundtrip_and_invalid_axes(tmp_path):
    x = pd.DataFrame([[1, 2], [3, np.nan]], index=["2020-01-02", "2020-01-03"], columns=["A", "B"])
    store = PanelStore(tmp_path); store.save("close", x)
    y = store.load("close")
    assert y.index.tolist() == ["20200102", "20200103"]
    with pytest.raises(ValueError): store.load("close", y.iloc[::-1])
    with pytest.raises(ValueError): validate_panel(y.iloc[::-1])
    with pytest.raises(ValueError): store.path("../secret")
    with pytest.raises(ValueError): validate_panel(pd.concat([y, y]))
    assert store.manifest()["close.pkl"]["shape"] == [2, 2]


def test_multi_source_reports_missing_separately():
    a = pd.DataFrame([[10., 20., 30.]], index=["20200102"], columns=list("ABC"))
    b = a.copy(); b.iloc[0, 0] = 11; b.iloc[0, 1] = np.nan
    s, diff = compare_sources(a, b)
    assert s["mismatch_count"] == 1 and s["secondary_missing_on_primary"] == 1
    assert diff.iloc[0, 0] == 1


def test_prepare_records_label_basis_and_adjusts_all_price_fields(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=30).strftime("%Y%m%d")
    ids = [str(i) for i in range(40)]
    x = pd.DataFrame(np.repeat((100 * 1.01 ** np.arange(30))[:, None], 40, axis=1), index=dates, columns=ids)
    store = PanelStore(tmp_path)
    for name in ("open", "high", "low", "close", "vwap"): store.save(name, x)
    store.save("volume", x * 0 + 1000)
    store.save("adj_factor", x * 0 + 2)
    store.save("market_cap", x * np.arange(1, 41))
    store.save("universe", x * 0 + 1)
    store.save("industry", pd.DataFrame("Tech", index=dates, columns=ids), numeric=False)
    prepare(store, mode="close_sum")
    raw = store.load("forward_10d_raw")
    assert raw.iloc[0, 0] == pytest.approx(.1, abs=1e-6)
    assert store.load("open_adj").iloc[0, 0] == pytest.approx(200)
    import json
    assert json.loads((tmp_path / "label_metadata.json").read_text())["mode"] == "close_sum"
    prepare(store, mode="vwap_compound")
    assert store.load("forward_10d_raw").iloc[0, 0] == pytest.approx(1.01 ** 10 - 1, abs=1e-6)
    with pytest.raises(ValueError): prepare(store, horizon=5)
