import numpy as np
import pandas as pd
import pytest
from usq.barra.attribution import cross_section, carino_link, risk_decomposition
from usq.mining.legacy_templates import compute, enumerate_expressions
from usq.research import validation_experiment


def test_constrained_wls_reconciles_and_industry_zero_sum():
    rng = np.random.default_rng(42)
    ids = pd.Index([str(i) for i in range(120)])
    styles = pd.DataFrame(rng.normal(size=(120, 3)), columns=["Size", "Beta", "Value"], index=ids)
    ind = pd.Series(np.where(np.arange(120) % 2, "Tech", "Other"), index=ids)
    cap = pd.Series(np.exp(rng.normal(20, 1, 120)), index=ids)
    r = .01 + .02 * styles.Size + .003 * ind.eq("Tech") + rng.normal(0, .001, 120)
    f, u, X, complete, d = cross_section(styles, ind, cap, r, pd.Series(True, index=ids))
    assert d["industry_constraint_error"] == pytest.approx(0, abs=1e-12)
    assert np.allclose(X @ f + u, r)
    assert d["r2"] > .99


def test_carino_active_geometric_identity_every_day():
    idx = ["20200102", "20200103", "20200106"]
    rp = pd.Series([.03, -.02, .01], index=idx)
    rb = pd.Series([.01, -.01, .02], index=idx)
    c = pd.DataFrame({"Market": (rp - rb) * .4, "Specific": (rp - rb) * .6}, index=idx)
    linked = carino_link(c, rp, rb)
    assert np.allclose(linked.sum(axis=1), (1 + rp).cumprod() - (1 + rb).cumprod())
    assert np.isfinite(carino_link(c * 0, rb, rb)).all().all()


def test_risk_variance_formula():
    rng = np.random.default_rng(1)
    w = pd.Series([.4, .6], index=["A", "B"])
    X = pd.DataFrame({"Market": [1., 1.]}, index=w.index)
    f = pd.DataFrame({"Market": rng.normal(0, .01, 100)})
    u = pd.DataFrame(rng.normal(0, .02, (100, 2)), columns=w.index)
    result = risk_decomposition(w, X, f, u)
    assert result["factor_variance"] == pytest.approx(f.Market.var())
    assert result["specific_variance"] == pytest.approx((w ** 2 * u.var()).sum())


def test_legacy_ledger_expression_no_future_dependence_and_unsafe_calls_rejected():
    rng = np.random.default_rng(3)
    x = pd.DataFrame(rng.normal(10, 1, (40, 10)))
    exp = "rank(ts_delta(close,5)/ts_std(close,10))"
    a = compute(exp, {"close": x})
    y = x.copy(); y.iloc[25:] *= 100
    b = compute(exp, {"close": y})
    pd.testing.assert_frame_equal(a.iloc[:25], b.iloc[:25])
    with pytest.raises(ValueError): compute("__import__('os').system('echo bad')", {"close": x})
    with pytest.raises(ValueError): compute("ts_delay(close,-1)", {"close": x})
    assert len(list(enumerate_expressions(["close", "volume"], [5, 10]))) == 24


def test_validation_research_does_not_accept_oos_and_is_seeded():
    rng = np.random.default_rng(2)
    a = pd.DataFrame(rng.normal(0, .02, (500, 10)), index=pd.bdate_range("2020-01-01", periods=500).strftime("%Y%m%d"))
    b = pd.DataFrame(rng.normal(0, .02, (200, 10)), index=pd.bdate_range("2022-01-03", periods=200).strftime("%Y%m%d"))
    s, null = validation_experiment(a, b, k=3, repeats=20)
    s2, null2 = validation_experiment(a, b, k=3, repeats=20)
    pd.testing.assert_frame_equal(null, null2)
    assert s["n_candidates"] == 10
    b.index = pd.bdate_range("2023-01-02", periods=200).strftime("%Y%m%d")
    with pytest.raises(ValueError): validation_experiment(a, b, k=3, repeats=20)
