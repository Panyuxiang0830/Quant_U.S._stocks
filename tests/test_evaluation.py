import numpy as np
import pandas as pd
import pytest
from usq.evaluation import evaluate_simple, evaluate_correlation
from usq.evaluation.direction_cv import direction_cv


def test_perfect_rankic_and_fixed_direction():
    x = pd.DataFrame(np.tile(np.arange(40), (60, 1)), index=pd.bdate_range("2020-01-01", periods=60).strftime("%Y%m%d"))
    s, daily = evaluate_simple(x, -x, x.notna(), direction=1)
    assert s["rankic_mean"] == pytest.approx(-1)
    s, _ = evaluate_simple(x, -x, x.notna(), direction=-1)
    assert s["rankic_mean"] == pytest.approx(1)
    assert "rankic_cv" not in s
    assert "cv" in daily


def test_user_cv_daily_rankic_signed_extreme_and_no_current_leak():
    ix = pd.bdate_range("2019-01-01", "2022-01-10").strftime("%Y%m%d")
    s = pd.Series(.02, index=ix); s.loc["20210104"] = -.1; s.loc["20210105"] = .2
    s.loc["20220107"] = .5
    out = direction_cv(s)
    assert out.loc["20220107", "denominator"] == -.1
    assert out.loc["20220107", "cv"] == pytest.approx(-5)
    s.loc["20220107"] = -.3
    out = direction_cv(s)
    assert out.loc["20220107", "denominator"] == .2
    assert out.loc["20220107", "cv"] == pytest.approx(-1.5)
    assert pd.isna(out.iloc[300].cv)  # less than two calendar years


def test_cv_expires_old_extreme_and_zero_denominator():
    ix = pd.bdate_range("2018-01-01", "2021-01-15").strftime("%Y%m%d")
    s = pd.Series(.1, index=ix); s.iloc[0] = -.9
    assert direction_cv(s).iloc[-1].cv == pytest.approx(1)
    s[:] = 0; s.iloc[-1] = .1
    assert pd.isna(direction_cv(s).iloc[-1].cv)


def test_correlation_drops_negative_duplicates_and_is_deterministic():
    rng = np.random.default_rng(1)
    a = pd.DataFrame(rng.normal(size=(60, 50)), index=pd.bdate_range("2020-01-01", periods=60).strftime("%Y%m%d"))
    b = pd.DataFrame(rng.normal(size=a.shape), index=a.index, columns=a.columns)
    fs = {"a": a, "minus_a": -a, "b": b, "constant": a * 0}
    kw = dict(date_step=2, stock_fraction=.8, min_pairs=50, seed=9)
    c, n, meta = evaluate_correlation(fs, {"a": .03, "minus_a": .02, "b": .01, "constant": 0}, a.notna(), **kw)
    assert c.loc["a", "minus_a"] == pytest.approx(-1)
    assert meta["kept"] == ["a", "b"]
    assert n.loc["a", "b"] == 30 * 40
    assert evaluate_correlation(fs, {"a": .03, "minus_a": .02, "b": .01}, a.notna(), **kw)[2]["sids"] == meta["sids"]
