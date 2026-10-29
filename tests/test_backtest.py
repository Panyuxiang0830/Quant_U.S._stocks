import numpy as np
import pandas as pd
import pytest
from usq.backtest import run_backtest, score_groups, performance


def panels():
    ix = pd.Index(["20200102", "20200103", "20200106", "20200107"])
    p = pd.DataFrame({"A": [100., 110., 121., 121.], "B": [100., 100., 100., 100.]}, index=ix)
    score = pd.DataFrame({"A": [2., 0., np.nan, np.nan], "B": [1., 2., np.nan, np.nan]}, index=ix)
    return score, p, p.notna()


def test_two_day_alignment_and_cost_convention():
    score, p, mask = panels()
    bt = run_backtest(score, p, mask, top_n=1, cost_bps=5)
    assert bt.daily.index.tolist() == ["20200106", "20200107"]
    assert bt.daily.iloc[0].gross_return == pytest.approx(.1)
    assert bt.daily.iloc[0].cost == pytest.approx(.0005)
    assert bt.daily.iloc[1].gross_return == 0
    assert bt.daily.iloc[1].cost == pytest.approx(.001)


def test_missing_selected_return_errors_without_reselecting():
    score, p, mask = panels(); p.loc["20200106", "A"] = np.nan
    with pytest.raises(ValueError, match="missing held return"):
        run_backtest(score, p, mask, top_n=1)


def test_future_tradability_not_used_in_signal_day_selection():
    score, p, mask = panels(); mask.loc["20200103", "A"] = False
    bt = run_backtest(score, p, mask, top_n=1)
    assert bt.weights.loc["20200106", "A"] == 1


def test_first_day_loss_is_drawdown():
    s = performance(pd.Series([-.1, .02]))
    assert s["max_drawdown"] == pytest.approx(-.1)


def test_equal_scores_stay_in_same_group():
    x = pd.DataFrame([[1., 1., 1., 1.]], columns=list("ABCD"))
    groups = score_groups(x, x.notna(), 50)
    assert groups.iloc[0].nunique() == 1


def test_turnover_uses_drifted_holdings():
    score, p, mask = panels(); score.iloc[:2, :] = [1, 1]
    # Top2 target always 50/50, but price changes force non-zero rebalance.
    bt = run_backtest(score, p, mask, top_n=2, cost_bps=0)
    assert bt.daily.iloc[1].turnover == pytest.approx(abs(.5 - .55/1.05))
