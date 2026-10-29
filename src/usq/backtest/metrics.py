import numpy as np
import pandas as pd


def performance(returns, turnover=None, annualization=252, rf_annual=0.0):
    r = pd.Series(returns).dropna().astype(float)
    if r.empty or (r <= -1).any() or not np.isfinite(r).all():
        raise ValueError("need finite simple returns > -1")
    nav = (1 + r).cumprod()
    # Include initial NAV=1, so a loss on the first day is counted as a drawdown.
    drawdown = nav / nav.cummax().clip(lower=1) - 1
    ann = float(nav.iloc[-1] ** (annualization / len(r)) - 1)
    vol = float(r.std(ddof=1) * np.sqrt(annualization))
    sharpe = float((r.mean() * annualization - rf_annual) / vol) if vol > 0 else np.nan
    mean_turn = float(pd.Series(turnover).reindex(r.index).mean()) if turnover is not None else np.nan
    # Local WQ-style diagnostic; not a calibrated BRAIN platform result.
    fitness = sharpe * np.sqrt(abs(ann) / max(mean_turn, .125)) if np.isfinite(mean_turn) else np.nan
    return {"days": len(r), "total_return": float(nav.iloc[-1] - 1), "annual_return": ann,
            "annual_volatility": vol, "sharpe": sharpe,
            "max_drawdown": float(drawdown.min()), "mean_daily_turnover": mean_turn,
            "fitness_local": float(fitness)}
