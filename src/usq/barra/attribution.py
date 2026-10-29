from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section(styles, industry, cap, returns, estu, min_obs=30):
    """Joint constrained WLS: Market + full industries + style exposures.

    Industry cap-weighted returns sum to zero. Objective weights=sqrt(cap), so
    OLS rows are multiplied by cap**.25 to implement those objective weights.
    """
    complete = styles.notna().all(axis=1) & industry.notna() & cap.gt(0)
    fit = complete & returns.notna() & np.isfinite(returns) & estu.eq(1)
    if fit.sum() < min_obs:
        raise ValueError("insufficient risk-regression observations")
    industries = sorted(industry[fit].astype(str).unique())
    dummies = pd.get_dummies(industry.astype("string"), dtype=float).reindex(columns=industries, fill_value=0)
    dummies.columns = [f"Industry:{c}" for c in industries]
    X = pd.concat([pd.Series(1., index=styles.index, name="Market"), dummies, styles], axis=1)
    capfit = cap[fit].to_numpy(float)
    share = cap[fit].groupby(industry[fit].astype(str)).sum().reindex(industries)
    share = (share / share.sum()).to_numpy()
    p, k = len(industries), X.shape[1]
    # Delete last industry free parameter, reintroduce via zero-sum constraint.
    C = np.delete(np.eye(k), p, axis=1)
    C[p, 1:p] = -share[:-1] / share[-1]
    xf, y = X.loc[fit].to_numpy(float), returns[fit].to_numpy(float)
    sw = (capfit / np.median(capfit)) ** .25
    reduced = (xf @ C) * sw[:, None]
    if fit.sum() <= reduced.shape[1] + 2 or np.linalg.matrix_rank(reduced) < reduced.shape[1]:
        raise ValueError("underdetermined/collinear risk regression")
    coef = C @ np.linalg.lstsq(reduced, y * sw, rcond=None)[0]
    f = pd.Series(coef, index=X.columns)
    residual = returns - X @ f
    e = residual[fit].to_numpy()
    w = sw ** 2
    denom = np.sum(w * (y - np.average(y, weights=w)) ** 2)
    diagnostics = {"n_obs": int(fit.sum()), "r2": float(1 - np.sum(w * e ** 2) / denom) if denom > 0 else np.nan,
                   "condition_number": float(np.linalg.cond(reduced)),
                   "industry_constraint_error": float(np.dot(share, coef[1:p + 1]))}
    return f, residual, X, complete, diagnostics


def carino_link(contributions, portfolio, benchmark=None):
    """Running geometric contribution linking; sums to Rp-Rb at every date."""
    rp = portfolio.reindex(contributions.index).astype(float)
    rb = pd.Series(0., index=rp.index) if benchmark is None else benchmark.reindex(rp.index)
    if rp.isna().any() or rb.isna().any() or (rp <= -1).any() or (rb <= -1).any():
        raise ValueError("Carino requires complete returns > -1")
    def k(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        out = np.empty_like(a)
        same = np.abs(a - b) < 1e-12
        out[same] = 1 / (1 + a[same])
        out[~same] = (np.log1p(a[~same]) - np.log1p(b[~same])) / (a[~same] - b[~same])
        return out
    kt = k(rp, rb)
    K = k((1 + rp).cumprod() - 1, (1 + rb).cumprod() - 1)
    return contributions.mul(kt, axis=0).cumsum().div(K, axis=0)


def risk_decomposition(weights, exposure, factor_history, specific_history, min_obs=60):
    """Prior-data-only ex-ante variance proxy, daily covariance, diagonal specifics."""
    names = exposure.columns
    fh = factor_history.reindex(columns=names).dropna()
    if len(fh) < min_obs or len(specific_history) < min_obs:
        return {"annual_volatility": np.nan, "factor_variance": np.nan,
                "specific_variance": np.nan, "status": "insufficient_prior_history"}
    F = fh.cov().to_numpy()
    b = exposure.T @ weights
    sv = specific_history.reindex(columns=weights.index).var(ddof=1)
    held = weights.ne(0)
    if sv[held].isna().any():
        return {"annual_volatility": np.nan, "status": "missing_specific_history"}
    vf = float(b.to_numpy() @ F @ b.to_numpy())
    vs = float((weights.pow(2) * sv.fillna(0)).sum())
    return {"annual_volatility": float(np.sqrt(max(0, vf + vs) * 252)),
            "factor_variance": vf, "specific_variance": vs, "status": "estimated_prior_only"}


def analyze(backtest, vwap_adj, universe, exposures, market_cap, benchmark_weights=None, min_obs=30):
    """Return portfolio/benchmark/active exposures, linked PnL, rolling risk.

    Default benchmark is signal-date market-cap weights on the same universe.
    Every modeled interval uses the exact portfolio VWAP returns, not close returns.
    """
    cal = vwap_adj.index
    interval = vwap_adj / vwap_adj.shift(1) - 1
    styles = [k for k in exposures if k != "industry"]
    industry = exposures["industry"]
    fr, specific, diagnostics, designs = {}, {}, {}, {}
    # Fit history before OOS too, so OOS risk uses only earlier observations.
    first = int(cal.searchsorted(backtest.daily.index[0]))
    begin = max(2, first - 252)
    last = int(cal.searchsorted(backtest.daily.index[-1]))
    for idx in range(begin, last + 1):
        d, signal_date = cal[idx], cal[idx - 2]
        style = pd.DataFrame({k: exposures[k].loc[signal_date] for k in styles})
        try:
            f, u, X, complete, diag = cross_section(style, industry.loc[signal_date],
                market_cap.loc[signal_date], interval.loc[d], universe.loc[signal_date], min_obs)
        except ValueError as e:
            diagnostics[d] = {"error": str(e)}
            if d in backtest.daily.index:
                raise ValueError(f"Barra {d}: {e}") from e
            continue
        fr[d], specific[d], diagnostics[d] = f, u, diag
        # Retain designs for report dates only, avoiding a full T*N*K cube.
        if d in backtest.daily.index:
            designs[d] = (X, complete, signal_date)
    factor_returns = pd.DataFrame(fr).T.fillna(0).sort_index()
    spec = pd.DataFrame(specific).T.sort_index()
    pc, bc, pe, be, risk, br = {}, {}, {}, {}, {}, {}
    for d in backtest.daily.index:
        if d not in designs:
            raise ValueError(f"risk model unavailable on portfolio date {d}")
        X, complete, sig = designs[d]
        f = factor_returns.loc[d].reindex(X.columns)
        w = backtest.weights.loc[d]
        if ((w != 0) & ~complete).any():
            raise ValueError(f"missing held stock risk exposure at {d}")
        if benchmark_weights is None:
            bw = market_cap.loc[sig].where(universe.loc[sig].eq(1) & complete, 0).fillna(0)
            bw = bw / bw.sum()
        else:
            # Supplied rows are already interval holdings, indexed by exit dates.
            bw = benchmark_weights.loc[d].reindex(w.index).fillna(0)
            if ((bw != 0) & ~complete).any():
                raise ValueError(f"missing benchmark exposure at {d}")
        r = interval.loc[d]
        if (bw.ne(0) & r.isna()).any():
            raise ValueError(f"missing benchmark held return {d}")
        x = X.fillna(0)
        pe[d], be[d] = x.T @ w, x.T @ bw
        p, b = pe[d] * f, be[d] * f
        gross_p, gross_b = float((w * r.fillna(0)).sum()), float((bw * r.fillna(0)).sum())
        p["Specific"] = gross_p - p.sum()
        b["Specific"] = gross_b - b.sum()
        p["Costs"] = -float(backtest.daily.loc[d, "cost"] + backtest.daily.loc[d, "borrow_cost"])
        b["Costs"] = 0.
        pc[d], bc[d], br[d] = p, b, gross_b
        # At entry d-1, only returns with exit strictly before d-1 are known.
        prior_end = cal[cal.get_loc(d) - 1]
        fh = factor_returns.loc[factor_returns.index < prior_end].tail(252)
        sh = spec.loc[spec.index < prior_end].tail(252)
        risk[d] = risk_decomposition(w, x, fh, sh)
    port_c = pd.DataFrame(pc).T.fillna(0)
    bench_c = pd.DataFrame(bc).T.reindex(columns=port_c.columns).fillna(0)
    benchmark = pd.Series(br).reindex(backtest.daily.index)
    port_ret = backtest.daily.net_return
    if not np.allclose(port_c.sum(axis=1), port_ret, atol=1e-10):
        raise AssertionError("portfolio attribution does not reconcile")
    if not np.allclose(bench_c.sum(axis=1), benchmark, atol=1e-10):
        raise AssertionError("benchmark attribution does not reconcile")
    linked = carino_link(port_c - bench_c, port_ret, benchmark)
    expected = (1 + port_ret).cumprod() - (1 + benchmark).cumprod()
    if not np.allclose(linked.sum(axis=1), expected, atol=1e-8):
        raise AssertionError("geometric attribution does not reconcile")
    return {"factor_returns": factor_returns, "specific_returns": spec,
            "portfolio_exposures": pd.DataFrame(pe).T.fillna(0),
            "benchmark_exposures": pd.DataFrame(be).T.fillna(0),
            "active_exposures": pd.DataFrame(pe).T.fillna(0) - pd.DataFrame(be).T.fillna(0),
            "portfolio_contributions": port_c, "benchmark_contributions": bench_c,
            "active_contributions": port_c - bench_c, "active_linked": linked,
            "portfolio_linked": carino_link(port_c, port_ret),
            "risk": pd.DataFrame(risk).T,
            "regression_diagnostics": pd.DataFrame(diagnostics).T,
            "benchmark_returns": benchmark.to_frame("benchmark_return")}
