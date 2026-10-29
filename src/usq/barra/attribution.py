from __future__ import annotations

import numpy as np
import pandas as pd


def cross_section(styles, industry, cap, returns, estu, min_obs=30):
    """Compatibility facade for the original joint WLS estimator."""
    from .model import fit_cross_section
    from .core.model import design_matrix
    exp = styles.assign(industry=industry.map(lambda x: 'Industry:'+str(x) if pd.notna(x) else x), cap=cap, estu=estu.eq(1))
    cs = fit_cross_section(exp, returns, list(styles), min_obs)
    inds = [c for c in cs.factor_returns.index if c not in styles and c != 'Market']
    X = design_matrix(exp, list(styles), inds)
    fit = estu.eq(1) & styles.notna().all(axis=1) & industry.notna() & cap.gt(0) & returns.notna() & returns.abs().lt(.35)
    shares = cap[fit].groupby(exp.industry[fit]).sum().reindex(inds); shares /= shares.sum()
    diag = {'n_obs': cs.n_obs, 'r2': cs.r2,
            'industry_constraint_error': float(shares @ cs.factor_returns[inds])}
    return cs.factor_returns, cs.specific_returns, X, industry.notna() & styles.notna().all(axis=1), diag


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


def attribute_weights(weights, model, benchmark_weights=None, costs=None, expected_net=None):
    """Reference attribution on interval holdings, aligned through cached dates.

    Missing style exposures use the original zero fill; held industry or return
    gaps raise. Costs are explicit, outside the four explanatory return groups.
    """
    from .core.attribution import attribute, carino_link as final_link, group
    if weights.empty or weights.index.has_duplicates or weights.columns.has_duplicates:
        raise ValueError('empty/duplicate holdings axes')
    dates = weights.index
    if not dates.isin(model.factor_returns.index).all(): raise ValueError('model does not cover every holdings date')
    if not np.isfinite(weights.to_numpy(float)).all(): raise ValueError('holdings must be finite; zero rows represent cash')
    unknown = weights.columns.difference(model.returns.columns)
    if len(unknown) and weights[unknown].ne(0).any().any(): raise ValueError('unknown held securities')
    w = weights.reindex(columns=model.returns.columns, fill_value=0.)
    if benchmark_weights is None:
        b = pd.DataFrame(0., index=dates, columns=w.columns)
    else:
        if not dates.isin(benchmark_weights.index).all(): raise ValueError('missing benchmark holdings dates')
        if not np.isfinite(benchmark_weights.to_numpy(float)).all(): raise ValueError('benchmark weights must be finite')
        extra = benchmark_weights.columns.difference(w.columns)
        if len(extra) and benchmark_weights[extra].ne(0).any().any(): raise ValueError('unknown benchmark securities')
        b = benchmark_weights.reindex(index=dates, columns=w.columns, fill_value=0.)
    coverage = {}
    shifted = []
    for d in dates:
        sig, entry = model.alignment.loc[d, ['signal_date', 'entry_date']]
        exp = model.exposures.loc[sig].reindex(w.columns)
        held = w.loc[d].ne(0) | b.loc[d].ne(0)
        if (held & (model.returns.loc[d].isna() | ~np.isfinite(model.returns.loc[d]))).any():
            raise ValueError(f'missing held return {d}; resolve corporate-action/delisting marks')
        if (held & exp.industry.isna()).any(): raise ValueError(f'missing held industry {d}')
        coverage[d] = {'held_with_zero_filled_style': int((held & exp[model.styles].isna().any(axis=1)).sum()),
                       'held_stocks': int(held.sum()), 'signal_date': sig, 'entry_date': entry}
        shifted.append(pd.concat({entry: exp}, names=['trade_date','ts_code']))
    # Reference attribute() looks up the preceding exposure row. Relabel only
    # this view to entry dates; its values remain the signal-date exposures.
    exposures = pd.concat(shifted).sort_index()
    f = model.factor_returns.reindex(dates)
    pc, pe = attribute(w, exposures, f, model.returns, model.styles)
    bc, be = attribute(b, exposures, f, model.returns, model.styles)
    cost = pd.Series(0.,index=dates) if costs is None else costs.reindex(dates)
    if cost.isna().any() or (cost < 0).any() or not np.isfinite(cost).all(): raise ValueError('invalid/missing costs')
    pc['Costs'] = -cost; pc['Total'] -= cost
    bc['Costs'] = 0.
    pc = pc.fillna(0.); bc = bc.reindex(columns=pc.columns, fill_value=0.).fillna(0.)
    rp, rb = pc.Total, bc.Total
    if expected_net is not None and not np.allclose(rp, expected_net.reindex(dates),atol=1e-10):
        raise ValueError('holdings/model return basis differs from strategy net return')
    if (rp <= -1).any() or (rb <= -1).any(): raise ValueError('return <= -100%; geometric attribution undefined')
    active = pc-bc
    for c in (pc,bc,active):
        if not np.allclose(c.drop(columns='Total').sum(axis=1),c.Total,atol=1e-10):
            raise AssertionError('daily attribution does not reconcile')
    target = active if benchmark_weights is not None else pc
    port = rp; bench = rb if benchmark_weights is not None else None
    linked = final_link(target,port,bench)
    grouped = group(linked.drop('Costs'),model.styles);grouped['Costs']=linked['Costs']
    running = carino_link(target.drop(columns='Total'),port,bench)
    if not np.allclose(running.iloc[-1],linked.reindex(running.columns),atol=1e-9):
        raise AssertionError('running and reference final Carino disagree')
    return {'factor_returns': f, 'portfolio_exposures': pe, 'benchmark_exposures': be,
            'active_exposures': pe-be, 'portfolio_contributions':pc.drop(columns='Total'),
            'benchmark_contributions':bc.drop(columns='Total'), 'active_contributions':active.drop(columns='Total'),
            'active_linked':carino_link(active.drop(columns='Total'),rp,rb),
            'portfolio_linked':carino_link(pc.drop(columns='Total'),rp),
            'benchmark_returns':rb.to_frame('benchmark_return'),
            'daily_returns':pd.DataFrame({'portfolio':rp,'benchmark':rb}),
            'daily_contrib':target, 'summary':pd.DataFrame({'linked_contribution':linked,
                'mean_exposure':(pe-be if benchmark_weights is not None else pe).mean().reindex(linked.index)}),
            'groups':grouped.to_frame('linked_contribution'), 'exposure_coverage':pd.DataFrame(coverage).T,
            'regression_diagnostics':model.diagnostics.reindex(dates)}


def cap_benchmark(model, dates):
    rows = {}
    for d in dates:
        exp = model.exposures.loc[model.alignment.loc[d,'signal_date']]
        w = exp.cap.where(exp.estu.eq(True) & exp.cap.gt(0),0).fillna(0.)
        if w.sum() <= 0: raise ValueError(f'empty market-cap benchmark {d}')
        rows[d] = w / w.sum()
    return pd.DataFrame(rows).T.fillna(0.)


def analyze(backtest, vwap_adj=None, universe=None, *, model, benchmark_weights=None):
    """Use one persisted model for any factor/score portfolio; never refit here."""
    bw = cap_benchmark(model,backtest.weights.index) if benchmark_weights is None else benchmark_weights
    return attribute_weights(backtest.weights,model,bw,
        costs=backtest.daily.cost+backtest.daily.borrow_cost,expected_net=backtest.daily.net_return)
