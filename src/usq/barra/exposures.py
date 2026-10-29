"""US point-in-time inputs for the original descriptor/exposure construction."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .core import config
from .core.descriptors import market_return, price_descriptors
from .core.exposures import style_exposures


def core_hashes():
    root = Path(__file__).parent / 'core'
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def annual_growth_panels(reports, dates, codes, years=5, min_years=3):
    """First-publication fiscal-year slopes, observable at available_date only.

    available_date is the first trading date usable at signal close, prepared by
    the data team from filing timestamps. It is not the fiscal period end.
    """
    need = {'sid', 'fiscal_year', 'available_date', 'earnings', 'revenue'}
    if not need.issubset(reports): raise ValueError(f'annual_fundamentals requires {sorted(need)}')
    r = reports.copy(); r['sid'] = r.sid.astype(str); r['available_date'] = r.available_date.astype(str)
    pd.to_datetime(r.available_date, format='%Y%m%d', errors='raise')
    r['fiscal_year'] = pd.to_numeric(r.fiscal_year, errors='raise').astype(int)
    r = r.sort_values(['sid', 'available_date', 'fiscal_year'], kind='stable').drop_duplicates(['sid', 'fiscal_year'], keep='first')
    out = {k: pd.DataFrame(np.nan, index=dates, columns=codes) for k in ('EGRO', 'SGRO')}
    for sid, g in r.groupby('sid', sort=False):
        if sid not in codes: continue
        known, events = {}, {k: {} for k in out}
        for available, block in g.groupby('available_date', sort=True):
            for row in block.itertuples(): known[row.fiscal_year] = (row.earnings, row.revenue)
            latest = max(known)
            for j, key in enumerate(out):
                pairs = sorted((yr, values[j]) for yr, values in known.items()
                               if latest - years < yr <= latest and pd.notna(values[j]) and np.isfinite(values[j]))
                value = np.nan
                if len(pairs) >= min_years:
                    x, y = np.asarray(pairs, float).T; scale = np.abs(y).mean()
                    if scale > 0: value = np.polyfit(x, y, 1)[0] / scale
                events[key][available] = value
        for key in out:
            s = pd.Series(events[key], dtype=float)
            out[key][sid] = s.sort_index().reindex(dates, method='ffill')
    return out


def build_exposures(store, dates=None, descriptor_dir=None):
    price = store.load('close_adj')
    dates = price.index if dates is None else pd.Index(dates)
    if not dates.isin(price.index).all(): raise ValueError('exposure dates absent from calendar')
    cap = store.load('market_cap', price).where(lambda x: x > 0)
    universe = store.load('universe', price).eq(1)
    ret = (price / price.shift(1) - 1).replace([np.inf, -np.inf], np.nan)
    # Original descriptor expects percentage turnover. Project input is a ratio
    # of shares traded to total shares; convert exactly once at the boundary.
    turnover = store.load('volume', price) / store.load('shares_out', price).where(lambda x: x > 0) * 100
    desc = price_descriptors(ret, cap, turnover, market_return(ret, cap, universe), list(dates))
    me = cap.loc[dates]
    def load(name):
        return store.load(name, price).loc[dates] if store.path(name).exists() else None
    book, earnings, ocf = load('book_equity'), load('earnings_ttm'), load('operating_cashflow_ttm')
    liabilities, assets = load('total_liabilities'), load('total_assets')
    debt, preferred = load('long_term_debt'), load('preferred_equity')
    if book is not None: desc['BTOP'] = book / me
    if earnings is not None: desc['ETOP'] = earnings / me
    if ocf is not None: desc['CETOP'] = ocf / me
    if liabilities is not None and assets is not None: desc['DTOA'] = liabilities / assets.where(lambda x: x > 0)
    # Unknown debt/preferred amounts are not silently treated as zero.
    if debt is not None and preferred is not None:
        desc['MLEV'] = (me + preferred + debt) / me
        if book is not None: desc['BLEV'] = ((book + preferred + debt) / book).where(book > 0)
    annual = store.root / 'annual_fundamentals.parquet'
    if annual.exists(): desc.update(annual_growth_panels(pd.read_parquet(annual), dates, price.columns))
    if descriptor_dir:
        from usq.data import PanelStore
        extra = PanelStore(descriptor_dir)
        valid = {n for spec in config.STYLE_SPEC.values() for n, _ in spec}
        for p in sorted(extra.root.glob('*.pkl')):
            if p.stem not in valid: raise ValueError(f'unknown descriptor {p.stem}')
            if p.stem in desc: raise ValueError(f'duplicate descriptor {p.stem}; use one audited source')
            desc[p.stem] = extra.load(p.stem, price).loc[dates]
    desc = {k: v.replace([np.inf, -np.inf], np.nan) for k, v in desc.items()}
    factors = style_exposures(desc, cap.loc[dates], universe.loc[dates])
    meta = {'model': 'US adaptation of the reference CNE5-style research model',
            'engine': 'reference_barra_v1', 'core_sha256': core_hashes(),
            'styles': list(factors), 'style_spec': config.STYLE_SPEC,
            'available_descriptors': list(desc),
            'missing_descriptors': sorted({n for spec in config.STYLE_SPEC.values() for n, _ in spec} - set(desc)),
            'missing_styles': [s for s in config.STYLE_FACTORS if s not in factors],
            'standardization': 'cap-weighted mean; equal-weight population std ddof=0',
            'orthogonalization': config.ORTHOGONALIZE, 'orthogonalization_cap_power': config.ORTHO_CAP_POWER,
            'rf_annual': config.RF_ANNUAL, 'regression_max_abs_return': config.REGRESSION_MAX_ABS_RETURN,
            'return_basis_for_attribution': 'adjusted VWAP T+1 to T+2; exposure known at signal T',
            'turnover_input': 'volume/shares_out ratio converted to percentage for original descriptors',
            'annual_growth': 'last five known fiscal years; minimum three; first publication only'}
    factors['industry'] = store.load('industry', price, numeric=False).loc[dates]
    return factors, meta
