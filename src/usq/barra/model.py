"""Persist the reference-style US model independently of strategy positions."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .core import config
from .core.model import cross_section as original_cross_section, design_matrix
from .core.exposures import to_long
from .exposures import build_exposures, core_hashes
from usq.utils import sha256, json_dump


@dataclass
class Model:
    exposures: pd.DataFrame
    factor_returns: pd.DataFrame
    returns: pd.DataFrame
    alignment: pd.DataFrame
    diagnostics: pd.DataFrame
    styles: list[str]
    meta: dict


def fit_cross_section(exp, ret, styles, min_obs=30):
    """Original WLS, with explicit empty/rank guards before its numerical call."""
    fit = (exp.industry.notna() & exp.estu.eq(True) & ret.notna() & np.isfinite(ret)
           & exp.cap.gt(0) & np.isfinite(exp.cap) & exp[styles].notna().all(axis=1)
           & np.isfinite(exp[styles]).all(axis=1) & ret.abs().lt(config.REGRESSION_MAX_ABS_RETURN))
    industries = sorted(exp.loc[fit, 'industry'].unique())
    if fit.sum() < min_obs or not industries: raise ValueError('insufficient risk-regression observations')
    X = design_matrix(exp.loc[exp.industry.notna()], styles, industries)
    xf = X.reindex(exp.index).loc[fit].to_numpy(float)
    if fit.sum() <= xf.shape[1] + 1 or np.linalg.matrix_rank(xf) < xf.shape[1] - 1:
        raise ValueError('underdetermined/collinear risk regression')
    return original_cross_section(exp, ret, styles)


def input_signature(store, start, end, min_obs, descriptor_dir=None):
    files = list(sorted(store.root.glob('*.pkl')))
    annual = store.root / 'annual_fundamentals.parquet'
    if annual.exists(): files.append(annual)
    extra = sorted(Path(descriptor_dir).glob('*.pkl')) if descriptor_dir else []
    root = Path(__file__).parent
    return {'start': start, 'end': end, 'min_obs': min_obs,
            'data_sha256': {p.name: sha256(p) for p in files},
            'descriptor_sha256': {p.name: sha256(p) for p in extra},
            'core_sha256': core_hashes(),
            'adapter_sha256': {n: sha256(root/n) for n in ('model.py', 'exposures.py')},
            'rf_annual': config.RF_ANNUAL,
            'return_basis': 'VWAP_adj[d]/VWAP_adj[d-1]-1; exposure at d-2'}


def load_model(path):
    root = Path(path); meta = json.loads((root/'meta.json').read_text())
    for name, digest in meta['artifact_sha256'].items():
        if sha256(root/name) != digest: raise ValueError(f'incomplete/changed model artifact {name}; rebuild model')
    return Model(pd.read_parquet(root/'exposures.parquet'), pd.read_parquet(root/'factor_returns.parquet'),
                 pd.read_parquet(root/'asset_returns.parquet'), pd.read_parquet(root/'alignment.parquet'),
                 pd.read_parquet(root/'diagnostics.parquet'), meta['styles'], meta)


def build_model(store, path, start='20220101', end='20251231', min_obs=30, descriptor_dir=None, force=False):
    root = Path(path)
    signature = input_signature(store, start, end, min_obs, descriptor_dir)
    if (root/'meta.json').exists() and not force:
        model = load_model(root)
        if model.meta['input_signature'] != signature:
            raise ValueError('model inputs/settings changed; choose a new model_dir or run barra-build --force')
        return model
    price = store.load('vwap_adj'); cal = price.index
    positions = [i for i, d in enumerate(cal) if start <= d <= end and i >= 2]
    if not positions: raise ValueError('model range outside input calendar')
    signals = cal[[i-2 for i in positions]]
    factors, meta = build_exposures(store, signals, descriptor_dir)
    universe = store.load('universe', price).eq(1)
    cap = store.load('market_cap', price)
    # Prefix protects style/market names from industry-name collisions.
    industry = factors.pop('industry').map(lambda x: 'Industry:'+str(x) if pd.notna(x) else x)
    exp = to_long(factors, industry, cap.loc[signals], universe.loc[signals])
    returns = (price / price.shift(1)-1).replace([np.inf,-np.inf],np.nan)
    rows, specs, diag, align = {}, {}, {}, {}
    for i in positions:
        d, sig, entry = cal[i], cal[i-2], cal[i-1]
        align[d] = {'signal_date': sig, 'entry_date': entry}
        try:
            cs = fit_cross_section(exp.loc[sig], returns.loc[d], meta['styles'], min_obs)
        except ValueError as e:
            diag[d] = {'n_obs': 0, 'error': str(e)}
            continue
        rows[d], specs[d] = cs.factor_returns, cs.specific_returns
        diag[d] = {'n_obs': cs.n_obs, 'r2': cs.r2, 'error': None}
    if not rows: raise ValueError('no estimable model dates; inspect warmup, descriptors and universe')
    root.mkdir(parents=True,exist_ok=True)
    artifacts = {'exposures.parquet': exp, 'factor_returns.parquet': pd.DataFrame(rows).T.sort_index(),
                 'specific_returns.parquet': pd.DataFrame(specs).T.sort_index().astype('float32'),
                 'asset_returns.parquet': returns, 'diagnostics.parquet': pd.DataFrame(diag).T.sort_index(),
                 'alignment.parquet': pd.DataFrame(align).T.sort_index()}
    for name, frame in artifacts.items():
        temp = root/(name+'.tmp'); frame.to_parquet(temp); temp.replace(root/name)
    meta.update(input_signature=signature, requested_range=[start,end],
                model_range=[min(rows),max(rows)], successful_dates=len(rows), skipped_dates=len(diag)-len(rows),
                artifact_sha256={n:sha256(root/n) for n in artifacts},
                missing_style_policy='excluded from estimation; held exposure filled with 0 and reported',
                missing_held_return_policy='raise; never fill zero',
                timing='signal-date exposure stored unchanged; adapter maps it to entry-date for reference attribute()',
                risk_covariance='not part of the reference return-attribution workflow')
    json_dump(meta,root/'meta.json')
    return load_model(root)
