"""Holdings-only attribution against a prebuilt model, plus explicit timing."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .model import load_model
from .attribution import attribute_weights, cap_benchmark


def to_wide(positions):
    if isinstance(positions,(str,Path)):
        path = Path(positions)
        if path.suffix == '.parquet': positions = pd.read_parquet(path)
        elif path.suffix == '.pkl': positions = pd.read_pickle(path)
        elif path.suffix == '.csv':
            positions = pd.read_csv(path,dtype={'trade_date':str,'sid':str,'ts_code':str})
            if 'weight' not in positions.columns: positions = positions.set_index(positions.columns[0])
        else: raise ValueError('positions must be CSV, Parquet or trusted pickle')
    sid = 'sid' if 'sid' in positions else 'ts_code'
    if {'trade_date',sid,'weight'}.issubset(positions.columns):
        if positions[['trade_date',sid,'weight']].isna().any().any(): raise ValueError('missing position key or weight')
        positions = positions.copy();positions[sid]=positions[sid].astype(str)
        positions = positions.pivot_table(index='trade_date',columns=sid,values='weight',aggfunc='sum')
    wide = positions.copy();wide.index=wide.index.astype(str);wide.columns=wide.columns.astype(str)
    pd.to_datetime(wide.index,format='%Y%m%d',errors='raise')
    if wide.index.has_duplicates or wide.columns.has_duplicates: raise ValueError('duplicate position axes')
    wide=wide.apply(pd.to_numeric,errors='raise').fillna(0.).sort_index()
    if not np.isfinite(wide.to_numpy()).all(): raise ValueError('nonfinite position weight')
    return wide


def interval_weights(positions, model, timing='signal_close', start=None, end=None):
    wide = to_wide(positions)
    if wide.empty: raise ValueError('empty holdings')
    if timing == 'interval':
        out=wide.loc[(wide.index >= (start or wide.index[0])) & (wide.index <= (end or wide.index[-1]))]
        if out.empty: raise ValueError('empty attribution range')
        expected=model.alignment.index[(model.alignment.index>=out.index[0])&(model.alignment.index<=out.index[-1])]
        if not expected.isin(out.index).all(): raise ValueError('interval weights missing trading dates; use explicit zero rows for cash')
        return out
    if timing != 'signal_close': raise ValueError('timing must be signal_close or interval')
    cal=model.returns.index
    # Nontrading snapshot dates map to the preceding trading day, then enter at
    # the next trading day's VWAP. Reject the pre-calendar -1 wraparound case.
    idx=cal.searchsorted(wide.index,side='right')-1
    if (idx<0).any() or (idx+1>=len(cal)).any(): raise ValueError('snapshot outside model input calendar')
    wide.index=cal[idx+1];wide=wide[~wide.index.duplicated(keep='last')]
    unknown=wide.columns.difference(model.returns.columns)
    if len(unknown) and wide[unknown].ne(0).any().any(): raise ValueError('unknown held securities')
    wide=wide.reindex(columns=model.returns.columns,fill_value=0.)
    dates=model.alignment.index
    dates=dates[(dates>wide.index.min()) & (dates>=(start or dates[0])) & (dates<=(end or dates[-1]))]
    if dates.empty: raise ValueError('positions and model range do not overlap')
    current=None;rows={};want=set(dates)
    for d in cal[cal.searchsorted(wide.index.min()):cal.searchsorted(dates[-1],side='right')]:
        if current is not None:
            if d in want: rows[d]=current.copy()
            r=model.returns.loc[d]
            if (current.ne(0)&r.isna()).any(): raise ValueError(f'missing held return {d}')
            net_asset=1+float(current@r.fillna(0.))
            if net_asset<=0: raise ValueError('portfolio NAV exhausted while drifting snapshots')
            # Including implicit cash is essential for partial and long-short
            # portfolios; normalizing by sum(stock values) would change leverage.
            current=current*(1+r.fillna(0.))/net_asset
        if d in wide.index: current=wide.loc[d].copy()
    return pd.DataFrame(rows).T


def attribute_positions(positions, model, benchmark_positions=None, timing='signal_close',
                        benchmark_timing='signal_close', start=None, end=None, benchmark='cap_weighted',costs=None):
    if isinstance(model,(str,Path)): model=load_model(model)
    w=interval_weights(positions,model,timing,start,end)
    if benchmark_positions is not None:
        bw=interval_weights(benchmark_positions,model,benchmark_timing,w.index[0],w.index[-1])
    elif benchmark=='cap_weighted': bw=cap_benchmark(model,w.index)
    elif benchmark is None: bw=None
    else: raise ValueError('provide constituent benchmark positions or use cap_weighted/None')
    return attribute_weights(w,model,bw,costs=costs)
