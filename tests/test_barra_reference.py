import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest
from usq.barra.core import config
from usq.barra.core.descriptors import price_descriptors, market_return
from usq.barra.core.exposures import style_exposures
from usq.barra.core.preprocess import standardize, combine, orthogonalize
from usq.barra.exposures import annual_growth_panels
from usq.barra.model import Model, build_model, load_model, fit_cross_section
from usq.barra.api import interval_weights, attribute_positions
from usq.barra.attribution import attribute_weights
from usq.data import PanelStore
from usq.report import save_barra

FIXTURE=json.loads((Path(__file__).parent/'fixtures/barra_reference.json').read_text())


def test_reference_core_sources_are_byte_identical():
    import usq.barra.core as core
    for n,digest in FIXTURE['source_sha256'].items():
        assert hashlib.sha256((Path(core.__file__).parent/f'{n}.py').read_bytes()).hexdigest()==digest


def test_descriptors_exposures_and_regression_match_unmodified_source_fixture():
    rng=np.random.default_rng(FIXTURE['numerical']['seed'])
    dates=pd.bdate_range('2019-01-01',periods=620).strftime('%Y%m%d');cols=pd.Index([f'S{i:03d}' for i in range(60)])
    market=rng.normal(.0002,.008,(620,1));ret=pd.DataFrame(market+rng.normal(0,.012,(620,60)),index=dates,columns=cols)
    cap=pd.DataFrame(rng.lognormal(22,1,(620,60)),index=dates,columns=cols)
    turn=pd.DataFrame(rng.lognormal(0,.3,(620,60)),index=dates,columns=cols);universe=cap.gt(0);wanted=dates[-2:]
    desc=price_descriptors(ret,cap,turn,market_return(ret,cap,universe),list(wanted))
    for key in ['BTOP','ETOP','CETOP','EGRO','SGRO','MLEV','DTOA','BLEV']:
        desc[key]=pd.DataFrame(rng.normal(1,.3,(2,60)),index=wanted,columns=cols)
    for key,d in desc.items(): np.testing.assert_allclose(d,FIXTURE['numerical']['descriptors'][key],rtol=1e-10,atol=1e-12)
    factors=style_exposures(desc,cap.loc[wanted],universe.loc[wanted])
    assert list(factors)==config.STYLE_FACTORS
    for key,d in factors.items(): np.testing.assert_allclose(d,FIXTURE['numerical']['factors'][key],rtol=1e-10,atol=1e-11)
    exp=pd.DataFrame({n:f.iloc[-1] for n,f in factors.items()});exp['industry']=np.where(np.arange(60)%2,'Tech','Other')
    exp['cap']=cap.iloc[-1];exp['estu']=True
    r=pd.Series(rng.normal(.001,.01,60),index=cols);cs=fit_cross_section(exp,r,list(factors))
    assert list(cs.factor_returns.index)==FIXTURE['numerical']['factor_names']
    np.testing.assert_allclose(cs.factor_returns,FIXTURE['numerical']['factor_returns'],atol=1e-12)
    np.testing.assert_allclose(cs.specific_returns,FIXTURE['numerical']['specific_returns'],atol=1e-12)
    assert cs.r2==pytest.approx(FIXTURE['numerical']['r2'],abs=1e-12)


def test_reference_standardization_combination_and_orthogonalization():
    rng=np.random.default_rng(9)
    x=pd.DataFrame(rng.normal(size=(3,50)));cap=pd.DataFrame(rng.lognormal(3,1,(3,50)))
    z=standardize(x,cap)
    assert np.allclose((z*cap).sum(axis=1),0,atol=1e-10)
    assert np.allclose(z.std(axis=1,ddof=0),1)
    a=x.copy();a.iloc[0,0]=np.nan
    assert combine([(a,.74),(x+2,.26)]).iloc[0,0]==pytest.approx(x.iloc[0,0]+2)
    res=orthogonalize(2*x+x**2,[x],cap**.5)
    assert np.allclose((res*x*cap**.5).sum(axis=1),0,atol=1e-9)


def test_price_descriptors_use_no_future_observations():
    rng=np.random.default_rng(7);dates=pd.bdate_range('2018-01-01',periods=620).strftime('%Y%m%d')
    ret=pd.DataFrame(rng.normal(0,.01,(620,8)),index=dates);cap=ret*0+1e9;turn=ret*0+1.
    wanted=list(dates[580:585]);m=market_return(ret,cap,cap.gt(0))
    a=price_descriptors(ret,cap,turn,m,wanted)
    ret.iloc[590:]=.5;turn.iloc[590:]=1000;cap.iloc[590:]=1e15
    b=price_descriptors(ret,cap,turn,market_return(ret,cap,cap.gt(0)),wanted)
    for k in a: pd.testing.assert_frame_equal(a[k],b[k])


def test_growth_uses_available_fiscal_year_reports_and_ignores_revisions():
    dates=pd.Index(['20220301','20220401','20230401','20240401'])
    r=pd.DataFrame({'sid':['A']*5,'fiscal_year':[2018,2019,2020,2021,2020],
        'available_date':['20190301','20200301','20210301','20230301','20240301'],
        'earnings':[10.,12.,14.,999.,9999.],'revenue':[100.,120.,140.,999.,9999.]})
    a=annual_growth_panels(r,dates,pd.Index(['A']))
    assert a['EGRO'].loc['20220301','A']==pytest.approx(2/12)
    assert a['EGRO'].loc['20220401','A']==pytest.approx(2/12)
    changed=r.copy();changed.loc[3,'earnings']=-1e8
    b=annual_growth_panels(changed,dates,pd.Index(['A']))
    pd.testing.assert_frame_equal(a['EGRO'].iloc[:2],b['EGRO'].iloc[:2])
    assert a['EGRO'].iloc[-1,0]==a['EGRO'].iloc[-2,0]


def toy_model():
    rng=np.random.default_rng(51);days=pd.Index(['20230102','20230103','20230104','20230105','20230106'])
    ids=pd.Index([f'A{i}' for i in range(50)])
    exposures={}
    for j,d in enumerate(days):
        exposures[d]=pd.DataFrame({'Size':rng.normal(size=50)+j,'Beta':rng.normal(size=50),
            'industry':np.where(np.arange(50)%2,'Industry:Tech','Industry:Other'),
            'cap':rng.lognormal(10,1,50),'estu':True},index=ids)
    returns=pd.DataFrame(rng.normal(.001,.01,(5,50)),index=days,columns=ids)
    alignment=pd.DataFrame({'signal_date':days[:3],'entry_date':days[1:4]},index=days[2:])
    rows={}
    for d,sig in alignment.signal_date.items(): rows[d]=fit_cross_section(exposures[sig],returns.loc[d],['Size','Beta']).factor_returns
    return Model(pd.concat(exposures,names=['trade_date','ts_code']),pd.DataFrame(rows).T,returns,alignment,
                 pd.DataFrame(index=days[2:]),['Size','Beta'],{})


def test_holdings_attribution_uses_signal_exposures_not_entry_close_and_reconciles_costs():
    model=toy_model();dates=model.factor_returns.index
    w=pd.DataFrame(1/50,index=dates,columns=model.returns.columns)
    w.iloc[:,25:]=0;w*=2
    b=pd.DataFrame(1/50,index=dates,columns=w.columns)
    costs=pd.Series(.0005,index=dates)
    result=attribute_weights(w,model,b,costs)
    sig=model.alignment.iloc[0].signal_date
    expected=float(model.exposures.loc[sig,'Size']@w.iloc[0])
    assert result['portfolio_exposures'].iloc[0].Size==pytest.approx(expected)
    rp=(w*model.returns.loc[dates]).sum(axis=1)-costs;rb=(b*model.returns.loc[dates]).sum(axis=1)
    assert np.allclose(result['portfolio_contributions'].sum(axis=1),rp)
    assert np.allclose(result['active_contributions'].sum(axis=1),rp-rb)
    assert np.allclose(result['active_linked'].sum(axis=1),(1+rp).cumprod()-(1+rb).cumprod())
    assert result['groups'].linked_contribution.sum()==pytest.approx((1+rp).prod()-(1+rb).prod())
    assert result['groups'].loc['Costs','linked_contribution']<0


def test_missing_held_marks_raise_and_missing_styles_are_reported():
    m=toy_model();w=pd.DataFrame(1/50,index=m.factor_returns.index,columns=m.returns.columns)
    d=w.index[0];sid=w.columns[0];sig=m.alignment.loc[d,'signal_date']
    m.exposures.loc[(sig,sid),'Beta']=np.nan
    res=attribute_weights(w,m)
    assert res['exposure_coverage'].loc[d,'held_with_zero_filled_style']==1
    m.returns.loc[d,sid]=np.nan
    with pytest.raises(ValueError,match='missing held return'): attribute_weights(w,m)


@pytest.mark.parametrize('long_short',[False,True])
def test_snapshot_drift_preserves_cash_and_long_short_leverage(long_short):
    m=toy_model();ids=m.returns.columns
    m.returns.loc[:]=0;m.returns.loc['20230104',ids[0]]=.1
    snap=pd.DataFrame(0.,index=['20230102'],columns=ids);snap.iloc[0,0]=.5
    if long_short:snap.iloc[0,1]=-.5
    w=interval_weights(snap,m)
    assert w.loc['20230104',ids[0]]==pytest.approx(.5)
    assert w.loc['20230105',ids[0]]==pytest.approx(.55/1.05)
    assert w.loc['20230105',ids[1]]==pytest.approx(-.5/1.05 if long_short else 0)
    with pytest.raises(ValueError,match='outside'):interval_weights(snap.rename(index={'20230102':'20100101'}),m)


def test_holdings_api_long_form_active_and_absolute():
    m=toy_model();pos=pd.DataFrame({'trade_date':'20230102','sid':m.returns.columns[:10],'weight':.1})
    result=attribute_positions(pos,m)
    assert np.allclose(result['active_exposures'],result['portfolio_exposures']-result['benchmark_exposures'])
    absolute=attribute_positions(pos,m,benchmark=None)
    assert absolute['summary'].loc['Total','linked_contribution']==pytest.approx((1+absolute['daily_returns'].portfolio).prod()-1)
    w=interval_weights(pos,m)
    with pytest.raises(ValueError,match='missing trading dates'):interval_weights(w.drop(w.index[1]),m,timing='interval')


def model_store(tmp_path):
    rng=np.random.default_rng(6);dates=pd.bdate_range('2019-01-01',periods=620).strftime('%Y%m%d');ids=[f'S{i}' for i in range(50)]
    close=pd.DataFrame(50*np.exp(np.cumsum(rng.normal(.0001,.01,(620,50)),axis=0)),index=dates,columns=ids)
    store=PanelStore(tmp_path/'panels')
    for n,f in {'close_adj':close,'vwap_adj':close*.999,'market_cap':close*pd.Series(rng.lognormal(17,1,50),index=ids),
                'volume':pd.DataFrame(rng.lognormal(14,.4,close.shape),index=dates,columns=ids),
                'shares_out':close*0+1e8,'universe':close*0+1}.items():store.save(n,f)
    store.save('industry',pd.DataFrame(np.tile(np.where(np.arange(50)%2,'Tech','Other'),(620,1)),index=dates,columns=ids),numeric=False)
    return store,dates


def test_model_cache_is_reused_rejects_changed_input_and_checks_artifacts(tmp_path,monkeypatch):
    store,dates=model_store(tmp_path)
    with pytest.warns(UserWarning):m=build_model(store,tmp_path/'model',dates[-4],dates[-1])
    assert len(m.factor_returns)==4
    from usq.barra import model as module
    monkeypatch.setattr(module,'build_exposures',lambda *a,**kw:pytest.fail('cache must not rebuild exposures'))
    loaded=build_model(store,tmp_path/'model',dates[-4],dates[-1])
    pd.testing.assert_frame_equal(loaded.factor_returns,m.factor_returns)
    store.save('volume',store.load('volume')*2)
    with pytest.raises(ValueError,match='inputs/settings changed'):build_model(store,tmp_path/'model',dates[-4],dates[-1])
    path=tmp_path/'model/factor_returns.parquet';path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed model artifact'):load_model(tmp_path/'model')


def test_report_preserves_reference_panels_and_adds_performance_drawdown_and_costs(tmp_path):
    m=toy_model();w=pd.DataFrame(1/50,index=m.factor_returns.index,columns=m.returns.columns)
    res=attribute_weights(w,m,costs=pd.Series(.001,index=w.index))
    path=save_barra(res,None,tmp_path,{'has_benchmark':False})
    html=path.read_text();payload=json.loads(re.search(r'const D = (.*?);\n',html).group(1))
    assert 'chart-performance' in html and 'chart-drawdown' in html and 'chart-pick-expo' in html
    assert 'chart-heat' in html and 'chart-industry' in html and 'chart-style-unit' in html
    assert '申万' not in html and 'mk-corr' not in html and '<script src=' not in html
    nav=(1+res['daily_returns'].portfolio).cumprod();dd=nav/nav.cummax().clip(lower=1)-1
    assert payload['performance']['max_drawdown']==pytest.approx(dd.min())
    assert np.allclose(payload['performance']['portfolio'],nav-1,atol=5e-7)
    assert 'Costs' in payload['groups'] and all(c!='Costs' for c in [x['name'] for x in payload['industries']])


def test_us_adapter_turnover_units_and_optional_descriptor_gaps(tmp_path,monkeypatch):
    from usq.barra import exposures as module
    store,dates=model_store(tmp_path)
    original=module.price_descriptors;seen={}
    def capture(ret,cap,turnover,mkt,wanted):
        seen['turnover']=turnover
        return original(ret,cap,turnover,mkt,wanted)
    monkeypatch.setattr(module,'price_descriptors',capture)
    with pytest.warns(UserWarning):exposures,meta=module.build_exposures(store,dates[-3:])
    expected=store.load('volume')/store.load('shares_out')*100
    pd.testing.assert_frame_equal(seen['turnover'],expected)
    assert set(meta['missing_styles'])=={'BookToPrice','EarningsYield','Growth','Leverage'}
    assert 'EPFWD' in meta['missing_descriptors']
    assert meta['standardization'].endswith('ddof=0')


def test_long_short_attribution_market_exposure_and_cost_reconciliation():
    model=toy_model();dates=model.factor_returns.index
    w=pd.DataFrame(.02,index=dates,columns=model.returns.columns);w.iloc[:,25:]=-.02
    res=attribute_weights(w,model,costs=pd.Series(.001,index=dates))
    assert np.allclose(res['portfolio_exposures'].Market,0,atol=1e-12)
    assert np.allclose(res['portfolio_contributions'].sum(axis=1),res['daily_returns'].portfolio)
    assert res['summary'].loc['Total','linked_contribution']==pytest.approx((1+res['daily_returns'].portfolio).prod()-1)


def test_cli_barra_uses_saved_holdings_without_scores_or_rerunning_backtest(tmp_path,monkeypatch):
    import yaml
    from usq import workflow
    from usq.cli import main
    model=toy_model();w=pd.DataFrame(1/50,index=model.factor_returns.index,columns=model.returns.columns)
    ret=(w*model.returns.loc[w.index]).sum(axis=1)
    root=tmp_path/'run';(root/'backtest').mkdir(parents=True)
    w.to_pickle(root/'backtest/weights.pkl')
    pd.DataFrame({'net_return':ret,'cost':0.,'borrow_cost':0.}).to_csv(root/'backtest/daily.csv')
    cfg={'data_dir':str(tmp_path/'data'),'run_dir':str(root)}
    config_path=tmp_path/'config.yaml';config_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(workflow,'barra_model',lambda cfg:model)
    monkeypatch.setattr(workflow,'backtest',lambda *a,**kw:pytest.fail('Barra must not rerun score backtest'))
    main(['--config',str(config_path),'barra'])
    assert (root/'barra/report.html').exists()
    saved=pd.read_csv(root/'barra/daily_returns.csv',index_col=0)
    assert np.allclose(saved.portfolio,ret)
