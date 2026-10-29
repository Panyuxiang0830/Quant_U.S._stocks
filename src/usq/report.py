"""Self-contained offline HTML reports. All numbers originate in emitted CSVs."""
from __future__ import annotations
from html import escape
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from usq.utils import json_dump


def _dates(index):
    return pd.to_datetime(index.astype(str), format="%Y%m%d")


def _lines(frame, title, percent=False):
    fig = go.Figure()
    for col in frame:
        fig.add_trace(go.Scatter(x=_dates(frame.index), y=frame[col], name=str(col), connectgaps=False))
    fig.update_layout(title=title, template="plotly_white", height=430,
                      legend=dict(orientation="h"), margin=dict(l=60, r=25, t=65, b=55))
    if percent: fig.update_yaxes(tickformat=".1%")
    return fig


def write_html(path, title, figures, note="", tables=()):
    chunks = []
    for i, f in enumerate(figures):
        chunks.append(f.to_html(full_html=False, include_plotlyjs=True if i == 0 else False))
    chunks += [t.to_html(escape=True, float_format=lambda x: f"{x:.6f}") for t in tables]
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>{escape(title)}</title><style>body{{font:15px/1.7 system-ui,sans-serif;color:#172b40;background:#f5f7fa;margin:0}}
main{{max-width:1240px;margin:auto;padding:30px}}h1{{font-size:27px}}p{{max-width:1050px}}
.card{{background:white;border:1px solid #dbe1e8;margin:20px 0;padding:15px;overflow:auto}}
table{{border-collapse:collapse;font-size:13px}}th,td{{padding:7px 12px;border:1px solid #dbe1e8}}</style>
<main><h1>{escape(title)}</h1><p>{escape(note)}</p>'''
    html += "".join(f'<div class="card">{c}</div>' for c in chunks) + "</main></html>"
    Path(path).write_text(html, encoding="utf-8")


def save_backtest(result, group_returns, out_dir, title="Strategy backtest"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    result.daily.to_csv(out / "daily.csv", index_label="date")
    result.weights.to_pickle(out / "weights.pkl", protocol=4)
    json_dump(result.summary, out / "summary.json")
    group_returns.to_csv(out / "group_returns.csv", index_label="date")
    navcols = [k for k in ("gross_nav", "net_nav", "benchmark_nav", "relative_nav") if k in result.daily]
    figures = [_lines(result.daily[navcols], "Cumulative NAV  gross net benchmark"),
               _lines(result.daily[["drawdown"]], "Net portfolio drawdown  includes initial NAV", True)]
    means = group_returns.mean()
    fig = go.Figure(go.Bar(x=means.index, y=means, text=group_returns.count(), name="mean daily net return"))
    fig.update_layout(template="plotly_white", title="Score groups  low to high  daily net return and observation count", height=420)
    fig.update_yaxes(tickformat=".2%")
    figures.append(fig)
    gnav = (1 + group_returns.fillna(0)).cumprod().where(group_returns.notna())
    figures.append(_lines(gnav, "Score group NAV  gaps mean empty bins  cash across gaps"))
    figures.append(_lines(result.daily[["turnover", "cost", "borrow_cost"]], "Turnover and modeled costs", True))
    yearly = result.daily.net_return.groupby(result.daily.index.str[:4]).apply(lambda x: (1 + x).prod() - 1).to_frame("net_return")
    write_html(out / "report.html", title, figures,
        note="Research simulation. T close signal, T+1 adjusted VWAP entry, T+2 exit. Rows use exit dates. "
             "Costs apply to dollars traded. Missing held marks abort. Group ties stay together; empty bins have no observation. "
             "Adjusted price returns include dividends only when the supplied adjustment factor does. No cash/share/impact ledger.",
        tables=[pd.DataFrame(result.summary["net"], index=["net"]).T, yearly])
    return out / "report.html"


def save_barra(result, backtest, out_dir, metadata=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for key, frame in result.items(): frame.to_csv(out / f"{key}.csv", index_label="date")
    json_dump(metadata or {}, out / "model_metadata.json")
    ret = backtest.daily.net_return
    b = result["benchmark_returns"].iloc[:, 0]
    nav = pd.DataFrame({"portfolio": (1 + ret).cumprod(), "benchmark": (1 + b).cumprod()})
    nav["relative"] = nav.portfolio / nav.benchmark
    dd = nav[["portfolio", "benchmark"]] / nav[["portfolio", "benchmark"]].cummax().clip(lower=1) - 1
    styles = [c for c in result["active_exposures"] if c != "Market" and not c.startswith("Industry:")]
    figs = [_lines(nav, "Strategy benchmark and relative NAV"), _lines(dd, "Portfolio and benchmark drawdown", True),
            _lines(result["portfolio_exposures"][styles], "Portfolio style exposures"),
            _lines(result["benchmark_exposures"][styles], "Benchmark style exposures"),
            _lines(result["active_exposures"][styles], "Active style exposures"),
            _lines(result["active_linked"], "Geometrically linked active return contributions", True),
            _lines(result["portfolio_linked"], "Geometrically linked portfolio return contributions", True),
            _lines(result["risk"][["annual_volatility"]].astype(float), "Prior-data-only annual risk proxy", True)]
    monthly = result["active_exposures"][styles].groupby(result["active_exposures"].index.str[:6]).mean()
    fig = go.Figure(go.Heatmap(z=monthly.T.to_numpy(), x=monthly.index, y=styles, colorscale="RdBu", zmid=0))
    fig.update_layout(template="plotly_white", title="Monthly active style exposure", height=440)
    figs.append(fig)
    summary = pd.DataFrame({"portfolio_mean": result["portfolio_exposures"].mean(),
                            "benchmark_mean": result["benchmark_exposures"].mean(),
                            "active_mean": result["active_exposures"].mean()})
    write_html(out / "report.html", "US research risk model and return attribution", figs,
        note="Research proxies, not licensed MSCI Barra USE4. Exposures from signal T explain the identical VWAP T+1 to T+2 "
             "interval used by the strategy. Costs are a separate negative contribution. Active linked contributions sum to "
             "cumulative portfolio return minus cumulative benchmark return. Risk covariance uses strictly prior intervals.", tables=[summary])
    return out / "report.html"
