"""把 run_attribution 的输出目录渲染成一个自包含的 HTML 报告（无外部依赖）。"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .core import config

STYLE_CN = {
    "Size": "市值", "Beta": "贝塔", "Momentum": "动量", "ResidualVolatility": "残差波动",
    "NonLinearSize": "非线性市值", "BookToPrice": "账面市值比", "Liquidity": "流动性",
    "EarningsYield": "盈利收益率", "Growth": "成长", "Leverage": "杠杆",
}


def _k(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = a - b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (np.log1p(a) - np.log1p(b)) / diff
    return np.where(np.abs(diff) < 1e-12, 1.0 / (1.0 + a), out)


def running_linked(contrib: pd.DataFrame, rp: pd.Series, rb: pd.Series) -> pd.DataFrame:
    """每个日期 t 的 Carino 链接累计贡献：各列之和恰好等于截至 t 的累计超额。"""
    kt = _k(rp, rb)
    Kt = _k((1 + rp).cumprod() - 1, (1 + rb).cumprod() - 1)
    return contrib.mul(kt, axis=0).cumsum().div(Kt, axis=0)


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, dtype={0: str}).rename(columns={"Country": config.MARKET})
    df.index = df.index.astype(str)
    return df


def build_report(out_dir: Path, portfolio: str, benchmark: str | None) -> Path:
    out_dir = Path(out_dir)
    metadata = json.loads((out_dir / "model_metadata.json").read_text()) if (out_dir / "model_metadata.json").exists() else {}
    coverage = _read(out_dir / "exposure_coverage.csv")
    contrib = _read(out_dir / "daily_contrib.csv")
    dates = contrib.index
    rets = _read(out_dir / "daily_returns.csv").reindex(dates)
    port = _read(out_dir / "daily_portfolio_exposure.csv").reindex(dates)
    bench = _read(out_dir / "daily_benchmark_exposure.csv") if benchmark else None
    bench = bench.reindex(dates) if bench is not None else port * 0.0
    active = port - bench
    fr = _read(out_dir / "daily_factor_returns.csv")
    fr = fr.reindex(dates) if fr is not None else None

    rp = rets["portfolio"]
    rb = rets["benchmark"] if benchmark else pd.Series(0.0, index=dates)
    styles = [s for s in config.STYLE_FACTORS if s in contrib.columns]
    industries = [c for c in contrib.columns if c not in styles and c not in (config.MARKET, "Specific", "Costs", "Total")]
    cum = running_linked(contrib, rp, rb)
    groups = pd.DataFrame({
        "Industry": cum[industries].sum(axis=1), "Style": cum[styles].sum(axis=1),
        "Specific": cum["Specific"], "Market": cum[config.MARKET], "Costs": cum["Costs"], "Total": cum["Total"],
    })
    last = cum.iloc[-1]
    rnd = lambda s, d=6: [round(float(v), d) for v in s]  # noqa: E731
    unit_cum = (lambda c: ((1 + fr[c].fillna(0)).cumprod() - 1)) if fr is not None else (lambda c: pd.Series(np.nan, index=dates))
    months = sorted(set(d[:6] for d in dates))
    monthly = active[styles].groupby(active.index.str[:6]).mean().reindex(months)

    pnav, bnav = (1+rp).cumprod(), (1+rb).cumprod()
    pdd = pnav / pnav.cummax().clip(lower=1) - 1
    bdd = bnav / bnav.cummax().clip(lower=1) - 1
    payload = {
        "return_basis": metadata.get("portfolio_return_basis", "与回测同口径净收益，成本单列"),
        "missing_styles": metadata.get("missing_styles", []),
        "zero_filled_stock_days": int(coverage.held_with_zero_filled_style.sum()) if coverage is not None else 0,
        "performance": {"portfolio": rnd(pnav-1), "benchmark": rnd(bnav-1),
                        "relative": rnd(pnav/bnav-1), "portfolio_dd": rnd(pdd), "benchmark_dd": rnd(bdd),
                        "max_drawdown": float(pdd.min()), "benchmark_max_drawdown": float(bdd.min())},
        "portfolio": portfolio, "benchmark": benchmark or "无（绝对收益）", "has_bench": bool(benchmark),
        "start": dates[0], "end": dates[-1], "days": len(dates), "dates": list(dates),
        "port_total": float((1 + rp).prod() - 1), "bench_total": float((1 + rb).prod() - 1),
        "groups": {k: rnd(v) for k, v in groups.items()},
        "market": {"contrib": float(last[config.MARKET]), "port": float(port[config.MARKET].mean()),
                   "bench": float(bench[config.MARKET].mean()), "active": float(active[config.MARKET].mean()),
                   "unit": float(unit_cum(config.MARKET).iloc[-1])},
        "styles": [{
            "key": s, "name": STYLE_CN.get(s, s), "contrib": float(last[s]),
            "port": float(port[s].mean()), "bench": float(bench[s].mean()),
            "active": float(active[s].mean()), "active_last": float(active[s].iloc[-1]),
            "active_std": float(active[s].std()), "unit": float(unit_cum(s).iloc[-1]),
            "port_ts": rnd(port[s], 4), "bench_ts": rnd(bench[s], 4),
            "unit_ts": rnd(unit_cum(s)), "contrib_ts": rnd(cum[s]),
        } for s in styles],
        "months": [f"{m[:4]}-{m[4:]}" for m in months],
        "heat": [rnd(monthly[s], 4) for s in styles],
        "industries": [{"name": i, "contrib": float(last[i]), "active": float(active[i].mean()),
                        "port": float(port[i].mean()), "bench": float(bench[i].mean()),
                        "unit": float(unit_cum(i).iloc[-1])} for i in industries],
    }
    title = f"{portfolio} vs {benchmark}" if benchmark else portfolio
    html = TEMPLATE.replace("__TITLE__", escape(f"{title} · Barra 归因")).replace(
        "__DATA__", json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a;
  --pos: #e34948; --neg: #2a78d6; --mid: #f0efec;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70;
    --pos: #e66767; --neg: #3987e5; --mid: #383835;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --s1: #3987e5; --s2: #d95926; --s3: #199e70;
  --pos: #e66767; --neg: #3987e5; --mid: #383835;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 17px; margin: 0 0 4px; font-weight: 600; }
.sub { color: var(--ink-2); margin: 0 0 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 24px; font-weight: 600; margin-top: 2px; }
.tile.hero .value { font-size: 32px; }
.tile .note { color: var(--muted); font-size: 12px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.h3 { font-size: 15px; font-weight: 600; margin: 16px 0 2px; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; vertical-align: -1px; margin-right: 6px; }
.ramp { display: inline-block; width: min(160px, 40vw); height: 10px; border-radius: 3px; vertical-align: -1px; margin: 0 6px; }
@media (max-width: 760px) { .grid2, .grid3 { grid-template-columns: 1fr; } }
@media (min-width: 761px) and (max-width: 980px) { .grid3 { grid-template-columns: 1fr; } }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink-2); font-size: 13px; margin: 4px 0 8px; }
.key { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; vertical-align: -1px; margin-right: 6px; }
.chart { position: relative; width: 100%; }
.chart svg { display: block; width: 100%; overflow: visible; }
.tip { position: absolute; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); white-space: nowrap; display: none; z-index: 5; }
.tip b { font-weight: 600; }
.tip .row { display: flex; justify-content: space-between; gap: 16px; font-variant-numeric: tabular-nums; }
.explain { color: var(--ink-2); }
.explain p { margin: 6px 0; }
.explain code { background: var(--mid); padding: 1px 5px; border-radius: 4px; color: var(--ink); }
select { font: inherit; color: var(--ink); background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 4px 8px; }
.toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; color: var(--ink-2); }
details { margin-top: 8px; color: var(--ink-2); }
summary { cursor: pointer; }
table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 4px 8px; border-bottom: 1px solid var(--grid); color: var(--ink); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 500; }
.table-wrap { overflow-x: auto; }
</style>
</head>
<body>
<main>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
  <p class="sub" id="model-note"></p>
  <div class="tiles" id="tiles"></div>

  <section class="card">
    <h2>策略累计收益与回撤</h2>
    <p class="sub">收益直接来自持仓的同区间 VWAP 收益；策略成本单独计入。回撤高水位包含初始净值 1。</p>
    <div class="legend" id="legend-performance"></div>
    <div class="chart" id="chart-performance"></div>
    <div class="chart" id="chart-drawdown"></div>
  </section>
  <section class="card">
    <h2>收益来源（累计）</h2>
    <p class="sub">每个时点都做了 Carino 链接：市场、行业、风格、特异及成本的分项之和等于截至当日的累计收益差。</p>
    <div class="legend" id="legend-groups"></div>
    <div class="chart" id="chart-groups"></div>
  </section>

  <section class="card">
    <h2>风格暴露</h2>
    <p class="sub">暴露单位为横截面标准差：+1 表示比全市场（市值加权）平均高 1 个标准差。主动暴露 = 组合 − 基准。</p>
    <div class="legend" id="legend-dumbbell"></div>
    <div class="chart" id="chart-dumbbell"></div>
    <h3 class="h3">主动暴露的月度变化</h3>
    <p class="sub" style="margin:0 0 6px">每格为当月日均主动暴露；红色为偏高、蓝色为偏低，颜色越深偏离越大。</p>
    <div class="chart" id="chart-heat"></div>
    <div class="legend" id="legend-heat"></div>
    <details open><summary>暴露明细表</summary><div class="table-wrap" id="table-expo"></div></details>
  </section>

  <section class="card">
    <h2>风格收益：暴露 × 因子单位收益 = 贡献</h2>
    <p class="sub">左：各因子对收益的累计贡献（有基准时为超额）；右：因子本身在这段时间的累计单位收益（暴露为 +1 的纯因子组合收益）。</p>
    <div class="grid2">
      <div><div class="sub" style="margin:0 0 4px">累计贡献</div><div class="chart" id="chart-style-contrib"></div></div>
      <div><div class="sub" style="margin:0 0 4px">因子累计单位收益</div><div class="chart" id="chart-style-unit"></div></div>
    </div>
    <div class="toolbar" style="margin-top:16px">单个因子拆解：<select id="style-pick"></select></div>
    <div class="grid3">
      <div><div class="sub" style="margin:0">① 暴露（每日）</div><div class="legend" id="legend-pick"></div><div class="chart" id="chart-pick-expo"></div></div>
      <div><div class="sub" style="margin:0">② 因子单位收益（累计）</div><div class="chart" id="chart-pick-unit"></div></div>
      <div><div class="sub" style="margin:0">③ 对超额的贡献（累计）</div><div class="chart" id="chart-pick-contrib"></div></div>
    </div>
    <p class="sub" style="margin-top:8px">③ 是逐日的“① 中的主动暴露 × 当日因子收益”再累计，所以主动暴露长期为正且因子上涨时，贡献为正。</p>
  </section>

  <section class="card">
    <h2>行业（输入数据的历史分类）</h2>
    <p class="sub">行业暴露就是行业权重。左：累计贡献；右：平均主动权重（超配为正、低配为负）。按贡献排序。</p>
    <div class="chart" id="chart-industry"></div>
    <details><summary>表格视图</summary><div class="table-wrap" id="table-industry"></div></details>
  </section>

  <section class="card explain">
    <h2 style="color:var(--ink)">市场因子是什么？和 Beta 有什么区别？</h2>
    <p><b>市场因子是横截面回归中的共同收益项。</b>每只股票暴露为 <code>1</code>，组合暴露等于股票净仓位。它与行业、风格一起估计，数值不保证等于任何具体指数收益。</p>
    <p><b>Beta 是另一个风格因子。</b>先估计每只股票对市场收益的历史敏感度，再按截面规则标准化。其暴露因股票而异，和所有股票均为 1 的市场列具有不同含义。</p>
    <p><b>为什么要有行业？</b>同一行业的股票会一起涨跌（例如券商在政策日集体大涨），这部分既不是全市场的，也不是风格能解释的。不放行业因子，这些共振会被错算进风格或特异收益里。</p>
    <p><b>在归因里：</b>组合对市场因子的暴露 = 股票总仓位。组合和基准都满仓时主动暴露为 0，对超额贡献为 0；仓位不满时就是“仓位择时”的贡献：<code>(仓位 − 1) × 市场因子收益</code>。</p>
    <p id="market-line"></p>
  </section>
</main>
<div class="tip" id="tip"></div>
<script>
const D = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const signed = (x, d, suffix = "") => { const t = Math.abs(x).toFixed(d);
  return +t === 0 ? "0" + suffix : (x > 0 ? "+" : "−") + t + suffix; };
const pct = (v, d = 2) => signed(v * 100, d, "%");
const num = (v, d = 3) => signed(v, d);
const fmtDate = s => `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const tip = document.getElementById("tip");
function showTip(evt, html, host) {
  tip.innerHTML = html; tip.style.display = "block";
  const r = host.getBoundingClientRect(), tw = tip.offsetWidth, th = tip.offsetHeight;
  let x = evt.clientX + 14, y = evt.clientY + 14;
  if (x + tw > window.innerWidth - 8) x = evt.clientX - tw - 14;
  if (y + th > window.innerHeight - 8) y = evt.clientY - th - 14;
  tip.style.left = (x + window.scrollX) + "px"; tip.style.top = (y + window.scrollY) + "px";
}
const hideTip = () => { tip.style.display = "none"; };
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const ticks = []; for (let v = Math.floor(lo / step) * step; v <= hi + 1e-12; v += step) ticks.push(+v.toFixed(12));
  return ticks;
}
// 刻度小数位按步长决定；unit=100 表示百分比
function tickFmt(t, ticks, unit, fmt) {
  if (Math.abs(t) < 1e-12) return "0";
  const step = (ticks[1] - ticks[0]) * unit;
  let d = 0; while (d < 4 && Math.abs(step * 10 ** d - Math.round(step * 10 ** d)) > 1e-6) d++;
  return fmt(t, d);
}
// 圆角只在远离零线的一端
function barPath(x0, x1, y, h) {
  const r = Math.min(4, Math.abs(x1 - x0), h / 2), dir = x1 >= x0 ? 1 : -1;
  if (Math.abs(x1 - x0) < 0.5) return `M${x0},${y}h0.5v${h}h-0.5z`;
  const xe = x1 - dir * r;
  return `M${x0},${y} H${xe} Q${x1},${y} ${x1},${y + r} V${y + h - r} Q${x1},${y + h} ${xe},${y + h} H${x0} Z`;
}

/* ---------- 水平发散条形图 ---------- */
function divBars(host, rows, { fmt, unit = 100, labelW = 92, valueFmt, rowH = 26, title }) {
  host.innerHTML = "";
  const W = host.clientWidth, padR = 64, padL = 8 + 60, H = rows.length * rowH + 28;
  if (W <= 0) return;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": title || "" }, host);
  const vals = rows.map(r => r.value), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const x0 = labelW + padL, x1 = W - padR, sx = v => x0 + (v - lo) / ((hi - lo) || 1) * (x1 - x0);
  const ticks = niceTicks(lo, hi, 4);
  ticks.forEach(t => { if (t < lo - 1e-12 || t > hi + 1e-12) return;
    el("line", { x1: sx(t), x2: sx(t), y1: 4, y2: H - 22, stroke: css("--grid"), "stroke-width": 1 }, svg);
    const tx = el("text", { x: sx(t), y: H - 6, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, svg);
    tx.textContent = tickFmt(t, ticks, unit, fmt); });
  el("line", { x1: sx(0), x2: sx(0), y1: 2, y2: H - 22, stroke: css("--axis"), "stroke-width": 1 }, svg);
  rows.forEach((r, i) => {
    const y = 4 + i * rowH, bh = Math.min(16, rowH - 8), by = y + (rowH - bh) / 2 - 2;
    const lab = el("text", { x: labelW, y: by + bh / 2 + 4, "text-anchor": "end", fill: css("--ink-2"), "font-size": 12 }, svg);
    lab.textContent = r.label;
    el("path", { d: barPath(sx(0), sx(r.value), by, bh), fill: r.value >= 0 ? css("--pos") : css("--neg") }, svg);
    const vx = r.value >= 0 ? sx(r.value) + 6 : sx(r.value) - 6;
    const vt = el("text", { x: vx, y: by + bh / 2 + 4, "text-anchor": r.value >= 0 ? "start" : "end",
      fill: css("--ink"), "font-size": 11, "font-variant-numeric": "tabular-nums" }, svg);
    vt.textContent = (valueFmt || fmt)(r.value);
    const hit = el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", e => showTip(e, r.tip, host));
    hit.addEventListener("mouseleave", hideTip);
  });
}

/* ---------- 折线图（十字线 + 提示框） ---------- */
function lineChart(host, dates, series, { fmt, unit = 100, height = 300, endLabels = true }) {
  host.innerHTML = "";
  const W = host.clientWidth, H = height, padL = 52, padR = endLabels ? 92 : 16, padT = 10, padB = 26;
  if (W <= 0) return;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" }, host);
  const all = series.flatMap(s => s.values), lo = Math.min(0, ...all), hi = Math.max(0, ...all);
  const ticks = niceTicks(lo, hi, 5), ylo = Math.min(lo, ticks[0]), yhi = Math.max(hi, ticks[ticks.length - 1]);
  const sx = i => padL + i / Math.max(1, dates.length - 1) * (W - padL - padR);
  const sy = v => padT + (yhi - v) / ((yhi - ylo) || 1) * (H - padT - padB);
  ticks.forEach(t => {
    el("line", { x1: padL, x2: W - padR, y1: sy(t), y2: sy(t), stroke: t === 0 ? css("--axis") : css("--grid"), "stroke-width": 1 }, svg);
    const tx = el("text", { x: padL - 8, y: sy(t) + 4, "text-anchor": "end", fill: css("--muted"), "font-size": 11 }, svg);
    tx.textContent = tickFmt(t, ticks, unit, fmt); });
  let lastQ = "", first = true, lastX = -1e9;
  dates.forEach((d, i) => { const q = d.slice(0, 4) + "Q" + Math.ceil(+d.slice(4, 6) / 3);
    if (q !== lastQ && (+d.slice(4, 6) - 1) % 3 === 0 && sx(i) - lastX >= 46) {
      const tx = el("text", { x: sx(i), y: H - 6, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, svg);
      tx.textContent = (+d.slice(4, 6) === 1 || first ? d.slice(0, 4) + "-" : "") + d.slice(4, 6);
      first = false; lastX = sx(i) + (tx.textContent.length > 2 ? 18 : 0); }
    lastQ = q; });
  series.forEach(s => {
    const d = s.values.map((v, i) => (i ? "L" : "M") + sx(i).toFixed(1) + "," + sy(v).toFixed(1)).join("");
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
  });
  if (endLabels) {  // 末端标签：按位置排序后错开，避免重叠，用引线连回线端
    const ends = series.map(s => ({ s, y: sy(s.values[s.values.length - 1]) })).sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i++) ends[i].ly = Math.max(ends[i].y, (ends[i - 1].ly ?? ends[i - 1].y) + 15);
    ends[0].ly = ends[0].y;
    ends.forEach(({ s, y, ly }) => {
      const xe = sx(s.values.length - 1);
      el("circle", { cx: xe, cy: y, r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2 }, svg);
      if (Math.abs(ly - y) > 2) el("line", { x1: xe + 5, y1: y, x2: xe + 12, y2: ly, stroke: css("--axis"), "stroke-width": 1 }, svg);
      const t = el("text", { x: xe + 14, y: ly + 4, fill: css("--ink-2"), "font-size": 11 }, svg);
      t.textContent = `${s.name} ${fmt(s.values[s.values.length - 1])}`; });
  }
  const cross = el("line", { y1: padT, y2: H - padB, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, svg);
  const dots = series.map(s => el("circle", { r: 4, fill: s.color, stroke: css("--surface"), "stroke-width": 2, visibility: "hidden" }, svg));
  const hit = el("rect", { x: padL, y: 0, width: Math.max(0, W - padL - padR), height: H, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width);
    const i = Math.max(0, Math.min(dates.length - 1, Math.round((px - padL) / (W - padL - padR) * (dates.length - 1))));
    cross.setAttribute("x1", sx(i)); cross.setAttribute("x2", sx(i)); cross.setAttribute("visibility", "visible");
    series.forEach((s, j) => { dots[j].setAttribute("cx", sx(i)); dots[j].setAttribute("cy", sy(s.values[i])); dots[j].setAttribute("visibility", "visible"); });
    showTip(e, `<b>${fmtDate(dates[i])}</b>` + series.map(s =>
      `<div class="row"><span><span class="key" style="background:${s.color}"></span>${s.name}</span><span>${fmt(s.values[i])}</span></div>`).join(""), host);
  });
  hit.addEventListener("mouseleave", () => { hideTip(); cross.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden")); });
}

/* ---------- 风格暴露：组合 vs 基准（哑铃图） ---------- */
function dumbbell(host, rows) {
  host.innerHTML = "";
  const W = host.clientWidth; if (W <= 0) return;
  const narrow = W < 520, rowH = 30, labelW = narrow ? 72 : 92, padR = narrow ? 52 : 104, H = rows.length * rowH + 30;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": "风格暴露：组合与基准" }, host);
  const vals = rows.flatMap(r => [r.port, r.bench]), lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
  const ticks = niceTicks(lo, hi, 5), xlo = Math.min(lo, ticks[0]), xhi = Math.max(hi, ticks[ticks.length - 1]);
  const x0 = labelW + 16, x1 = W - padR, sx = v => x0 + (v - xlo) / ((xhi - xlo) || 1) * (x1 - x0);
  let lastTick = -1e9;
  ticks.forEach(t => {
    el("line", { x1: sx(t), x2: sx(t), y1: 4, y2: H - 24, stroke: Math.abs(t) < 1e-12 ? css("--axis") : css("--grid"), "stroke-width": 1 }, svg);
    if (sx(t) - lastTick < 34) return; lastTick = sx(t);
    const tx = el("text", { x: sx(t), y: H - 8, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, svg);
    tx.textContent = tickFmt(t, ticks, 1, num); });
  const head = el("text", { x: W - 4, y: 12, "text-anchor": "end", fill: css("--ink-2"), "font-size": 11, "font-weight": 600 }, svg);
  head.textContent = D.has_bench ? (narrow ? "主动" : "主动暴露") : "暴露";
  rows.forEach((r, i) => {
    const cy = 10 + i * rowH + rowH / 2;
    const lab = el("text", { x: labelW, y: cy + 4, "text-anchor": "end", fill: css("--ink-2"), "font-size": 12 }, svg);
    lab.textContent = r.name;
    if (D.has_bench) {
      el("line", { x1: sx(r.bench), x2: sx(r.port), y1: cy, y2: cy, stroke: css("--axis"), "stroke-width": 2, "stroke-linecap": "round" }, svg);
      el("circle", { cx: sx(r.bench), cy, r: 5, fill: css("--s2"), stroke: css("--surface"), "stroke-width": 2 }, svg);
    }
    el("circle", { cx: sx(r.port), cy, r: 5, fill: css("--s1"), stroke: css("--surface"), "stroke-width": 2 }, svg);
    const v = D.has_bench ? r.port - r.bench : r.port;
    const vt = el("text", { x: W - 4, y: cy + 4, "text-anchor": "end", fill: css("--ink"), "font-size": 12, "font-weight": 600 }, svg);
    vt.textContent = num(v, 2);
    const hit = el("rect", { x: 0, y: cy - rowH / 2, width: W, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", e => showTip(e, `<b>${r.name}</b> (${r.key})` +
      `<div class="row"><span><span class="dot" style="background:${css("--s1")}"></span>组合</span><span>${num(r.port)}</span></div>` +
      (D.has_bench ? `<div class="row"><span><span class="dot" style="background:${css("--s2")}"></span>基准</span><span>${num(r.bench)}</span></div>` +
      `<div class="row"><span>主动（平均）</span><span>${num(r.active)}</span></div><div class="row"><span>主动（期末）</span><span>${num(r.active_last)}</span></div>` : ""), host));
    hit.addEventListener("mouseleave", hideTip);
  });
}

/* ---------- 月度主动暴露热力图 ---------- */
const hex2rgb = h => { h = h.replace("#", ""); return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16)); };
const mix = (a, b, t) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
function divColor(v, m) {
  const t = Math.min(1, Math.abs(v) / (m || 1));
  return mix(hex2rgb(css("--mid")), hex2rgb(v >= 0 ? css("--pos") : css("--neg")), Math.pow(t, 0.8));
}
function heatmap(host, rows, cols, M) {
  host.innerHTML = "";
  const W = host.clientWidth; if (W <= 0) return;
  const labelW = 92, cellH = 22, gapPx = 2, top = 4, H = top + rows.length * cellH + 22;
  const cw = (W - labelW - 8) / cols.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img", "aria-label": "月度主动暴露热力图" }, host);
  const m = Math.max(...M.flat().map(Math.abs)) || 1;
  rows.forEach((r, i) => {
    const y = top + i * cellH;
    const t = el("text", { x: labelW, y: y + cellH / 2 + 4, "text-anchor": "end", fill: css("--ink-2"), "font-size": 12 }, svg);
    t.textContent = r.name;
    cols.forEach((c, j) => {
      const v = M[i][j], x = labelW + 8 + j * cw;
      const rect = el("rect", { x: x + gapPx / 2, y: y + gapPx / 2, width: Math.max(1, cw - gapPx), height: cellH - gapPx, rx: 3,
        fill: v == null || isNaN(v) ? "transparent" : divColor(v, m) }, svg);
      rect.addEventListener("mousemove", e => showTip(e, `<b>${r.name}</b> · ${c}<div class="row"><span>月均主动暴露</span><span>${num(v)}</span></div>`, host));
      rect.addEventListener("mouseleave", hideTip);
    });
  });
  let lastX = -1e9;
  cols.forEach((c, j) => { const x = labelW + 8 + j * cw + cw / 2;
    if ((c.endsWith("-01") || j === 0 || +c.slice(5) % 3 === 1) && x - lastX >= 40) {
      const t = el("text", { x, y: H - 6, "text-anchor": "middle", fill: css("--muted"), "font-size": 11 }, svg);
      t.textContent = c.endsWith("-01") || j === 0 ? c : c.slice(5); lastX = x; } });
  document.getElementById("legend-heat").innerHTML =
 `<span style="white-space:nowrap">${num(-m, 2)}<span class="ramp" style="background:linear-gradient(90deg, ${css("--neg")}, ${css("--mid")}, ${css("--pos")})"></span>${num(m, 2)}</span>`;
}

/* ---------- 行业：两列共用行标签 ---------- */
function industryChart(host, rows) {
  host.innerHTML = "";
  const W = host.clientWidth, rowH = 24, labelW = 76, gap = 28, H = rows.length * rowH + 44;
  const colW = (W - labelW - gap) / 2;
  if (W <= 0) return;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, height: H, role: "img" }, host);
  const cols = [
    { key: "contrib", head: "累计贡献", x: labelW + 8, fmt: v => pct(v, 2) },
    { key: "active", head: D.has_bench ? "平均主动权重" : "平均权重", x: labelW + 8 + colW + gap, fmt: v => pct(v, 1) },
  ];
  cols.forEach(c => {
    const vals = rows.map(r => r[c.key]), m = Math.max(...vals.map(Math.abs)) || 1;
    const half = (colW - 120) / 2, zx = c.x + 60 + half, sx = v => zx + v / m * half;
    const h = el("text", { x: zx, y: 14, "text-anchor": "middle", fill: css("--ink-2"), "font-size": 12, "font-weight": 600 }, svg);
    h.textContent = c.head;
    el("line", { x1: zx, x2: zx, y1: 22, y2: H - 4, stroke: css("--axis"), "stroke-width": 1 }, svg);
    rows.forEach((r, i) => {
      const y = 26 + i * rowH, bh = 14, by = y + (rowH - bh) / 2, v = r[c.key];
      el("path", { d: barPath(zx, sx(v), by, bh), fill: v >= 0 ? css("--pos") : css("--neg") }, svg);
      const t = el("text", { x: v >= 0 ? sx(v) + 5 : sx(v) - 5, y: by + bh / 2 + 4, "text-anchor": v >= 0 ? "start" : "end",
        fill: css("--ink"), "font-size": 10.5 }, svg);
      t.textContent = c.fmt(v);
    });
  });
  rows.forEach((r, i) => {
    const y = 26 + i * rowH;
    const t = el("text", { x: labelW, y: y + rowH / 2 + 4, "text-anchor": "end", fill: css("--ink-2"), "font-size": 12 }, svg);
    t.textContent = r.name;
    const hit = el("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", e => showTip(e, `<b>${r.name}</b>` +
      `<div class="row"><span>组合权重</span><span>${(r.port * 100).toFixed(2)}%</span></div>` +
      (D.has_bench ? `<div class="row"><span>基准权重</span><span>${(r.bench * 100).toFixed(2)}%</span></div>` : "") +
      `<div class="row"><span>行业因子累计收益</span><span>${pct(r.unit, 2)}</span></div>` +
      `<div class="row"><span>累计贡献</span><span>${pct(r.contrib, 3)}</span></div>`, host));
    hit.addEventListener("mouseleave", hideTip);
  });
}

function table(host, head, rows) {
  host.innerHTML = `<table><thead><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr></thead><tbody>${
    rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function render() {
  const g = D.groups, last = k => g[k][g[k].length - 1];
  document.getElementById("title").textContent = `${D.portfolio} 相对 ${D.benchmark} · Barra 收益归因`;
  document.getElementById("subtitle").textContent = `${fmtDate(D.start)} 至 ${fmtDate(D.end)}，${D.days} 个交易日；参考模型计算流程的美股适配；${D.styles.length} 个可用风格 + ${D.industries.length} 个行业 + 市场；非授权商业模型`;
  document.getElementById("model-note").textContent = `${D.return_basis}；整体缺失风格：${D.missing_styles.join(", ") || "无"}；持仓风格零填充证券日数：${D.zero_filled_stock_days}`;
  const tiles = [
    [D.has_bench ? "累计收益差" : "累计收益", pct(last("Total")), `组合 ${pct(D.port_total)} · 基准 ${pct(D.bench_total)}`, true],
    ["行业", pct(last("Industry")), `${D.industries.length} 个行业合计`],
    ["风格", pct(last("Style")), `${D.styles.length} 个风格因子合计`],
    ["成本", pct(last("Costs")), "交易与借券成本单列"],
    ["最大回撤", pct(D.performance.max_drawdown), "策略净收益；含初始净值"],
    ["特异（选股）", pct(last("Specific")), "因子解释不了的部分"],
    ["市场（仓位）", pct(last("Market")), `组合仓位 ${(D.market.port * 100).toFixed(1)}%`],
  ];
  document.getElementById("tiles").innerHTML = tiles.map(([l, v, n, hero]) =>
    `<div class="tile${hero ? " hero" : ""}"><div class="label">${l}</div><div class="value">${v}</div><div class="note">${n}</div></div>`).join("");

  const perf = [
    {name: "策略累计收益", values: D.performance.portfolio, color: css("--s1")},
  ];
  const dd = [{name: "策略回撤", values: D.performance.portfolio_dd, color: css("--s1")}];
  if (D.has_bench) {
    perf.push({name: "基准累计收益", values: D.performance.benchmark, color: css("--s2")});
    perf.push({name: "相对净值 - 1", values: D.performance.relative, color: css("--s3")});
    dd.push({name: "基准回撤", values: D.performance.benchmark_dd, color: css("--s2")});
  }
  document.getElementById("legend-performance").innerHTML = perf.map(s => `<span><span class="key" style="background:${s.color}"></span>${s.name}</span>`).join("");
  lineChart(document.getElementById("chart-performance"), D.dates, perf, {fmt:pct,height:300});
  lineChart(document.getElementById("chart-drawdown"), D.dates, dd, {fmt:pct,height:240});
  const gs = [
    { name: "市场", values: g.Market, color: css("--muted") },
    { name: "成本", values: g.Costs, color: "#9972b8" },
    { name: "行业", values: g.Industry, color: css("--s1") },
    { name: "风格", values: g.Style, color: css("--s2") },
    { name: "特异", values: g.Specific, color: css("--s3") },
    { name: D.has_bench ? "收益差合计" : "收益合计", values: g.Total, color: css("--ink"), width: 2.5 },
  ];
  document.getElementById("legend-groups").innerHTML = gs.map(s => `<span><span class="key" style="background:${s.color}"></span>${s.name}</span>`).join("");
  lineChart(document.getElementById("chart-groups"), D.dates, gs, { fmt: pct, height: 320 });

  // 风格暴露（按主动暴露排序）
  const byExpo = [...D.styles].sort((a, b) => b.active - a.active);
  document.getElementById("legend-dumbbell").innerHTML =
    `<span><span class="dot" style="background:${css("--s1")}"></span>${D.portfolio}（组合，日均）</span>` +
    (D.has_bench ? `<span><span class="dot" style="background:${css("--s2")}"></span>${D.benchmark}（基准，日均）</span><span>右侧数字 = 主动暴露</span>` : "");
  dumbbell(document.getElementById("chart-dumbbell"), byExpo);
  heatmap(document.getElementById("chart-heat"), byExpo, D.months, byExpo.map(s => D.heat[D.styles.indexOf(s)]));
  table(document.getElementById("table-expo"),
    ["因子", "组合暴露", "基准暴露", "主动（平均）", "主动（期末）", "主动波动", "因子累计单位收益", "累计贡献"],
    byExpo.map(s => [`${s.name} <span style="color:var(--muted)">${s.key}</span>`, num(s.port), D.has_bench ? num(s.bench) : "—",
      num(s.active), num(s.active_last), s.active_std.toFixed(3), pct(s.unit, 2), pct(s.contrib, 2)]));

  // 风格收益
  const byContrib = [...D.styles].sort((a, b) => b.contrib - a.contrib);
  divBars(document.getElementById("chart-style-contrib"), byContrib.map(s => ({ label: s.name, value: s.contrib,
    tip: `<b>${s.name}</b><div class="row"><span>累计贡献</span><span>${pct(s.contrib, 3)}</span></div><div class="row"><span>平均主动暴露</span><span>${num(s.active)}</span></div><div class="row"><span>因子累计单位收益</span><span>${pct(s.unit, 2)}</span></div>` })),
    { fmt: pct, valueFmt: v => pct(v, 2), title: "风格因子累计贡献" });
  divBars(document.getElementById("chart-style-unit"), byContrib.map(s => ({ label: s.name, value: s.unit,
    tip: `<b>${s.name}</b><div class="row"><span>因子累计单位收益</span><span>${pct(s.unit, 2)}</span></div>` })),
    { fmt: pct, valueFmt: v => pct(v, 1), title: "风格因子累计单位收益" });

  const pick = document.getElementById("style-pick");
  if (!pick.options.length) {
    pick.innerHTML = byExpo.map(s => `<option value="${s.key}">${s.name} (${s.key})</option>`).join("");
    pick.addEventListener("change", drawPick);
  }
  drawPick();

  const ind = [...D.industries].sort((a, b) => b.contrib - a.contrib);
  industryChart(document.getElementById("chart-industry"), ind);
  table(document.getElementById("table-industry"), ["行业", "组合权重", "基准权重", "主动权重", "行业因子累计收益", "累计贡献"],
    ind.map(r => [r.name, (r.port * 100).toFixed(2) + "%", D.has_bench ? (r.bench * 100).toFixed(2) + "%" : "—",
      pct(r.active, 2), pct(r.unit, 2), pct(r.contrib, 3)]));

  document.getElementById("market-line").innerHTML =
    `本例：组合仓位（市场暴露）日均 <b>${(D.market.port * 100).toFixed(1)}%</b>，基准 <b>${(D.market.bench * 100).toFixed(1)}%</b>，` +
    `市场因子期间累计收益 <b>${pct(D.market.unit, 2)}</b>，对收益的累计贡献（有基准时为超额） <b>${pct(D.market.contrib, 2)}</b>。`;
}
function drawPick() {
  const s = D.styles.find(x => x.key === document.getElementById("style-pick").value);
  const ex = [{ name: "组合", values: s.port_ts, color: css("--s1") }];
  if (D.has_bench) ex.push({ name: "基准", values: s.bench_ts, color: css("--s2") });
  document.getElementById("legend-pick").innerHTML = ex.map(e => `<span><span class="key" style="background:${e.color}"></span>${e.name}</span>`).join("");
  lineChart(document.getElementById("chart-pick-expo"), D.dates, ex, { fmt: (v, d) => num(v, d ?? 3), unit: 1, height: 220, endLabels: false });
  lineChart(document.getElementById("chart-pick-unit"), D.dates, [{ name: s.name + " 单位收益", values: s.unit_ts, color: css("--ink") }], { fmt: pct, height: 220, endLabels: false });
  lineChart(document.getElementById("chart-pick-contrib"), D.dates, [{ name: "贡献", values: s.contrib_ts, color: css("--s3") }], { fmt: pct, height: 220, endLabels: false });
}
render();
let rt, lastW = document.querySelector("main").clientWidth;
new ResizeObserver(() => { const w = document.querySelector("main").clientWidth;
  if (w === lastW) return; lastW = w; clearTimeout(rt); rt = setTimeout(render, 120); }).observe(document.querySelector("main"));
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
</script>
</body>
</html>
"""
