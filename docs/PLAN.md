# 美股因子挖掘与量化策略：项目计划书

> 版本：v0.1（2026-10-25）　状态：**计划阶段，尚未开始实现**
> 目标：① 在 WorldQuant BRAIN（USA 区域）挖掘并提交更多 Alpha；② 构建自己的美股量化策略。
> 方法：沿用已有的暴搜和遗传规划（GP）挖掘算法，以及现有的 XGBoost 超参数和 Barra 归因代码。

---

## 0. 总览

| 期 | 内容 | 关键产出 | 验收标准 |
|---|---|---|---|
| 一期 | 数据：选数据源、下载、清洗，定好存储格式和加载接口，评估磁盘 | `/hy-tmp/data_pyx/us_quant/` 下的干净数据，`usq.data` 加载 API，数据质量报告 | 第 1.6 节的质量检查全部通过；GP 引擎可直接读取宽表 |
| 二期 | 因子评估平台（复刻 BRAIN 的评估方式）+ 暴搜/GP 挖掘 + 101/191 基准因子 | 本地 BRAIN 模拟器、因子库（表达式 + 指标 + BRAIN 表达式）、"多重检验"课题报告 | 模拟器与 BRAIN 真实结果的校准误差达标（第 2.3 节）；得到一批通过本地提交标准的表达式 |
| 三期 | XGBoost 训练（沿用超参数）+ 简单回测 | 模型、2024–2026.06 回测报告 | 无前视（purge）、含交易成本，报告收益、回撤、夏普和换手 |
| 四期 | 美股版 Barra 风险与归因 | 策略风格暴露和收益归因报告 | 先展示暴露，再展示贡献；截距项称为"市场因子" |

### 时间划分（全项目统一）

| 区间 | 用途 |
|---|---|
| 数据起点 ~ 2021-12-31 | 暴搜/GP 进化（IS）；XGB 训练集 |
| 2022-01-01 ~ 2022-12-31 | 因子验证（VAL）；XGB 早停和验证 |
| 2023 全年 | 间隔年，不参与任何选择 |
| 2024-01-01 ~ 2026-06-30 | 回测（OOS），**只在最后冻结方案后使用一次** |

规则：
- 每个区间边界前若干天的标签窗口会跨越边界，这些样本一律丢弃（purge），天数等于 label horizon。
- 数据起点见第 1.2 节。建议 GP 进化从 **2010** 开始，避开 2008 年危机期和较早的数据质量问题；2006–2009 的数据保留，可用于稳健性检查。

---

## 1. 一期：数据

### 1.1 服务器现状（2026-10-25 实测）

| 项目 | 现状 | 影响 |
|---|---|---|
| `/hy-tmp` | 400G 盘，已用 365G（92%），**仅剩 36G** | 共享盘。原始分钟和期权链不能全量落盘，见 1.4 节 |
| `/`（含 `/home`） | 30G，剩 5.8G | 代码和 venv 可以放，数据不能放 |
| `/hy-tmp/data_pyx` | 已用 3.9G（factor_filter 3.6G） | — |
| CPU / 内存 / GPU | 128 核 / 2 TB / 1× RTX 3090 24G | 算力充足，瓶颈在磁盘和数据费用 |
| Python | 3.11 + pandas 3.0 + xgboost 3.2.0；没有 polars/duckdb | 项目 venv 里补 pyarrow、duckdb |
| 外网 | 能访问 Massive(Polygon)、Databento、Tushare；**不能访问 GitHub**；WRDS 未测通 | 代码在 Mac 上推 GitHub 再 rsync 到服务器，或给服务器配代理 |

### 1.2 数据源调研

#### 日频行情（含复权）+ 基本面

| 数据源 | 内容 | 是否幸存者偏差 | 时点（PIT）正确性 | 费用 | 结论 |
|---|---|---|---|---|---|
| **WRDS：CRSP + Compustat**（经 NUS） | CRSP 日频：价格、复权因子 `cfacpr/cfacshr`、含分红的总收益 `ret`、退市收益 `dlret`、稳定的 PERMNO；Compustat 季报带公告日 `rdq` | 无（包含退市股） | 好（用 `rdq` 对齐） | 学校订阅，免费 | **首选**。先确认 NUS 是否开通（bizfdb.nus.edu.sg 页面列出了 CRSP、Compustat）。注意：学术授权通常不允许商用 |
| **Sharadar**（Nasdaq Data Link） | SEP 日频价格、ACTIONS 公司行为、TICKERS 证券主表（含退市股）、SF1 基本面（1990 年起，带 `datekey` 申报日）、S&P 500 历史成分 | 无 | 较好（按 `datekey` 对齐） | 个人订阅，价格需登录查看 | **WRDS 不可用时的首选**，性价比高 |
| Massive（原 Polygon.io）Stocks Advanced | 20 年以上历史、分钟和日频 flat files、财务和比率、公司行为；可按日期查询历史 ticker（含退市） | 需自己用历史 ticker 接口重建证券池 | 财报 PIT 字段需核实 | $199/月（按年付 8 折）；财务单独订阅 $29/月 | 适合拿**分钟数据**；日频基本面不如 Sharadar 规整 |
| Tushare 美股 | `us_daily`（未复权）、`us_adjfactor`（2006 起）、`us_income` 等财报 | 未说明是否含退市股 | **差**：财报没有公告日，只有报告期 | 要单独开通"美股历史日线"和"美股财报"权限 | 只用于交叉核对，不作主数据 |
| yfinance 等免费源 | 只有当前在市股票 | **有** | 无 | 免费 | 不用于研究 |

#### 分钟数据

| 数据源 | 说明 | 结论 |
|---|---|---|
| Massive flat files（stocks minute aggregates） | 最早到 2003-09，每天一个 S3 压缩 CSV，覆盖全市场 | **首选**。可订阅 1–2 个月，批量下载后计算日内特征（下载前核对授权条款） |
| Databento | 按量计费，交易所级数据 | 备选 |
| WRDS TAQ | 逐笔成交和报价，体量很大 | 有权限也只抽样使用 |

#### 期权数据

| 数据源 | 说明 | 结论 |
|---|---|---|
| OptionMetrics IvyDB（WRDS） | 学术界标准：隐含波动率曲面、Greeks、持仓量 | NUS 的 WRDS 页面没有列出，需要问图书馆 |
| ThetaData | 历史 EOD Greeks、持仓量和分钟数据；约 $40–160/月 | 自付费时的首选 |
| Massive Options | 分钟 K 线 flat files（全美期权市场） | 和股票数据同一家时可以考虑 |

**推荐组合**：
- 路线 A（先查，免费）：WRDS 的 CRSP + Compustat (+ OptionMetrics)，分钟数据另订 Massive。
- 路线 B（付费）：Sharadar（日频 + 基本面），再加 Massive Stocks Advanced 和 ThetaData，后两项按需订 1–2 个月。

**一期先只做日频 + 基本面**。分钟和期权排在一期之后（记作 1.5 期），只落地派生的日频特征。

> BRAIN 回测用的是 WorldQuant 自己的数据。我们的本地数据只用来预筛和建自有策略，提交到 BRAIN 的表达式必须写成 BRAIN 的字段名（见 2.5 节）。

### 1.3 数据格式

分三层，只读向下游流动：

```
/hy-tmp/data_pyx/us_quant/
├── raw/<vendor>/<dataset>/<ingest_date>/     # 原样下载，不修改（可压缩）
├── clean/                                    # Parquet(zstd)，长表，按年分区
│   ├── security_master.parquet   # sid, ticker 历史区间, 交易所, 证券类型, 上市/退市日, 行业(GICS/SIC)
│   ├── prices_daily/year=YYYY/   # date, sid, open, high, low, close, volume, vwap, adj_factor, ret_total, dlret, shares_out
│   ├── fundamentals_pit/         # sid, period_end, field, value, available_date (= 申报日后的下一个交易日)
│   ├── universe_daily/           # date, sid, adv20, mcap, in_top3000, in_top1000, in_top500, tradable
│   ├── options_features/         # (1.5 期) date, sid, atm_iv_30d, skew_25d, term_slope, pc_vol_ratio, pc_oi_ratio, iv_rv_spread
│   └── intraday_features/        # (1.5 期) date, sid, 由分钟数据算出的日内特征
├── panels/<universe>/<field>.npy             # 宽表 float32 memmap (T×N) + dates.npy + sids.npy，供 GP/评估平台使用
└── reports/                                  # 数据质量报告
```

要点：
- **主键是稳定的证券 ID（sid）**，用 PERMNO、Sharadar `permaticker` 或 FIGI，不用 ticker（ticker 会被重复使用、会改名）。
- 价格保存**原始价 + 累计复权因子**，读取时再算复权价。收益用数据源给出的含分红总收益，退市当天并入退市收益。
- 基本面只保存**PIT 格式**：每条记录带 `available_date`。宽表展开时按 `available_date` 做前向填充，杜绝前视。
- 宽表兼容现有 GP 引擎的输入（`字段 → 交易日 × 股票 DataFrame`），GP 只改数据加载器即可。
- 加载 API（`research_pyx/Quant_US_stocks/usq/data.py`）：
  `load_panel(fields, start, end, universe="TOP3000") -> dict[str, DataFrame]`，以及 `load_long(table, ...)`。

### 1.4 磁盘预算（估算，下载样本后再校正）

| 数据 | 估算体量 | 处理方式 |
|---|---|---|
| 日频长表 2006–2026.06（约 5,150 个交易日 × 平均约 6,000 只 × 约 12 列） | 原始约 1.5 GB，Parquet 约 0.7 GB | 全量保存 |
| 基本面 PIT | < 1 GB | 全量保存 |
| 宽表（TOP3000 历史上出现过的约 9,000 只 × 5,150 天 × float32 ≈ 185 MB/字段） | 30 个字段约 5.5 GB | 全量保存 |
| 分钟原始数据（全市场每天约 20 MB gz） | 2024–2026.06 约 13 GB；20 年约 80–100 GB | **流式处理**：逐日下载 → 算特征 → 删除原始文件，只留日内特征（< 2 GB）。必要时只保留最近 2 年的 TOP1000 原始数据 |
| 期权全链 EOD（每天约 100 万个合约） | 20 年约 200 GB | **不落地原始链**，只保存每只标的的日频期权特征（< 1 GB） |
| 二期因子矩阵 | 单个因子在 TOP3000 上约 62 MB | **不批量保存因子值**，只存表达式和指标，需要时重算；最终入选的 ≤ 200 个可以落盘（float16 约 6 GB） |

**结论**：一期加二期需要约 **10–15 GB**，36 GB 剩余空间能放下，但这是共享盘，不能挤占别人。如果要做分钟原始数据或大规模因子缓存，请先申请扩盘或挂一块新盘。

### 1.5 一期任务

1. 确认数据路线（第 6 节决策 1）。
2. 写下载器（可续传、增量、raw 层不可变），先下 3 个月数据，用来校正体量估算。
3. 建证券主表：普通股筛选（排除 ETF、ADR、优先股、权证、OTC，ADR 是否保留可配置），处理 ticker 变更，记录退市原因。
4. 构建清洗层，包括复权、总收益、退市收益和 PIT 基本面。
5. 构建股票池：BRAIN 风格的 TOP3000/TOP1000/TOP500（按 20 日 ADV 排序，定期重构；重构频率以 BRAIN 文档为准）。
6. 生成宽表并实现加载 API，写单元测试。
7. 出数据质量报告。

### 1.6 数据质量检查（验收标准）

- 主键 (date, sid) 唯一；交易日历和 NYSE 日历一致。
- OHLC 合理：low ≤ open, close ≤ high；成交量 ≥ 0。
- 复权一致性：拆股日前后复权价连续（不出现 ±50% 的假跳空）；抽查 AAPL（2014、2020 拆股）、TSLA、NVDA、GOOGL 类股。
- 总收益和复权收盘价算出的收益相差 < 1e-6（除息日除外，单独检查）。
- 幸存者偏差：抽查若干已退市公司（如 Lehman、Enron 之后的大型退市股、SVB 2023）在退市前是否存在。
- 和第二数据源交叉核对：随机 200 个 (date, sid)，比较收盘价和收益（如 Tushare `us_daily` × `us_adjfactor`）。
- 基本面 PIT：`available_date ≥ period_end`，且绝大多数在 1–90 天内；抽查若干公司的 10-Q 申报日。
- 缺失率报告：按字段、按年、按股票池分别统计。

---

## 2. 二期：因子评估平台 + 挖掘

### 2.1 思路

1. 先搭一个**本地 BRAIN 模拟器**，按 BRAIN 的方式把表达式变成多空组合，再算 Sharpe、Fitness、换手和相关性。
2. 用 BRAIN 真实结果**校准**模拟器（2.3 节）。
3. 做实验：rankIC 高能否推出 BRAIN 夏普高（2.4 节），据此决定暴搜的适应度用什么。
4. 暴搜/GP 产出候选 → 本地模拟器筛选 → 去重 → 提交 BRAIN。
5. 101 和 191 基准因子作为基线、相关性参照和 GP 种子。

> 一个判断：BRAIN 式评估的单因子开销**和 rankIC 是同一量级**。两者主要成本都是截面 rank，O(T·N)；中性化是按组去均值，PnL 是一次点积。真正额外的开销是"和因子库的相关性"，而这只是 T 维 PnL 向量之间的相关，很便宜。所以很可能可以直接用 Fitness 或 Sharpe 当暴搜适应度。2.4 节会实测两者的单因子耗时。

### 2.2 本地 BRAIN 模拟器规格

**模拟设置**（对齐 BRAIN 的 Settings）：region=USA；universe=TOP3000/1000/500；delay=1；decay（线性衰减天数）；neutralization（none/market/sector/industry/subindustry；行业用 GICS 或 SIC 映射）；truncation=0.08；pasteurization；NaN 处理；booksize=$20M。

**构建组合**：表达式值 → 截取当期股票池 → decay → 中性化（组内去均值）→ 截断单股权重 → 按 Σ|w| 归一化 → 多空美元中性。

**指标**：
- PnL_t = Σ w_{t-1,i} · r_{t,i}；delay=1 时，表达式只用 ≤ t−1 的数据。
- Sharpe = mean(daily PnL) / std(daily PnL) · √252
- Turnover = 日均 Σ|Δw| / Σ|w|
- Returns = 年化 PnL / (booksize/2)
- Drawdown（最大回撤）、Margin = PnL / 成交金额（bps）、多头和空头持仓数、分年度表
- **Fitness = Sharpe · √(|Returns| / max(Turnover, 0.125))**

**提交测试**（社区资料，阈值**需在你的 BRAIN 账号的提交标准页逐条核对**后再写进配置）：

| 测试 | 规则（USA D1） |
|---|---|
| Sharpe | ≥ 1.25 |
| Fitness | ≥ 1.0 |
| Turnover | 1% – 70% |
| 权重集中度 | 单股权重 ≤ 10%，持仓足够分散 |
| Sub-universe | 在更小的股票池（如 TOP1000）上 Sharpe ≥ 0.75·√(子池规模/原池规模)·原 Sharpe |
| Self-correlation | 和自己已提交 Alpha 的**日 PnL** 相关性 < 0.7；否则 Sharpe 须比相关的那个高至少 10% |
| Prod-correlation | 无法复刻（看不到全平台的 Alpha）。用"101 + 191 + 我们自己的因子库"的 PnL 相关性近似 |

**工程实现**：GPU 向量化（CuPy 或 Torch，RTX 3090），一次评估一批表达式。因子库只存 PnL 序列（T 维），相关性检查是一次矩阵乘。

### 2.3 校准（验收标准）

- 挑 30–50 个表达式（101 因子的一部分加简单量价因子，覆盖不同换手和中性化设置），在 BRAIN 上跑一遍，记录官方的 Sharpe、Turnover、Fitness 和 Returns。
- 本地模拟器跑同样的设置，要求：Sharpe 的截面相关 ≥ 0.9；Turnover 的相对误差中位数 ≤ 15%；通过/不通过的判断一致率 ≥ 85%。
- 达不到时，按顺序排查：股票池构建 → 复权和收益 → delay 对齐 → decay 和截断 → 行业分类。

> 时间窗口要注意：BRAIN 的 IS 窗口和我们的 2024–2026 回测期可能重叠。**用 BRAIN 结果给自有策略选因子时，会把回测期信息泄露进来**。所以自有策略的选择只用本地 ≤2022 的指标，BRAIN 结果只决定"要不要提交"。

### 2.4 实验：rankIC 和 BRAIN 指标的关系

- 样本：GP 或随机生成 5,000–20,000 个表达式，每个在 IS 区间同时算 RankIC/ICIR 和模拟器的 Sharpe/Fitness/Turnover。
- 看三件事：Spearman(ICIR, Sharpe)；按 ICIR 分十组后各组 Sharpe 和通过率的分布；换手的调节作用（同样的 IC，高换手的 Fitness 会被压低）。
- 计时：rankIC 和模拟器的单因子耗时对比。
- 决策规则：
  - 若模拟器耗时 ≤ 2×rankIC，暴搜直接用 Fitness（或 Sharpe 加换手惩罚）做适应度。
  - 否则先用 rankIC 做第一层粗筛，再用模拟器做第二层精筛。

### 2.5 挖掘

- **算法**：沿用 `research_pyx/GP`（层级 PCA-QD V4）和暴搜（`Factor_2/run_standalone.py`），只换数据加载器和算子集。
- **BRAIN 可提交性约束**：算子集限定在 BRAIN Fast Expression 里有的算子（`ts_mean, ts_std_dev, ts_rank, ts_delta, ts_corr, ts_covariance, ts_decay_linear, rank, zscore, group_neutralize, signed_power, ...`）；字段名映射到 BRAIN 字段（`close, open, vwap, volume, returns, cap, adv20` 及基本面字段）。写一个**表达式 → BRAIN 字符串**的翻译器，并做往返单元测试。
- **基准因子**：
  - Alpha101（本来就是在美股和 WQ 体系下写的，翻译成本低）。
  - 国泰君安 191（为 A 股设计，在美股上验证是否有效）。
  - 两者都进入因子库，作为 prod-corr 的近似参照和 GP 初始种群。
- **数据维度扩展顺序**：量价 → 基本面（PIT）→（1.5 期之后）日内特征、期权特征。
- **因子库**（Parquet 或 SQLite）：表达式、BRAIN 表达式、各区间指标、PnL 序列路径、相关簇、状态（候选 / 本地通过 / 已提交 / BRAIN 结果）。

### 2.6 课题：多加一年验证到底有没有意义

**问题**：如果搜出来的因子 IS rankIC 只是均值为 0 的正态噪声，"能被选出来"可能只是因为尝试次数够多；2022 验证也可能只是从幸运因子里再挑出 2022 碰巧也好的。

**实验设计**（全部只用 ≤2022 的数据设计，2024–2026 只在最后用一次）：

| 编号 | 实验 | 回答的问题 |
|---|---|---|
| E1 | **安慰剂搜索**：同一套暴搜/GP 配置在被打乱的标签上重跑。打乱方式为每天截面内随机置换股票的未来收益（破坏预测性，保留分布），或标签整体做随机时间平移 | 纯运气下，IS 最优 ICIR 和"通过筛选"的因子数是多少；真实搜索比它多出多少 |
| E2 | **IS → VAL 持续性**：所有候选因子的 Spearman(IS ICIR, VAL ICIR)，画分组图，并和 E1 中安慰剂的持续性比较 | 2022 验证是否有信息量 |
| E3 | **有效试验次数**：对候选因子的 PnL 或因子值做聚类或 PCA，估计独立试验数 N_eff；计算 Deflated Sharpe Ratio（Bailey & López de Prado）、Harvey-Liu-Zhu 的 t > 3 门槛和 BH-FDR | 扣除多重检验之后，还剩多少显著因子 |
| E4 | **选择规则对比**（冻结后在 2024–2026.06 上只评估一次）：(a) 只按 IS 排名取 Top-K；(b) IS 加 2022 同方向且过阈值（现行规则）；(c) IS 加多重检验校正（DSR 或 FDR）；(d) (b)+(c) | 加一年验证或做统计校正，能不能改善样本外的单因子表现和 XGB 表现 |
| E5 | **2022 的特殊性**：用 Barra 看通过 2022 验证的因子在风格上的暴露（2022 年加息、价值跑赢、成长回撤） | 验证集是否只是挑出了"押对 2022 风格"的因子 |

**产出**：一份报告，结论写成"在 X 次试验、N_eff≈Y 时，VAL 筛选让 OOS 胜率或 IC 提升 Z（置信区间）"或"没有显著提升"。

---

## 3. 三期：XGBoost 训练 + 简单回测

- **特征**：二期入选的因子（建议 ≤ 200–500 个，受内存和磁盘限制），以及 101/191 中入选的部分。每天截面做 rank 或 zscore。
- **标签**：delay=1 的未来 N 日收益（N 为 5 或 10，和 A 股项目的 `label_10` 对齐时取 10），每天截面 rank 化。各区间边界按 N 天做 purge。
- **模型**：沿用 `factor_filter/ff/xgb_tree.py` 的超参数，有两套（第 6 节决策 3）：
  - `tree`：max_depth=9, lr=0.005, n_estimators=6053 + 早停 50, min_child_weight=297, subsample=0.8, colsample=0.9, reg_lambda=5e-3。
  - `live`：max_depth=7, lr=0.03, n_estimators=500, gamma=0.1, reg_lambda=1.0, 不早停。
  - 训练 ≤2021，验证（早停）2022，2023 不用，测试 2024–2026.06。网格搜索留到有时间再做。
- **回测**（复用 `backtest_optimization` 的思路，换成美股参数）：
  - 组合：TOP1000 或 TOP3000 里多头 Top-N（50/100），以及十分位多空。
  - 执行：T 日信号，T+1 开盘价或 VWAP 成交。
  - 成本：单边 5–10 bps（佣金加点差冲击，做敏感性分析）；空头另加融券费，按 ADV 或市值分档。
  - 基准：股票池市值加权，以及 SPY/RSP。
  - 指标：年化收益、波动、Sharpe、最大回撤、换手、分年度表、IC 衰减。

---

## 4. 四期：Barra 风险与归因

- 把 `research_pyx/barra_analysis`（CNE5）改成美股版，参考 USE4：
  - 风格：Size, Beta, Momentum, ResidualVol, NonlinearSize, BookToPrice, Liquidity, EarningsYield, Growth, Leverage。
  - 行业：GICS 板块或行业（若只有 SIC，就映射到约 30–60 个行业）。
  - 市场因子（截距项）：横截面 WLS，权重 √市值，和现有实现一致。
- 基准：TOP3000 市值加权，或 S&P 500（Sharadar 有历史成分；CRSP 可近似）。
- 报告顺序：**先**讲组合、基准和主动部分的风格暴露及其随时间的变化，**再**讲各因子的收益贡献，并单独列出因子单位收益。截距项叫"市场因子"。
- 同时服务 E5 实验（因子和验证集的风格暴露）。

---

## 5. 工程约定

- 代码：`/home/research_pyx/Quant_US_stocks/`（本仓库）；数据：`/hy-tmp/data_pyx/us_quant/`。
- 服务器不能访问 GitHub，所以在 Mac 上提交和推送，再 rsync 或 `git pull` 到服务器（经 Mac 中转或配代理）。
- 仓库不提交数据、API Key、Token；Token 用环境变量（`TUSHARE_TOKEN`、`MASSIVE_API_KEY`、`NASDAQ_DATA_LINK_API_KEY`、`WRDS_USERNAME`）。
- 包结构草案：`usq/data`（下载、清洗、加载）、`usq/universe`、`usq/sim`（BRAIN 模拟器）、`usq/mining`（对接 GP/暴搜）、`usq/alphas`（101/191）、`usq/model`（XGB）、`usq/backtest`、`usq/barra`。每个模块配 `tests/`，用小样本自造数据测试。
- 每期结束：`docs/` 下写一份阶段报告，在 `CHANGELOG.md` 记一条。

---

## 6. 需要你确认的决策

1. **数据路线**：先查 NUS 是否能用 WRDS（CRSP/Compustat，最好还有 OptionMetrics）→ 能用走路线 A；不能用走路线 B（Sharadar）。月预算上限是多少？
2. **分钟和期权**：确认放到 1.5 期（一期只做日频和基本面），并且只保留派生特征、不存原始数据？
3. **XGB 超参数**：用 `tree`（6053 树 + 早停）还是 `live`（500 树）？
4. **IS 起点**：2010（建议）还是数据最早年份（2006/1998）？
5. **磁盘**：一期和二期约 10–15 GB 可以接受吗？是否需要申请扩盘（分钟数据或大规模因子缓存需要）？
6. **股票池**：挖掘用 TOP3000（BRAIN 默认），自有策略用 TOP1000，可以吗？

---

## 参考

- Massive（原 Polygon.io）价格页：https://massive.com/pricing ；分钟 flat files：https://massive.com/docs/flat-files/stocks/minute-aggregates.md ；历史证券主表：https://massive.com/knowledge-base/article/does-massive-have-a-downloadable-security-master
- Tushare 美股：us_daily https://tushare.pro/document/2?doc_id=254 ；us_adjfactor https://tushare.pro/document/2?doc_id=402 ；us_income https://tushare.pro/document/2?doc_id=394
- Sharadar（经 QuantRocket 的说明）：https://www.quantrocket.com/docs/data/fundamental/sharadar/ ；Nasdaq Data Link 定价说明：https://help.data.nasdaq.com/article/568-how-much-does-nasdaq-data-link-data-cost-how-do-i-find-pricing
- NUS 金融数据库 WRDS 页面：https://bizfdb.nus.edu.sg/wrds
- ThetaData 订阅说明：https://docs.thetadata.us/Articles/Getting-Started/Subscriptions.html
- BRAIN 提交标准（社区整理，需在平台核对）：https://www.quantt.co.uk/resources/worldquant-brain ；https://github.com/RussellDash332/WQ-Brain
- Kakushadze (2016), *101 Formulaic Alphas*；国泰君安 (2017)《基于短周期价量特征的多因子选股体系》（191 因子）
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*；Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns*
