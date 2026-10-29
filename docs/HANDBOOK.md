# 美股量化系统搭建说明书

版本 1.2　2026 年 10 月 29 日

面向数据组、因子挖掘负责人及验证课题研究组。

本项目建设日频美股量化研究流程：数据整理、十日收益标签、因子挖掘与评估、XGBoost 预测、简单组合回测，以及风格暴露和收益归因。组员应先统一数据与时间口径，再按依赖顺序运行模块。

统一时间划分：2015–2021 年用于因子挖掘，2022 年用于因子验证，2023–2025 年用于预先冻结方案后的滚动回测与可视化。数据覆盖至少包含 2015–2025 年；建议额外获取 2014 年作为长窗口预热，以及 2026 年初至少 11 个交易日作为标签尾部数据。

系统主要交付物包括字段宽表、候选因子清单、三层评估结果、冻结因子清单、六个半年模型、预测 score、回测报告和风格归因报告。数据采购和清洗由数据组负责，真实研究结论须等待真实数据与完整筛选流程运行后产生。

## 阅读路线

- 第 0 部分说明服务器及环境验收。
- 第 1 部分定义数据来源、存储、质量验收、VWAP 与 ylabel。
- 第 2 部分定义简单、复杂、因子间三层评估。
- 第 3 至 5 部分说明模型、简单回测、Barra 研究代理模型及可视化。
- 第 6 至 7 部分说明爆搜、101/191 接入和全项目时间边界。
- 后续部分给出验证课题、五人分工、运行命令及验收清单。

<!-- PAGEBREAK -->

## 0 服务器与运行环境

### 资源要求

| 项目 | 要求与验收方式 |
|---|---|
| 内存 | 可用配额约 100GB，建议选择 128GB 档；核对容器实际限制，不能只看宿主机总内存 |
| GPU | RTX 3090 24GB 起步，或显存与计算能力更高的 CUDA GPU；用 nvidia-smi 核对型号和可用显存 |
| CPU | 至少 50 个可用 CPU 核配额；区分物理核心、逻辑核心和共享 vCPU，确认长期可用配额 |
| 系统 | 建议 Linux，Python 3.11，独立虚拟环境；XGBoost 固定 3.1.3 |
| 磁盘 | 建议先预留 200GB 可用 SSD 空间，再根据全市场样本测算；数据、候选矩阵和临时训练矩阵分目录保存 |

100GB 内存并不意味着任意规模的全量因子都能同时载入。例如 2,800 个交易日、8,000 个历史证券、100 个 float32 因子的原始特征矩阵约 9GB，另有 pandas 索引、标签、掩码、选取副本和 XGBoost 工作内存。500 个因子仅原始矩阵就约 45GB，应先测峰值再扩大规模。

代码按字段读取、训练特征使用临时 memmap，XGBoost 使用 QuantileDMatrix。后者仍需要内存和显存，本版没有承诺外存训练或多 GPU 分布式能力。先以约 50 个因子、部分股票测试；扩大规模时保持时间与筛选规则不变。

### 环境验收

```bash
nproc
free -h
df -h
nvidia-smi
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
```

生产配置使用 `tree_method: hist`、`device: cuda`。应保存 Python、依赖、驱动和 GPU 信息，并核对训练审计中的 `actual_device`。若 XGBoost 回退到 CPU，正式 CUDA 运行会中止。CUDA 兼容性按固定版本核对，不套用最新版要求。[XGBoost 3.1 GPU 文档](https://xgboost.readthedocs.io/en/release_3.1.0/gpu/index.html)

<!-- PAGEBREAK -->

## 1 数据来源与采购调查

数据组两人共同决定数据路线。先做少量样本下载，核实历史长度、退市证券、复权、财报可用时间、行业历史和实际授权范围，再批量获取。没有核实的权限或字段不能视为已经具备。

| 来源 | 可调查的内容 | 适用方式与待核实项 |
|---|---|---|
| WRDS 下的 CRSP 与 Compustat | 历史证券、日频行情和公司行为，结合财务数据 | 优先检查学校是否有对应订阅；稳定 ID、退市和财报发布日期须分别核对 |
| Nasdaq Data Link 的 Sharadar SEP 与 SF1 | 日频股价、证券信息和财务字段 | 商业备选；确认 PIT 维度、datekey、退市覆盖和批量下载范围 |
| Massive | 美股聚合行情、VWAP、历史证券及公司行为接口 | 用于行情或第二来源核对；确认订阅可访问的历史区间、调整方式和 VWAP 成交集合 |
| SEC EDGAR | 提交历史和 XBRL 财务事实 | 可构建基本面来源；自行保留申报时间、修订版本、单位和证券映射 |
| Alpha Vantage | 日频、复权日频、部分指数及财务接口 | 适合抽样或辅助核对；全量研究前核实历史深度、限额和退市覆盖 |

CRSP 官方资料说明其覆盖在市和非在市证券，并提供永久标识及公司行为。它仍需要正确的证券过滤、收益字段和退市处理。[CRSP 数据说明](https://www.crsp.org/research/crsp-us-stock-databases/?activetab=docs)

Nasdaq 的 Sharadar 数据可作为行情加基本面的调查路线；具体字段、账户权限及当期价格由数据组核实。[Nasdaq Data Link](https://www.nasdaq.com/products/data/data-link)

SEC 的公开 API 提供公司提交历史和 XBRL JSON，也有批量文件。财报期末日不等于市场可用日，当前聚合事实也不能直接当作历史 vintage。[SEC API 文档](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)

行情供应商对 VWAP、复权和交易时段的定义可能不同，不能未经对齐就把差异当成错误。Massive 对其 VWAP 成交集合有专门说明。[Massive VWAP 定义](https://massive.com/knowledge-base/article/how-is-a-stock-bars-vwap-calculated-and-why-can-it-fall-outside-the-high-and-low)

数据源比较表应交付：样本字段、可用年份、历史退市覆盖、PIT 能力、缺失比例、费用与许可结论，以及最终主来源和核对来源。

<!-- PAGEBREAK -->

## 1 数据范围与字段清单

### 日频量价与证券资料

最低字段包括原始 open、high、low、close、volume，原始 VWAP 或可计算 VWAP 的成交金额 amount，以及统一定义的 adj_factor。还需 market_cap、历史 industry、交易日历、当日 universe 掩码、上市退市记录、ticker 变更与稳定证券 ID 映射。

成交量统一为股，成交金额和市值统一为美元；价格为美元每股。若源字段使用千股、千美元、手或其他单位，必须在转换阶段处理，不能把倍数留给下游猜测。volume 可使用浮点数，避免不必要的截断。

### 基本面

至少调查 shares_out、book_equity、earnings_ttm、revenue_ttm、total_assets、total_liabilities。若缺失某类基本面，先保留明确缺失和来源标记，不用未来公告值填补历史。上述字段同时服务因子构建、市值行业中性化及风格代理模型。

行业须保存历史可用分类。优先采用统一的 GICS 或其他可解释分类；不同来源的行业体系先建映射。不能把今天的行业分类无条件覆盖 2015 年，也不能把 PERMCO 等公司 ID 与证券级 ID 混用。

### 重要指数

建议收集 S&P 500、Nasdaq Composite、Nasdaq 100、Dow Jones Industrial Average、Russell 2000；可补充 VIX 作为市场状态变量。指数表单独保存为日期乘指数代码的宽表，例如 `indices_close.pkl`，不要混入股票证券列。

若使用 SPY、QQQ、DIA、IWM 作为可交易代理，必须标明 ETF 身份。价格指数、总收益指数、ETF 调整价和含分红收益属于不同口径；比较策略时明确基准是哪一个。本版默认基准为当时股票池市值加权组合，指数数据可用于辅助展示或扩展基准。

### 股票池建议

从 NYSE、Nasdaq 等常见交易所普通股开始，明确 ADR、REIT、ETF、OTC、多重股份类别和微盘股的处理规则。流动性、上市天数和市值条件只用当时已知信息。为减少长窗口暴露缺失，可先约定上市满 252 个交易日，并记录 ADV 等过滤规则。

历史证券列应取研究期间证券的并集；证券当时未上市或不在股票池时，用 universe=0 和对应缺失表示。不能用 2025 年仍存续的股票清单回溯整个历史。

<!-- PAGEBREAK -->

## 1 数据存储与加载接口

每个字段保存为一个 pickle 文件，内部为 pandas DataFrame，行是日期字符串 `YYYYMMDD`，列是稳定证券 ID。所有股票字段使用完全相同的交易日历、列集合和列顺序。

```text
data/
  raw/<vendor>/<dataset>/<ingest_date>/
  panels/
    calendar.csv
    open.pkl  high.pkl  low.pkl  close.pkl
    volume.pkl  amount.pkl  vwap.pkl
    adj_factor.pkl  market_cap.pkl
    industry.pkl  universe.pkl
    book_equity.pkl  earnings_ttm.pkl
    revenue_ttm.pkl  total_assets.pkl
    total_liabilities.pkl  shares_out.pkl
    indices_close.pkl
  metadata/
    security_master.parquet
    fundamental_vintages.parquet
    source_mapping.json
    data_dictionary.csv
```

原始数据保留下载版本；基本面 vintage 和证券主表可用 Parquet 长表保存。下游股票宽表的数值字段通常写成 float32，industry 保存分类字符串，universe 使用明确的 0/1，不保留未知状态。

`calendar.csv` 至少包含 date 列，来自独立核实的美股交易日历，并与股票面板索引逐项一致。它不能由某一只股票的有效报价日反推。代码会检查完整面板轴的一致性，不会悄悄取交集后丢股票或丢日期。

```python
from usq.data import PanelStore
store = PanelStore("data/panels")
close = store.load("close")
cap = store.load("market_cap", reference=close)
industry = store.load("industry", reference=close, numeric=False)
```

字段名称、来源及单位写入数据字典，例如 market_cap 统一表示美元计价的总市值。不同来源使用不同字段名或单位时，在转换阶段统一后再保存。

pickle 只在团队可信产物之间流转。每个数据快照保存文件哈希、形状、覆盖区间、来源、复权和财报时点说明；不要把凭据写入文件或仓库。指数表列集合不同，应单独读取，不能使用股票面板的列对齐检查。

<!-- PAGEBREAK -->

## 1 时点数据与多来源验收

### PIT 基本面与公司行为

财务长表至少保存 sid、period_end、available_date、field、value、来源与 vintage。available_date 是该版本可以进入信号的首个日期，依据公告时间、时区及信号截止时点确定。仅有报告期而没有可用时间的数据，不能直接用于历史选股。

将基本面展开到宽表时，按证券和可用日期向前延续。禁止反向填充，也禁止用后续重述替换当时所知的旧版本。`pit_panel` 可展开单字段，但 available_date 及版本冲突需要数据组事先整理。

复权因子统一满足 adjusted_price = raw_price × adj_factor。拆股调整与包含现金分红的总收益调整应分别说明。如果只提供拆股复权，本版回测就是拆股调整价格收益，不能命名为含分红总收益。退市终值、并购换股和分红事件要保留原始依据。

### 多来源比较

先对齐 sid、交易日期、正常交易时段、币种、单位和复权方式，再比较 close、volume、VWAP、market_cap 及抽样财务字段。价格容差可从绝对 0.01 美元加相对 0.1% 开始；它是排查阈值，不是所有字段通用的通过标准。

```bash
usq compare --primary work/vendor_a_close.pkl \
  --secondary work/vendor_b_close.pkl --output runs/source_compare
```

工具输出共同观测数、差异数、差异率、第二来源缺失数及差异明细。成交量和财务字段应按自身单位另设容差。VWAP 偶尔超出某供应商高低价范围不应自动判错，应先核对其成交纳入规则。

### 数据验收门槛

必须检查：日期和证券唯一性；2015–2025 覆盖；交易日历一致；当时在池股票的价格、复权、市值及行业有效；每年股票数与覆盖率；IPO 与退市样本；拆股、大额分红、停牌及 ticker 重用样本；基本面公告前不可见。

运行 `usq audit-data` 检查机械性条件，报告位于 `runs/data_audit.json`。来源是否真正覆盖退市、PIT vintage 是否正确、复权是否包含分红，仍需要人工样本核验。出现持仓收益缺失时先补清数据，不能通过删除未来出问题的股票来改善回测。

<!-- PAGEBREAK -->

## 1 VWAP 与 ylabel 的计算

### VWAP

优先使用供应商定义清晰的日 VWAP；若有同口径成交金额和成交量，可计算 VWAP = amount / volume。volume 必须大于零，成交金额和股数单位必须匹配。只有 OHLCV 时，不能把 HLC 均价或收盘价伪装成真实 VWAP。

VWAP_adj = VWAP × adj_factor。open、high、low、close 也使用同一复权因子生成调整价，避免爆搜表达式混用未复权开盘价和复权收盘价。

### 默认十日标签

信号在 T 日收盘后形成，预定 T+1 日 VWAP 入场。默认原始标签为：

```text
y_raw[T,i] = VWAP_adj[T+11,i] / VWAP_adj[T+1,i] - 1
label_end[T] = T+11 的实际交易日期
```

分子分母相隔十个交易日；这里的 T+1、T+11 按真实交易日历移动。整个标签窗口内的价格必须可用，否则标签缺失。若数据只到 2025 年末，最后若干信号无法形成完整十日标签，不能补零。

### 市值与行业中性化

每个信号日做横截面 WLS，以 y_raw 为因变量，以信号日 log(market_cap) 和行业哑变量为解释变量。完整行业哑变量已经张成截距空间，无需再加入冗余常数列。

```text
y_raw[T,i] = a[T] + b[T] × log(cap[T,i])
             + industry_effect[T,i] + residual[T,i]
WLS objective weight[i] = sqrt(cap[T,i])
ylabel[T,i] = 截面标准化后的 residual[T,i]
```

数据缺失、行业未知、市值非正的股票不进入当日回归。ylabel 是监督学习目标，不是实际可交易收益；回测和收益归因必须回到原始调整价格区间。

默认使用上述 VWAP 区间收益；如需未来十日收盘收益之和，可设置 `label.mode: close_sum`。两种模式都记录 label_end 和元数据，不混合训练结果。

运行 `usq prepare` 生成调整价、`forward_10d_raw.pkl`、`label_10.pkl`、`label_end.csv`、`label_metadata.json` 和文件 manifest。任何模式、权重或标准化变动都应重新生成标签并创建新的实验目录。

<!-- PAGEBREAK -->

## 2 第一层评估 IC RankIC ICIR 与 CV

第一层用于低成本初筛。每天在当时股票池内，将因子值与对应十日 ylabel 配对，删除双方缺失的观测，并要求至少达到预设股票数，默认 30 只。常数因子的相关系数不可计算，不能当作零 IC。

| 指标 | 本项目定义 |
|---|---|
| IC | 当日因子值与 ylabel 的截面 Pearson 相关系数 |
| RankIC | 当日双方截面排名的 Pearson 相关系数，等价于处理并列值后的 Spearman 相关 |
| ICIR | 日 IC 均值 / 日 IC 标准差；RankICIR 同理 |
| CV | 当日日 RankIC 为正时，除以前两个日历年日 RankIC 的最小值；为负时除以最大值。历史不含当天；不足完整两年、有效观测少于 252、当天为零或分母为零时记为缺失 |
| 覆盖率 | 有效配对数 / 当时在池且标签可用的观测数 |

未来十日标签相互重叠，日 IC 不是独立观测。评估时同时检查各时间段的表现，并保留候选全集以研究多重检验影响。

因子方向只在 2015–2021 确定并冻结。若训练期原始平均 RankIC 为负，后续使用相反方向；不能见到 2022 或 2023–2025 变负后再翻转。额外输出四个连续时间块的稳定性诊断，不用于四折训练或 bagging。

<!-- PAGEBREAK -->

## 2 第二层评估 单因子与分层

第二层针对第一层保留的候选，成本更高。使用冻结方向构造单因子组合，以可执行价格收益评价交易表现，并观察因子是否只是行业或风格暴露。

### 分层单调性

按每天信号从低到高分组，默认复杂评估分 10 组。相同分数保留在同一组，允许空组，避免人为打散大量并列分数制造单调图。报告每组平均日净收益、累计曲线、有效天数及组号与平均收益的 Spearman 相关。

同时输出相邻组收益递增比例。单调性不要求每个短窗口每组都严格递增，应结合置信区间、市场阶段、样本数、成本和换手判断。阈值必须在查看 2023–2025 结果之前冻结。

### 单因子组合

本版多空权重先把当日因子截面排名去均值，再按绝对权重和归一化至 1。组合净敞口约为零、多空各约半本；它与多头 Top-N 组合的收益量级不同。

T 日信号对应 T+1 到 T+2 VWAP 区间。输出成本前后累计收益、年化收益、波动、Sharpe、最大回撤、换手和借券费估计。初次建仓及之后调仓均计成交成本，调仓量比较的是目标权重与上一期收益漂移后的权重。

### Fitness 与 Barra

本地 Fitness 定义为 Sharpe × sqrt(abs(年化收益) / max(日均单边换手, 0.125))，输出名称为 fitness_local。Sharpe、收益和换手依本项目模拟口径计算；它是受 WorldQuant 风格启发的综合指标，没有与 BRAIN 私有平台逐项校准，不能当作官方提交分数。

复杂评估可加 `--barra`，为候选输出风格代理模型暴露和收益贡献。缺失风格数据会明确报出或标明未计算；不把空结果写成零暴露。实际选择可结合 RankIC、分层、成本后 Sharpe、回撤及暴露，避免仅凭单一高分决定。

```bash
usq evaluate --level complex --stage validation --barra
```

初始配置中的收益或单调性门槛属于起步研究参数。开始正式筛选前应统一批准和冻结，不应在失败后不断放宽直到得到想要的回测。

<!-- PAGEBREAK -->

## 2 第三层评估 因子间相关性

本层解决因子冗余。默认使用 2015–2021 数据，按每 10 个交易日取一天、随机固定抽取 25% 证券进行降采样，随机种子固定为 2964。具体采样日期与证券都会写入元数据，确保复现。

每个采样日先对因子做截面排名和标准化，再把采样日及证券观测展开，计算因子对之间的 Pearson 相关。输出带符号相关矩阵及共同有效观测数；筛选使用绝对相关，以识别互为相反数的重复因子。

这一定义是采样排名因子值的 pooled correlation，不等于收益相关性，也不是 BRAIN 平台的 production correlation。它适合低成本排除明显冗余，最终入选因子可再用更高采样密度及 PnL 相关性复查。

### 筛选规则

1. 先按 2015–2021 的定向平均 RankIC 排序，相同质量用因子名确定顺序。
2. 先检查 2022 固定方向 RankIC、覆盖率及复杂评估门槛。
3. 对通过的候选，若与已选因子的绝对相关不低于 0.7，删除较后者。
4. 共同有效观测不足或相关系数缺失时保守处理，不能把缺失当成独立。
5. 将通过或删除原因、冲突因子、相关阈值及评估文件哈希写入 selected.json。

默认共同观测数门槛为 100。小股票池或高缺失因子需要重新设定合理门槛，不能沿用一个数值而忽略样本规模。

```bash
usq evaluate --level correlation --stage mining
usq select
```

可独立调用的 API 为 `evaluate_simple`、`evaluate_complex`、`evaluate_correlation`。简单层返回摘要和日序列；复杂层返回单因子回测、分组与可选归因；相关层返回相关矩阵、共同观测矩阵和筛选审计。

当没有因子通过预设规则时，流程中止并保留失败证据。不能为了让模型继续训练而隐式改变方向、降低阈值或强行保留至少一个因子。

<!-- PAGEBREAK -->

## 3 XGBoost 模型与半年重训

模型采用单一 XGBoost 回归器，使用冻结因子作为特征、十日中性化 ylabel 作为目标。从 2015 年开始扩展训练，2023–2025 每半年重训一次，总共六个模型。不进行四折训练、bagging、网格调参或早停。

| 参数 | 固定值 |
|---|---|
| objective | reg:squarederror |
| max_depth | 7 |
| learning_rate | 0.03 |
| n_estimators | 500 |
| subsample | 1 |
| colsample_bytree | 1 |
| gamma | 0.1 |
| reg_lambda | 1 |
| eval_metric | rmse |
| tree_method | hist |
| device | cuda |
| max_bin | 256 |
| missing | 配置为 null，原生接口使用 np.nan |

补充固定随机种子 2964；CPU 线程默认 48，给系统和数据处理预留资源。截图中的 missing=None 在本版解释为没有有限数值缺失哨兵，不能将真实零值当作缺失。[XGBoost 参数文档](https://xgboost.readthedocs.io/en/release_3.1.0/parameter.html)

每个因子每天先在当时股票池内做 0.5% 与 99.5% 分位截断，再做截面 zscore。该操作只使用同日已知的因子分布。NaN 特征由 XGBoost 处理，缺失标签不参与训练；禁止将未知 ylabel 填成零。

输入是 selected.json、冻结因子宽表、label_10、label_end 和 universe。输出包括六个 `.ubj` 模型、scores.pkl 和 training_audit.json。审计应显示每期实际设备、树数、训练起止日、最晚标签结束日、样本数、预测数、因子哈希及选择文件哈希。

本版训练使用连续 2015 起点。后续半年可以吸收已成熟的既往预测期标签，这是预先固定的在线重训政策；因子集合、方向、阈值、模型超参数均保持冻结。不能在看到回测后重新挑因子再称为同一次样本外检验。

<!-- PAGEBREAK -->

## 4 Score 简单回测与 50 组检查

### 默认多头组合

每天在 T 日已知股票池内按 score 从高到低选 Top-N，默认 N=100，等权配置。并列 score 使用输入中固定证券顺序打破选择并列，规则保持不变。与分层图不同，Top-N 必须产生确定的持仓集合。

信号 T 在 T+1 日 VWAP 形成目标持仓，赚取至 T+2 日 VWAP 的调整价格收益；报表行标注为区间结束日期。后续按每日目标权重调仓，不假设每天全部卖出再买入。初始 NAV 为 1，年末按市场价格估值，不额外假设强制清仓。

```text
gross_return[d] = sum_i weight[d,i] × asset_return[d,i]
traded[d] = sum_i abs(target[d,i] - drifted_previous[d,i])
cost[d] = traded[d] × cost_bps / 10000
net_return[d] = gross_return[d] - cost[d] - borrow_cost[d]
```

成本默认每买入或卖出一美元计 5bps。日单边换手为 traded/2；初次从现金建成满仓时，该指标为 0.5，成本仍对完整买入金额收取。多空研究默认另计年化 300bps 借券费，按 252 个交易日摊销，实际费率及可借性需要另行建模。

### 输出

回测报告展示成本前后净值、同池市值加权基准、相对净值、净收益回撤、换手、成本和年度收益。最大回撤把初始 NAV=1 纳入高水位，首日亏损不会被漏掉。

score 另外按每日排名分 50 组，G01 最低、G50 最高，展示平均日净收益、观测数及累计曲线。相同 score 不拆开，空组明确缺失；空档在分组净值图中按现金区间显示间断，并保留有效样本数。小股票池的 50 组结果会很噪，应结合每组股票数阅读。

选股不会预先过滤未来收益缺失的证券。持仓缺少有效价格时中止，要求补齐退市、并购或其他终值处理。预测区间内部整日没有 score 也会中止，避免跳过不利日期。

本版是目标权重研究回测。尚未模拟完整账户、整数股、融资和保证金、成交容量、市场冲击、逐笔撮合或交易失败后的持仓冻结，因此成本后研究收益仍不能直接称为真实账户净收益。

<!-- PAGEBREAK -->

## 5 Barra 研究代理模型的风格暴露

本模块使用美股数据构建透明的风格代理，通过约束横截面 WLS 估计因子收益并进行组合归因。它不是授权 MSCI USE4 产品，也不宣称复刻其全部描述子、风险调整及校准方法。

| 风格 | 本版代理定义 |
|---|---|
| Size | log(总市值) 的截面标准化 |
| Beta | 252 日股票收益与市场代理收益滚动协方差 / 市场方差，至少 126 日 |
| Momentum | log(close_adj[T-21] / close_adj[T-252]) |
| ResidualVolatility | 股票收益减去滚动 Beta 乘市场收益后的 126 日波动，至少 60 日 |
| NonLinearSize | 标准化 Size 的三次方，对常数和 Size 回归后的残差 |
| Liquidity | 21 日平均成交金额 / 市值的对数，至少 15 日 |
| BookToPrice | PIT 账面权益 / 当日市值 |
| EarningsYield | PIT TTM 利润 / 当日市值 |
| Growth | PIT TTM 收入相对 252 日前的变化率 |
| Leverage | PIT 总负债 / 总资产 |

每个风格在当日股票池内使用中位数和 MAD 去极值，再做截面 zscore。基本面缺少对应输入时，该风格不会伪造，模型元数据将列出缺失的可选风格。当前因子集合、公式、窗口与可用字段一起保存。

行业使用数据组提供的历史分类。回归同时包含 Market、完整行业哑变量和上述可用风格。所有股票的 Market 暴露为 1，行业因子加市值权重约束以解决共线性；这是解释市场效应与行业相对效应的结构。[MSCI 关于市场与行业约束的说明](https://www.msci.com/documents/10199/ed6e42a3-c1fa-4430-89ba-efd5a5b52558)

新上市股票和基本面缺失股票可能没有完整暴露。本版不会把持仓缺失暴露悄悄填为零；应在建立股票池时确定上市预热要求，并核对缺失覆盖。回归样本不足或设计矩阵共线时明确中止，保留诊断。

风格代理采用收盘历史构建，但收益归因回归使用与策略相同的 VWAP 区间收益。区分描述子输入价与所解释收益的价格口径，避免策略用 VWAP 而归因却使用另一套收盘收益。

<!-- PAGEBREAK -->

## 5 收益归因 风险估计与可视化

### 逐日收益归因

对区间 T+1 到 T+2，以 T 日已知暴露解释同一区间的股票 VWAP 调整价收益。WLS 目标权重为 sqrt(市值)，换成普通最小二乘输入时，行乘数为市值的四次方根；不要误乘完整平方根。

```text
r_i = Market_return + industry_exposure_i × industry_return
      + style_exposure_i × style_return + specific_return_i
portfolio_exposure = X' × portfolio_weights
portfolio_gross_return = sum(factor_exposure × factor_return)
                         + portfolio_specific_return
portfolio_net_return = portfolio_gross_return + Costs
```

成本项 Costs 为负，包含交易成本及借券估计。组合、基准、主动权重分别处理；主动暴露等于组合暴露减基准暴露。默认基准是同池可解释股票的信号日市值加权组合，应核对其证券覆盖是否与策略一致。

每日因子贡献、特异贡献和成本之和必须与组合净收益逐日对上。多期使用 Carino 链接，主动累计贡献之和对上“组合累计收益减基准累计收益”；这与组合净值除基准净值的相对收益是不同量，图表名称要分清。

### 风险估计

使用入场时已经观察到的历史因子收益协方差和特异方差估计组合风险，默认最多回看 252 日、至少 60 日。组合日方差近似为 b'Fb + sum(w_i² × specific_variance_i)，再年化。特异协方差暂按对角近似，未实现官方模型的波动状态、特异风险收缩或优化偏差调整。

### 可视化清单

必须展示策略、基准与相对净值；策略与基准回撤；组合、基准、主动风格暴露的时序；按月主动暴露热图；组合和主动累计收益贡献；预测风险曲线及回归覆盖诊断。因子单位收益、组合暴露和实际贡献不能混为一张含义不清的图。

```bash
usq backtest --barra
# 仅需重新构建该分析时，也可运行 usq barra
```

HTML 自包含，可离线打开。相应 CSV 保存于 `runs/barra/`，包括 factor_returns、specific_returns、exposures、contributions、linked、risk 和 regression_diagnostics。收益和回撤图直接使用回测的组合净收益序列，保证展示与计算同源。

<!-- PAGEBREAK -->

## 6 爆搜与因子登记

爆搜使用递归表达式生成器及其配套算子。先确定基础字段、已有因子、时间窗口和复杂度，再逐层展开候选，通过量纲类别与跨层算子规则过滤，按批次并行计算并评估。搜索范围和筛选规则在查看回测期之前冻结。

### 递归展开与过滤

complexity=1 生成基础字段及直接作用于字段的单、双字段表达式；complexity=2 或 3 继续对前层结果嵌套操作。已有标准因子或自定义核心结构可作为新的基础字段，参与同一套组合生成。

算子包括加减乘除、pow、log、abs、rank、zscore，滚动 mean、std、max、min、sum、median、skew、kurt，以及 pct_change、lag、delta、correlation、beta、alpha。保留原生成器的跨层限制，例如 rank 后不再嵌套 rank 或 zscore，mean 后不再嵌套 std 等统计操作。

字段注册为 price、volume、amount、percentage、unitless 等类别，默认仅保留 percentage 和 unitless 输出。新增字段必须声明类别，不自动把未知字段当成无量纲。规则用于兼容表达式生成，不替代对每个公式经济含义和严格量纲的审查。

高层单表达式按规则遍历；双表达式组合按固定种子在每层抽取至多 100 对，因此不是所有二叉表达式的完全穷举。候选保留源名称、结构树、输出类别和哈希，便于复现。

### 算子数值口径

rank 使用每日截面排名，并列值采用 method=min；mean、sum、median 的最少观测数为 1，std、max、min、skew、correlation、beta 为 3，kurt 为 5。保持这些原算子定义，不替换成另一套完整窗口或平均并列排名实现。

### 搜索预算与输出

max_trials 对应最终选择预算，top_k 限制保留完整因子矩阵的数量。生成器同时保留低层候选，小预算可能先被低层表达式占满；复杂度设为 3 不意味着已覆盖所有三层组合，应检查实际目录和数量。

```text
runs/mining/
  search_config.json
  expression_catalog.json
  trials.sqlite
  batches/factor_evaluation_report_id_00000.csv
  trial_rankic/<factor>.parquet
  factors/<factor>.pkl
  candidates.json
```

候选评价进程只接收截至 2021 年的面板及预热数据，且剔除标签结束日跨入 2022 的样本。入选因子再按冻结表达式计算完整日期范围，供验证和模型使用。评估保留全部成功、失败和淘汰日志。

支持多进程批次计算，默认 8 个工作进程、每批 64 个候选。Linux 默认 fork，其他平台默认 spawn 并显式初始化数据。先测峰值内存再增大并行度。恢复时按已完成候选日志续算；引擎、配置或数据哈希改变时使用新目录。

<!-- PAGEBREAK -->

## 6 爆搜配置与已有因子扩展

完整配置在 configs/default.yaml。下面示例演示把既有因子和核心结构加入基础字段，并要求新候选至少包含指定的既有因子：

```yaml
mining:
  fields: [open, high, low, close, volume, returns, vwap]
  complexity: 3
  windows: [5, 10, 20, 60, 120]
  time_periods: [1, 5, 10]
  dimension_filter: [unitless, percentage]
  max_trials: 5000
  max_workers: 8
  batch_size: 64
  seeds:
    - name: reversal
      expression: rank(ts_delta(close,5))
      dimension: unitless
  seed_factor_dir: data/seed_factors
  seed_factors:
    - name: wq101_003
      dimension: unitless
  required_fields: [wq101_003]
```

seed_factors 对应已经计算好的宽表，示例文件为 `data/seed_factors/wq101_003.pkl`。它必须与主数据日期及证券顺序完全一致。这里只注册公式已核对的因子；不把同名文件存在当成公式正确的证明。

seeds 中的核心结构先按同一套原算子计算为基础字段，再递归组合。兼容的 ts_delta、ts_mean、ts_std、ts_corr 等名称会映射为 delta、mean、std、correlation；未提供的算子明确拒绝，不自动混用旧模板的计算方法。

required_fields 留空时允许全部合规组合；填入字段名后，每个候选必须引用其中至少一个字段。它可用于专门挖掘基本面、某组标准因子或新核心结构的扩展，而不重复搜索完全无关的旧字段组合。

新增字段的类别通过 additional_field_dimensions 注册。价格使用 price，美元总金额使用 amount，股数使用 volume；收益率使用 percentage。自定义核心结构应明确输出类别，避免隐式猜测。

max_trials=5000 是小预算起点。实际大规模运行前先记录生成数量、重复结构数、单因子时间和峰值内存，再决定预算与进程数。提高复杂度会先构建中间层，预算限制不能保证中间目录的内存占用也同样小。

引擎版本与生成器、算子的文件哈希写入 search_config.json；表达式目录写入 expression_catalog.json。新引擎必须使用新实验目录，不能与旧模板结果混用。旧日志只提供读取兼容，不再用于生成新候选。

<!-- PAGEBREAK -->

## 6 WorldQuant101 与国泰君安191 的接入

这两组因子可作为标准基线和结构种子，与爆搜候选使用同一份数据、ylabel、时间划分和三层评估。不要给基准因子使用不同的股票池或收益口径。[101 Formulaic Alphas 原始论文](https://arxiv.org/abs/1601.00991)

本版提供 101 和 191 全部编号的批量输入检查与统一评估入口，未内置全部 292 个标准公式。因子负责人可以选择经核对的外部实现计算矩阵，再导入本系统。每个编号都记录已导入、缺失或全缺失状态；缺少行业、市值或指数输入时不能跳过后仍宣布全库完成。

| 因子库 | 文件命名 | 编号范围 |
|---|---|---|
| WorldQuant101 | wq101_001.pkl 等 | 001 至 101 |
| 国泰君安191 | gtja191_001.pkl 等 | 001 至 191 |

```bash
usq import-library --source work/wq_outputs \
  --family wq101 --require-complete
usq import-library --source work/gtja_outputs \
  --family gtja191 --require-complete
```

导入矩阵须与股票数据使用同一完整日期和证券顺序。导入时在 2015–2021 确定方向，并登记到候选清单；之后重新运行三层评估与 select。不可在模型冻结后悄悄补进回测表现好的经典因子。

### 实现核对要点

外部公式库先核对 rolling 窗口是否包含当天、排名的轴和并列值、标准差自由度、衰减权重方向、industry_neutralize 的分类粒度、分母为零、SMA 定义，以及非整数窗口的取整规则。多股票测试必须验证 rank 是同日截面，而非每只股票时间排名。

包含行业和基准指数的因子必须补齐对应历史输入。使用标准库文档定位参数依赖，再对照原始公式抽样手算或双实现比较。DolphinDB 提供两套模块的参数和用法说明，可作为实现调查入口，但引擎与模块使用条件需自行核实。[WQ101 文档](https://docs.dolphindb.com/en/Tutorials/wq101alpha.html)；[GTJA191 文档](https://docs.dolphindb.com/zh/modules/gtja191Alpha/191alpha.html)

提取核心结构继续爆搜时，保留原始家族编号、变更窗口与算子、表达式哈希和尝试次数。原因子、参数变体及反号版本都属于同一研究家族的一部分，多重检验不能按最终保留数量计算。

<!-- PAGEBREAK -->

## 7 时间划分与防止信息泄露

| 时间 | 用途 | 允许的操作 |
|---|---|---|
| 2014 或更早的补充数据 | 预热 | 构建长窗口和历史描述子，原则上不计入正式搜索评分 |
| 2015–2021 | 因子挖掘与初筛 | 搜索表达式、确定方向、初始相关性及预设阈值 |
| 2022 | 因子验证 | 固定方向下执行已确定的验证与复杂评估规则，冻结最终清单 |
| 2023–2025 | 滚动回测与可视化 | 六次固定政策训练和预测，输出回测及归因，不再改因子选择规则 |
| 2026 年初补充日期 | 价格与标签尾部 | 补齐 2025 年末标签；不纳入 2023–2025 收益报告 |

### 六个预测区间

```text
2023 H1：2023-01-01 至 2023-06-30
2023 H2：2023-07-01 至 2023-12-31
2024 H1：2024-01-01 至 2024-06-30
2024 H2：2024-07-01 至 2024-12-31
2025 H1：2025-01-01 至 2025-06-30
2025 H2：2025-07-01 至 2025-12-31
```

每个区间均从 2015 年开始训练，只使用 label_end 严格早于该区间起点的样本。例：2023 H2 训练可吸收截至 2023 年 6 月已经完全成熟的标签，不能使用结束日在 7 月的十日收益。

purge 按标签真实结束日期判断。默认 VWAP 标签为 T+11，所以不能机械照搬“删最后十行”而忽略入场延迟、节假日和标签缺失。2015–2021 搜索要求 label_end 早于 2022-01-01，2022 验证要求早于 2023-01-01。

2022 参与筛选以后就是模型选择数据，不再是独立证明。2023–2025 可以验证一套事先冻结的流程；如果据此改表达式、阈值、股票池、成本或模型，再报告同一段结果就具有选择偏差，需要另留未来区间。

保存每次实验的配置、哈希、因子方向、候选数、选择日志、模型训练审计和价格收益口径。后续模型吸收已发生数据是预设重训政策；后续人工根据回测重新选因子是另一种行为，二者要在研究记录中区分。

<!-- PAGEBREAK -->

## 8 验证是否有意义的研究课题

核心假设：爆搜产生的每个因子都没有真实预测力，日 RankIC 的长期均值为零。训练期高分只是多次尝试产生的极端值；再筛 2022，也可能只是二次挑选幸运者。

两名研究组员应等待候选日志和日 RankIC 产生后开展实证。第一步需要候选全集，包含失败或未保留的表达式，而不是只研究入选因子。应记录家族相关、尝试数、初筛规则和 2022 门槛。

### 实验一 零预测力下的筛选结果

固定候选集合，对各因子的日 RankIC 去均值，再联合按日期区块抽样，保持候选间相关结构和短期时间依赖。区块长度可比较 10、20、40 个交易日；独立构造伪训练期和伪验证期，重复完整的方向选择和筛选规则。

观察纯噪声下的训练最佳 RankIC、验证通过数、通过者的验证 RankIC 以及因子家族集中度。若算法生成表达式会根据标签自适应，还应在打乱标签上重新运行完整搜索；仅对固定候选 Bootstrap 不能校正自适应搜索路径。

### 实验二 比较选择流程的增量价值

冻结等数量 K 的两套方案：A 只按训练期取 Top-K；B 从固定的训练候选范围内再按 2022 验证规则取 K。候选数量不足要如实报告，不能让 B 取更少因子后直接比较而忽略数量影响。

比较 2023–2025 的单因子 RankIC、胜率、成本后 Sharpe、回撤、组合结果和风格暴露，并对增量做按日期及因子家族的区块置信区间。回测期只在两套方案都冻结后揭示一次，不能反复调整 B 的规则。

### 实验三 多重检验与风格解释

记录原始尝试数与有效独立维度，比较原始门槛和预设的多重检验控制。因子高度相关时，不能直接把所有试验当作独立正态变量；采用 FDR 等方法时也必须说明各自假设。

用 Barra 风格代理检查 2022 通过者是否集中在少数风格。若只因某年价值、动量或市值环境获得高分，验证可能主要选中了当年的市场状态，需要用多个历史滚动验证窗检验可迁移性。

结论应写成“在固定尝试预算与规则下，加入验证使未来表现提高多少，区间多宽”，或“没有足够证据显示改善”。不能从高验证分数直接推出验证有效，也不能仅因日 RankIC 正负波动就推出所有因子都是噪声。

<!-- PAGEBREAK -->

## 8 课题工具与证据文件

代码提供固定候选集合的区块 Bootstrap 工具，目的在于检查二次筛选的乐观偏差；它不是完整的自适应爆搜零假设实验，也不会直接给出“验证有用”的结论。

先从 trials.sqlite 中恢复全部成功计算的表达式，输出原始方向的训练与验证日 RankIC。方向在研究函数内部只按训练期确定。不要把已经定向的训练序列与未定向的验证序列混用。

```bash
usq trial-ic --stage mining --output runs/research/is_ic.parquet
usq trial-ic --stage validation --output runs/research/val_ic.parquet
usq research --is-ic runs/research/is_ic.parquet \
  --val-ic runs/research/val_ic.parquet \
  --output runs/research/null --k 20 --repeats 500 --block 20
```

两份输入矩阵的列必须完全相同，包含所有成功候选。训练日期不晚于 2021 年，验证日期限定为 2022 年；研究工具拒绝把 2023–2025 数据作为选择阶段输入。

当前工具输出候选数、训练与验证 RankIC 排名持续性、训练 Top-K 的验证均值、在验证期再次选择后的验证均值，以及条件零假设下对应分布。最后一项重复使用验证数据，必然带选择乐观偏差，工具会明确标注。

内置演示规则是：从训练 Top-3K 中，再按验证取 K。它仅用于展示如何比较相同数量的选择流程，不等于已经替代项目 select 中的 RankIC、分层和 Sharpe 门槛。课题负责人应在正式研究前冻结将要检验的规则，并在零假设重采样中完整重放相同规则。

真正评估验证的增量价值，还需要将 A、B 两套冻结因子清单分别送入同一模型和回测政策，比较未来增量；相关实验应另存配置，不能覆盖主策略结果。研究者同时报告没有通过或方向反转的候选，防止只展示幸运的子样本。

实验包至少包括候选全集 ID、表达式、来源家族、各期日 RankIC、缺失比例、全部尝试日志、阈值、选中列表、随机种子、区块长度、重复数及输出哈希。

<!-- PAGEBREAK -->

## 9 五人分工与依赖关系

| 角色 | 人数 | 主要职责 | 可验收产物 |
|---|---|---|---|
| 数据组 A | 1 | 行情、VWAP、公司行为、指数、历史证券与交易日历 | 来源比较表、行情与指数数据、稳定 ID 映射、质量样本 |
| 数据组 B | 1 | PIT 基本面、历史行业、市值、字段与单位对齐 | vintage 长表、行业与市值宽表、PIT 样本、数据字典 |
| 因子挖掘负责人 | 1 | 递归组合爆搜、101/191 接入、三层评估组织、候选归档 | 搜索配置、全试验日志、标准库完整性审计、评估结果、冻结候选 |
| 验证课题研究组 | 2 | 待因子结果产生后开展零假设与验证增量研究 | 冻结实验设计、候选全集分析、Null 分布、未来比较及结论 |

两名数据组员应交叉核验对方负责字段，尤其是市值单位、复权与财报时点。因子负责人不能在数据质量未通过时直接全量爆搜；课题组不能只拿到最终入选因子便开始推断多重检验结果。

### 里程碑

M0 资源和环境就绪：服务器配额核实、依赖安装、单元测试与小样本流程通过，确认真实 GPU 训练设备。

M1 数据接口冻结：两名数据负责人提交主来源、核对来源、字段字典、PIT 与退市样本；audit-data 通过，ylabel 和 VWAP 计算抽查完成。

M2 候选产生：因子负责人先测小预算搜索，再按冻结预算运行完整语法；保留所有尝试日志。标准库若纳入本轮，须在进入验证前完成登记。

M3 验证和选择冻结：完成简单、复杂、相关三层评估，生成 selected.json，确认方向、成本和模型政策；之后才能揭示正式回测结果。

M4 主流程回测：生成六个半年模型、score、50 组结果和收益回撤图，核对持仓价格与基准口径，完成逐日和多期归因对账。

M5 研究课题：两名研究人员按提前冻结的规则比较 A、B 和零假设结果，结合风格暴露解释增量。结果可以是不显著，不能以必须获得正结论作为验收标准。

模型、回测和归因的运行由团队在 M1 前指定一名集成对接人，负责版本、参数和产物核对。这里不额外假设第六名组员，也不改变已约定的五人职责。

<!-- PAGEBREAK -->

## 附录 一次正式实验的运行顺序

先在仓库根目录修改 configs/default.yaml 中的路径。真实数据不得放入 Git；为每次正式实验创建独立 run_dir，并同步 factor_dir 指向该轮 mining/factors。

```bash
# 1 数据组验收，生成标签
usq audit-data
usq prepare

# 2 固定预算搜索，确定训练期方向
usq mine

# 3 简单评估
usq evaluate --level simple --stage mining
usq evaluate --level simple --stage validation

# 4 复杂复筛与降采样相关
usq evaluate --level complex --stage validation --barra
usq evaluate --level correlation --stage mining
usq select

# 5 清单冻结后，训练和评估未来区间
usq train
usq backtest --barra
```

有额外配置时，将 `--config 路径` 放在子命令前，例如：

```bash
usq --config configs/experiment_01.yaml train
```

所有三层评估都可独立运行。复杂评估默认不强制计算 Barra，需要时加 --barra；正式复筛如计划使用风格信息，就应纳入该选项并核对模型字段。

101/191 应在最终验证与 select 前导入。导入后重新执行简单和复杂评估、相关计算与选择，不直接手改已经冻结的清单。若搜索或验证规则改变，另开实验目录并保留旧版本。

小样本人工数据验证可运行 `usq demo --output runs/demo --full`。它会生成虚构证券和工作日日期，明确使用 CPU 并训练六个 500 树模型；这些日期不是美股交易日历，人工数据指标不能用来评价策略有效性。

<!-- PAGEBREAK -->

## 附录 产物验收与故障处理

| 阶段 | 主要产物 | 核对内容 |
|---|---|---|
| 数据 | data_audit.json、manifest、字典与样本 | 日期证券轴、单位、PIT、退市、复权及来源 |
| 标签 | label_10.pkl、label_end.csv、label_metadata.json | 十日定义、WLS 控制项、标准化、实际结束日期 |
| 爆搜 | trials.sqlite、trial_rankic、candidates.json | 全尝试数、错误数、预算、表达式和固定方向 |
| 简单评估 | mining_simple、validation_simple 与日序列 | IC、RankIC、ICIR、CV、覆盖率 |
| 复杂评估 | 单因子 HTML、分组、净收益及可选归因 | 单调性、成本、Sharpe、Fitness、暴露 |
| 相关与选择 | correlation、overlap、selected.json | 采样、相关阈值、冲突及删除原因 |
| 模型 | 六个 ubj、scores.pkl、training_audit.json | 500 树、实际设备、标签成熟边界及文件哈希 |
| 回测与归因 | backtest 与 barra 目录下 HTML 和 CSV | 50 组、收益回撤、逐日与多期贡献恒等式 |

常见失败及处理：数据轴不一致时回到数据转换阶段，禁止直接取交集掩盖丢失；标签尾部缺失时补 2026 年初数据或保留缺失；没有因子通过时先报告现有证据，不自动放宽阈值；CUDA 不可用时检查安装和驱动，不把 CPU 结果记作 GPU 验收。

持仓价格缺失需要核对公司行为和退市终值；不能删除未来出问题的股票。Barra 暴露缺失需要检查 PIT 字段和预热覆盖，或在建池时预先规定上市天数；不要事后按回测表现删证券。

预测全日缺失时检查数据和因子覆盖；如希望采用现金或延续旧仓，需事先定义并实现策略政策，再重新运行。研究回测与风险模型不能各自把同一缺失收益处理成不同数值。

正式验收不仅看程序退出码，还应随机抽取若干日期手工复算标签、选股、组合收益、成本和归因贡献。保存版本和参数，使另一名组员在相同数据快照上能够复现。

<!-- PAGEBREAK -->

## 参考资料与方法边界

以下来源用于核对数据能力、公式库依赖、XGBoost 参数及风险模型结构。数据订阅、账户访问和具体授权范围应由实际采购者核实；本手册不声称已获得任何付费数据权限。

- [CRSP US Stock Databases](https://www.crsp.org/research/crsp-us-stock-databases/?activetab=docs)
- [Nasdaq Data Link](https://www.nasdaq.com/products/data/data-link)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Massive VWAP 计算说明](https://massive.com/knowledge-base/article/how-is-a-stock-bars-vwap-calculated-and-why-can-it-fall-outside-the-high-and-low)
- [Alpha Vantage API 文档](https://www.alphavantage.co/documentation/)
- [XGBoost 3.1.3 参数](https://xgboost.readthedocs.io/en/release_3.1.0/parameter.html)
- [XGBoost 3.1.3 GPU](https://xgboost.readthedocs.io/en/release_3.1.0/gpu/index.html)
- [101 Formulaic Alphas](https://arxiv.org/abs/1601.00991)
- [DolphinDB WQ101](https://docs.dolphindb.com/en/Tutorials/wq101alpha.html)
- [DolphinDB GTJA191](https://docs.dolphindb.com/zh/modules/gtja191Alpha/191alpha.html)
- [MSCI Market Factor](https://www.msci.com/documents/10199/ed6e42a3-c1fa-4430-89ba-efd5a5b52558)
- [MSCI USE4 产品说明](https://www.msci.com/documents/10199/242721/Barra_US_Equity_Model_USE4.pdf/d7625289-cade-4e88-96ae-696219af5b67)

统计诊断不能替代独立未来表现。Fitness 为本地定义，Barra 为透明代理模型；评估结果应连同样本区间、覆盖、成本及估计方法一起报告。

本系统首先服务可复现的研究：让团队明确输入来自哪里、信号何时可知、标签何时成熟、模型何时更新，以及每一笔收益如何从价格和权重算出。只有真实数据验收、冻结规则和回测归因对账同时完成后，才进入实际策略有效性的讨论。
