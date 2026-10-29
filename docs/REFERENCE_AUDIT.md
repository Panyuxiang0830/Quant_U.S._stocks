# 远端实现核对记录

核对日期：2026-10-29。只读检查 gpushare 的实际源文件和一个真实字段面板；没有把旧历史回测输出当作当前系统运行结果。

## 数据格式

`/home/research_pyx/barra_analysis/barra/data.py` 从 `/hy-tmp/data/daily_update/<field>.pkl` 读取 DataFrame，索引为 `YYYYMMDD` 字符串、列为股票代码。实际读取 `close_adj.pkl`，形状 `(1896, 5427)`，首日 `20190102`，数值 float64，示例列 `000001.SZ`、`000002.SZ`、`000004.SZ`。

另有 `/hy-tmp/autodl-tmp/EOD_Columns/eod` 约 9,993,431,318 字节的单文件容器。新系统选择独立字段宽表接口，不要求数据组重建旧 10GB 容器。美股列使用稳定证券 ID，单位另定为美元和股。

## 模型

`/home/research_pyx/factor_filter/ff/xgb_tree.py` 同时有 tree 与 LIVE_PARAMS 两套配置。截图对应 LIVE_PARAMS：深度 7、学习率 .03、固定 500 树、subsample=1、colsample_bytree=1、gamma=.1、lambda=1、hist/cuda、max_bin=256。

新系统沿用这一组参数，按本项目要求改为 2015 起点、2023–2025 半年扩展重训，移除原来的四折、季度划分、bagging 和早停。原生 XGBoost 通过 num_boost_round=500 实现固定树数，None 缺失配置映射为 np.nan。

## 标签与 CV 的差异

`/home/model/整合版.ipynb` 的 `eod_Forward_Barra_Regression` 对未来 `pct_chg` 做求和，对 log(close×shares) 和行业回归，实际 WLS 权重使用 log 市值；其中还引用未来首日 trade_status。

新系统不照搬未来交易状态筛选，也不允许未来窗口部分缺失后跳过缺失求和。默认目标改为可执行的 VWAP T+1→T+11 复合收益，WLS 权重采用 sqrt(cap)，并可显式选择 close_sum 模式。二者不是对旧标签逐字复刻，差异写入标签元数据。

`/home/research_pyx/GP/scripts/calculate_hof_cv.py` 中的旧指标为“20 日滚动 RankIC 均值极值 / 全期均值”。本项目依据最新明确约定使用“当天日 RankIC / 前两年日 RankIC 极值”，正值选最小、负值选最大，窗口不含当天。它既不是旧脚本的同名统计量，也不是变异系数。

## Barra

`barra/model.py` 使用 Market + 全行业 + 风格的联合横截面回归，以行业市值权重收益和为零约束消除共线性。回归目标权重 sqrt(cap)，OLS 行乘数 cap**.25。

`barra/attribution.py` 区分组合/基准/主动权重、收益贡献和暴露，并使用 Carino 多期链接。旧代码把缺失个股收益视为零；新系统对持仓缺失价格明确报错，并单列成本。

`barra/report.py` 已包含贡献、组合/基准/主动暴露和因子单位收益等展示；核对中未见完整组合净值与回撤图。新报告增加策略/基准/相对净值和回撤，回归收益与策略 VWAP 区间一致。风格描述子重新定义为透明美股研究代理，未宣称复刻 CNE5 或 USE4。

## 源文件 SHA256

| 源文件 | SHA256 |
|---|---|
| barra/data.py | bddbc01a0eebc265f78768b1ab5a61ecab2244ea1eaac308696e667480d1377e |
| barra/model.py | a1d8f14203d0b4081fd066c5383a2f273b9cd10ec1a2ac27693dde1e88757ec6 |
| barra/attribution.py | 4b99089db5e5a0376cf1adb2515446e5d1ff2bac680ed5942902be79934f6b06 |
| barra/report.py | 4e2dff993a73c64707ac54819b63728f684090008c10917162483a957b901a5a |
| ff/xgb_tree.py | fab6c9ae2d8b1772d4acc8f34cf92c9a9de0a3d2ea090df252a798595f5aa5ba |
| model/整合版.ipynb | 0356dee9bccb9ed15fc6059623eb402202876b58f5313f26ce1932ff1eb3253c |
| GP/scripts/calculate_hof_cv.py | b1d52674912c3e737f22ff52c6fa6bdb3eefc0e49fd7562b67046faa6ca9ba67 |
