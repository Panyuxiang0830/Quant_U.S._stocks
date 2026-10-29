# 美股量化研究系统

日频数据接口 → 十日市值行业中性标签 → 爆搜 → 三层因子评估 → 半年重训 XGBoost → 50 组回测 → 风格风险及收益归因。

组员搭建与分工请阅读 [详细说明书](docs/HANDBOOK.md)。数据由数据组寻找和整理，本仓库不附带真实行情、财报或策略收益结果。

## 安装和验证

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
usq demo --output runs/demo --full
```

`demo --full` 使用人工数据，实际训练六个半年模型，每个固定 500 棵树，明确使用 CPU。生成自包含的回测和 Barra HTML。它不验证真实美股 Alpha，也不替代服务器 CUDA 验收。Mac 若缺 OpenMP，需安装 `libomp`；生产环境建议 Linux。

## 正式运行顺序

先准备 `data/panels/` 与独立交易日历 `calendar.csv`，按说明书统一稳定证券 ID、复权、PIT 基本面和退市处理。路径可在 `configs/default.yaml` 修改。所有命令从仓库根目录执行。

```bash
usq audit-data
usq prepare
usq mine
usq evaluate --level simple --stage mining
usq evaluate --level simple --stage validation
usq barra-build
usq evaluate --level complex --stage validation --barra
usq evaluate --level correlation --stage mining
usq select
usq train
usq backtest --barra
```

`select` 未通过既定规则时明确中止，不会为凑数量强行保留因子。首次正式运行前应冻结候选预算、阈值、交易成本和数据版本。

## 代码分类

| 目录 | 功能 |
|---|---|
| `src/usq/data/` | 字段宽表、PIT 展开、多来源比较、数据审计、VWAP 和标签 |
| `src/usq/evaluation/` | 简单、复杂、因子间三层独立调用 |
| `src/usq/mining/` | 原版递归生成器与算子、多进程批次、量纲/跨层过滤、已有因子扩展、断点日志及 IC 导出 |
| `src/usq/model/` | 单模型 XGBoost，2015 起扩展训练，2023–2025 每半年重训 |
| `src/usq/backtest/` | VWAP 区间收益、成本、换手、50 组与指标 |
| `src/usq/barra/` | 参考描述子/合成/正交化核心、美股适配、固定模型库、持仓归因与原报告扩展 |
| `src/usq/research/` | 候选全集的区块 Bootstrap 验证研究工具 |
| `src/usq/report.py` | 离线净值、回撤、分组、暴露及归因可视化 |
| `tests/` | 时间对齐、purge、CV、PIT、缺失值、归因恒等式等测试 |

## 口径与实现边界

| 基础指标 | 定义 |
|---|---|
| IC | 日因子值与标签的截面 Pearson 相关 |
| RankIC | 日因子排名与标签排名的截面相关 |
| ICIR | 日 IC 均值除以日 IC 标准差；RankICIR 同理 |
| CV | 每日 RankIC 为正时除以前两个日历年历史最小值，为负时除以历史最大值；不含当天，历史不足或零分母记为缺失 |

- 默认标签为 `VWAP_adj[T+11]/VWAP_adj[T+1]-1`，再做信号日市值行业 WLS 残差和截面标准化。`label.mode: close_sum` 可切到旧式未来十日收盘日收益之和；标签元数据会记录差异。
- 模型使用截图参数，固定 500 树，不早停、不四折、不 bagging。`missing: null` 转为原生接口的 `np.nan`，零值仍是有效值。
- 回测是日频目标权重研究模拟。含按成交金额计的成本及多空借券费估算，不包含完整账户、逐笔撮合、容量冲击、资金约束或强制清仓。
- Barra 沿用参考实现的描述子、标准化、合成、正交化、WLS 和收益归因核心，提供美股财务与 VWAP 适配；先固化模型库，再输入持仓。原报告增加收益、回撤和成本。不是授权 MSCI USE4 产品；迁移边界见 [BARRA_PORT.md](docs/BARRA_PORT.md)。
- 101/191 提供完整编号的批量导入、缺失审计和统一评估接口，**未内置全部 292 条标准公式实现**。可将经核对的外部公式输出接入；`--require-complete` 会阻止缺号时声称全库完成。
- 默认 CUDA；若 GPU 不可用会拒绝静默回退。CPU 运行须在配置中明确指定。
- 爆搜默认使用 `FactorExpressionGenerator` 与配套 `Operators`，可配置复杂度、窗口、量纲类别、已有因子和并行进程。原版高层双表达式配对每层最多抽取 100 对；它不是所有树结构的完全穷举。旧模板仅用于历史日志读取，不再作为默认生成器。

计算口径见 [实现口径记录](docs/REFERENCE_AUDIT.md)，迁移边界见 [递归引擎核对](docs/ENGINE_PORT.md)，验证范围见 [验证记录](docs/VALIDATION.md)。
