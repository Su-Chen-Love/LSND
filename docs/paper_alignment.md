# LSND 论文对齐审计

本文档把论文《A Base Integer Programming Model and Benchmark Suite for Liner-Shipping Network Design》中的关键建模对象映射到当前代码，并标记当前阶段的实现状态。

## §6.1 Reference MIP

| 论文对象 | 代码模块 | 当前状态 | 备注 |
|---|---|---|---|
| 目标函数 (2a)–(2d) | `src/model/mip_model.py` | `部分实现` | 已覆盖 TC、fuel、port、canal、handling、reject penalty、charter-out；仍需继续核对三索引流量与转运统计。 |
| 约束 (3) 蝴蝶流量守恒 | `src/model/mip_model.py` | `实现风险` | 现有实现仍是直接索引式，后续需继续比对论文中的 `U_rs(hi)d` 细节。 |
| 约束 (4) 到达目的地流量 | `src/model/mip_model.py` | `已实现` | 采用 `W_r(ij)` 辅助变量。 |
| 约束 (6) 容量绑定 `m_r * Y_r` | `src/model/mip_model.py` | `已实现` | 仍需继续验证与蝴蝶/重复港口交互时的正确性。 |
| 端口吃水兼容 | `src/model/mip_model.py` / `src/utils/cost_calculator.py` | `部分实现` | 距离与运河数据已支持吃水过滤，但主模型尚未系统性使用到所有路径选择逻辑。 |

## §6.2 Metaheuristic + MCFP

| 论文对象 | 代码模块 | 当前状态 | 备注 |
|---|---|---|---|
| Figure 7 终端节点 + port-call 节点图 | `src/model/mcfp.py`, `src/utils/network_builder.py` | `已重构` | 现在显式建模 `load / unload / sail / transship / omit` 边。 |
| MCFP commodity flow | `src/model/mcfp.py` | `已重构` | 使用 source/sink + call-node 守恒，支持重复港口和 `r = s` 转运边。 |
| 候选列评估 `R^u ∪ {col}` | `src/algorithm/metaheuristic.py` | `已调整` | 从“只接受全局改进”改为“按当前邻域最佳网络推进，并独立维护全局最优”。 |
| Backtracking / tabu | `src/algorithm/metaheuristic.py`, `src/algorithm/tabu_search.py` | `部分实现` | 已保留回退与删除逻辑；MIP gap 评分仍是近似项。 |

## §6.3 AUX(v, s, η)

| 论文对象 | 代码模块 | 当前状态 | 备注 |
|---|---|---|---|
| 约束 (19) `Q_od <= tau * k_hat / eta` | `src/model/route_generation.py` | `已修正` | 去掉了旧的固定轮次近似。 |
| 约束 (22)(23) `phi_out / phi_in` | `src/model/route_generation.py` | `已修正` | 改为按残差需求份额计算，不再强制最小 0.15。 |
| 业务规则 1: 大船周频 | `src/model/route_generation.py` | `已实现` | `capacity >= 1200` 时强制 weekly。 |
| 业务规则 2: `capacity >= 4200` 至少四周 rotation | `src/model/route_generation.py` | `已实现` | 通过 `tau >= 28 / T` 加入。 |
| 业务规则 3 | - | `未实现` | 与论文一致，暂不实现。 |

## Benchmark / Observability

| 目标 | 代码模块 | 当前状态 | 备注 |
|---|---|---|---|
| Base / Low / High 场景 | `src/data_reader.py` | `已实现` | 按 LINER-LIB README 缩放 TC rate 和 quantity。 |
| Solver backend 记录 | `src/utils/solver_backend.py`, `src/algorithm/solver.py` | `已实现` | 支持 `auto / gurobi / cbc`，当前环境自动回退 CBC。 |
| 多次运行与种子 | `tests/run_benchmark.py`, `src/algorithm/solver.py` | `已实现` | 支持 `--runs`、`--seed-base`。 |
| 诊断文件导出 | `src/algorithm/solver.py` | `已实现` | 可输出 JSON 诊断文件。 |

## 下一步重点

1. 继续核对 `src/model/mip_model.py` 的三索引流量与转运变量索引，减少与论文公式 (3)–(4) 的偏差。
2. 逐步把吃水兼容距离接入所有 route cost 计算，而不只停留在基础工具层。
3. 用 `Baltic Base` 多次复现实验验证本轮 `MCFP + metaheuristic` 改造是否显著降低拒绝率并改善目标值。
