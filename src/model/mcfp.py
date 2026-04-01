"""
多商品流问题 (MCFP) 求解模块 — 实现论文 §6.2 中的货物流量分配。

给定当前航线集合 R_u，求解最优的货物流量分配。
这是元启发式中每次迭代评估航线网络质量的核心组件。

参考: Brouer et al. (2014), Transportation Science 48(2), pp. 299-300, Figure 7
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pulp

from src.utils.cost_calculator import VesselClass, Port, CostCalculator, CostConfig
from src.utils.network_builder import Rotation


@dataclass
class MCFPResult:
    """MCFP 求解结果。"""
    status: str
    objective: float                           # 目标函数值 (成本 - 收入)
    total_revenue: float                       # 总运输收入
    total_handling_cost: float                 # 总装卸费
    total_transship_cost: float                # 总转运费
    total_reject_penalty: float                # 总拒运惩罚
    flow: Dict[str, Dict[Tuple[str, str], float]]  # rotation -> (o,d) -> flow
    rejected: Dict[Tuple[str, str], float]     # (o,d) -> rejected FFE
    residual_demand: Dict[Tuple[str, str], float]  # (o,d) -> residual demand


class MCFPSolver:
    """
    多商品流问题求解器。

    构建论文 Figure 7 描述的图结构并求解 LP。
    使用 LP 松弛（连续变量）以快速求解。
    """

    def __init__(
        self,
        rotations: List[Rotation],
        ports_dict: Dict[str, Port],
        demands: List[Tuple[str, str, float, float]],  # (o, d, quantity, revenue)
        dist_min: Dict[Tuple[str, str], float],
        config: CostConfig = None,
    ):
        self.rotations = rotations
        self.ports_dict = ports_dict
        self.demands = demands
        self.dist_min = dist_min
        self.config = config or CostConfig()
        self.cost_calc = CostCalculator(self.config)

    def solve(self, vessel_assignments: Optional[Dict[str, int]] = None,
              time_limit: int = 120) -> MCFPResult:
        """
        求解多商品流问题。

        参数:
        - vessel_assignments: 每条航线的船舶分配数 (若 None，使用航线自带值)
        - time_limit: 求解时限

        返回:
        - MCFPResult
        """
        model = pulp.LpProblem("MCFP", pulp.LpMinimize)
        R = self.rotations
        G = self.demands

        # 预计算航线容量
        rot_caps = {}
        for rot in R:
            n_vessels = vessel_assignments.get(rot.id, rot.num_vessels) if vessel_assignments else rot.num_vessels
            round_trip_days = self.cost_calc.rotation_round_trip_time(
                rot.vessel_class, rot.speed, rot.port_calls, self.dist_min
            )
            m_r = self.cost_calc.num_round_trips(round_trip_days)
            rot_caps[rot.id] = {
                "n_vessels": n_vessels,
                "m_r": m_r,
                "capacity_per_leg": rot.vessel_class.capacity * m_r * n_vessels,
            }

        # ========== 变量 ==========

        # V_r_od: 在航线 r 起点 o 处首次装载的 (o,d) 需求
        V = {}
        for rot in R:
            for o, d, _, _ in G:
                if o in rot.port_calls:
                    V[(rot.id, o, d)] = pulp.LpVariable(
                        f"V_{rot.id}_{o}_{d}", lowBound=0
                    )

        # X_r(ij)d: 航线 r 边 (i,j) 上运往 d 的流量
        X = {}
        for rot in R:
            for i, j in rot.edges:
                for _, d, _, _ in G:
                    X[(rot.id, i, j, d)] = pulp.LpVariable(
                        f"X_{rot.id}_{i}_{j}_{d}", lowBound=0
                    )

        # U_rs(port, d): 在港口 port 从航线 r 转运到航线 s 的 d 方向流量
        U = {}
        for r_rot in R:
            for s_rot in R:
                if r_rot.id == s_rot.id:
                    continue
                common = set(r_rot.port_calls) & set(s_rot.port_calls)
                for port in common:
                    for _, d, _, _ in G:
                        if d != port:
                            U[(r_rot.id, s_rot.id, port, d)] = pulp.LpVariable(
                                f"U_{r_rot.id}_{s_rot.id}_{port}_{d}", lowBound=0
                            )

        # O_od: 拒运量
        O = {}
        for o, d, _, _ in G:
            O[(o, d)] = pulp.LpVariable(f"O_{o}_{d}", lowBound=0)

        # ========== 目标函数 ==========
        # 最小化: 拒运惩罚 - 运输收入 + 装卸费 + 转运费
        # (航线固定运营成本在外部计算，MCFP 只优化流量)
        obj = pulp.LpAffineExpression()

        for o, d, k_od, q_od in G:
            # 拒运惩罚
            obj += self.config.reject_penalty * O[(o, d)]
            # 运输收入 (负值, 因为最小化)
            for rot in R:
                if (rot.id, o, d) in V:
                    obj -= q_od * V[(rot.id, o, d)]
            # 装卸费
            port_o = self.ports_dict.get(o)
            port_d = self.ports_dict.get(d)
            for rot in R:
                if (rot.id, o, d) in V:
                    if port_o:
                        obj += port_o.move_cost * V[(rot.id, o, d)]
                    # 卸货费在目的地 (通过到达 d 的流量计算)

        # 卸货费
        for rot in R:
            for i, j in rot.edges:
                port = self.ports_dict.get(j)
                if port:
                    # X_r(ij)j 是到达目的地 j 的流量
                    if (rot.id, i, j, j) in X:
                        obj += port.move_cost * X[(rot.id, i, j, j)]

        # 转运费
        for key, u_var in U.items():
            _, _, port_code, _ = key
            port = self.ports_dict.get(port_code)
            if port:
                obj += port.transship_cost * u_var

        model += obj, "MCFP_objective"

        # ========== 约束 ==========

        # 需求满足: O_od + sum_r V_r_od = k_od
        for o, d, k_od, _ in G:
            model += (
                O[(o, d)] + pulp.lpSum(
                    V[(rot.id, o, d)] for rot in R if (rot.id, o, d) in V
                ) == k_od,
                f"demand_{o}_{d}",
            )

        # 流量守恒 (简化: 非蝴蝶)
        # 使用集合去重: 同一 (rot, triple_idx, d) 只需一个约束
        for rot in R:
            triples = rot.triples
            seen_constraints = set()
            for t_idx, (h, i, j) in enumerate(triples):
                for _, d, _, _ in G:
                    if i == d:
                        continue
                    ckey = (rot.id, t_idx, d)
                    if ckey in seen_constraints:
                        continue
                    seen_constraints.add(ckey)

                    lhs = pulp.LpAffineExpression()
                    if (rot.id, h, i, d) in X:
                        lhs += X[(rot.id, h, i, d)]
                    if (rot.id, i, d) in V:
                        lhs += V[(rot.id, i, d)]
                    for s_rot in R:
                        if s_rot.id == rot.id:
                            continue
                        key = (s_rot.id, rot.id, i, d)
                        if key in U:
                            lhs += U[key]

                    rhs = pulp.LpAffineExpression()
                    if (rot.id, i, j, d) in X:
                        rhs += X[(rot.id, i, j, d)]
                    for s_rot in R:
                        if s_rot.id == rot.id:
                            continue
                        key = (rot.id, s_rot.id, i, d)
                        if key in U:
                            rhs += U[key]

                    model += (lhs == rhs, f"flow_{rot.id}_t{t_idx}_{d}")

        # 边容量约束
        for rot in R:
            cap = rot_caps[rot.id]["capacity_per_leg"]
            for i, j in rot.edges:
                model += (
                    pulp.lpSum(
                        X[(rot.id, i, j, d)]
                        for _, d, _, _ in G
                        if (rot.id, i, j, d) in X
                    ) <= cap,
                    f"cap_{rot.id}_{i}_{j}",
                )

        # ========== 求解 ==========
        solver = pulp.COIN_CMD(path="/opt/homebrew/bin/cbc", timeLimit=time_limit, msg=False)
        model.solve(solver)

        status = pulp.LpStatus[model.status]
        objective = pulp.value(model.objective) if model.status == 1 else float("inf")

        # 提取结果
        rejected = {}
        residual = {}
        for o, d, k_od, _ in G:
            o_val = pulp.value(O[(o, d)]) or 0.0
            rejected[(o, d)] = o_val
            residual[(o, d)] = o_val  # 残差需求 = 拒运量

        flow = {}
        for rot in R:
            rot_flow = {}
            for o, d, _, _ in G:
                if (rot.id, o, d) in V:
                    val = pulp.value(V[(rot.id, o, d)]) or 0.0
                    if val > 0.001:
                        rot_flow[(o, d)] = val
            if rot_flow:
                flow[rot.id] = rot_flow

        # 统计
        total_revenue = 0.0
        total_handling = 0.0
        total_transship = 0.0
        total_penalty = 0.0

        for o, d, k_od, q_od in G:
            transported = k_od - rejected.get((o, d), 0.0)
            total_revenue += q_od * transported
            port_o = self.ports_dict.get(o)
            port_d = self.ports_dict.get(d)
            if port_o:
                total_handling += port_o.move_cost * transported
            if port_d:
                total_handling += port_d.move_cost * transported
            total_penalty += self.config.reject_penalty * rejected.get((o, d), 0.0)

        for key, u_var in U.items():
            val = pulp.value(u_var) or 0.0
            if val > 0.001:
                _, _, port_code, _ = key
                port = self.ports_dict.get(port_code)
                if port:
                    total_transship += port.transship_cost * val

        return MCFPResult(
            status=status,
            objective=objective,
            total_revenue=total_revenue,
            total_handling_cost=total_handling,
            total_transship_cost=total_transship,
            total_reject_penalty=total_penalty,
            flow=flow,
            rejected=rejected,
            residual_demand=residual,
        )
