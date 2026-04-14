"""
LSNDP MIP 主模型 — 严格实现 Brouer et al. (2014) §6.1 公式 (2)-(10)。

给定一组航线 (rotations) R，求解最优的船舶分配和货物流量分配。
这是论文中"Reference Model for the LSNDP"的完整实现。

参考: Transportation Science 48(2), pp. 297-299
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

import pulp

from src.utils.cost_calculator import (
    VesselClass, Port, CostCalculator, CostConfig,
    build_vessels_from_data, build_ports_from_data,
    build_distance_dict,
)
from src.utils.network_builder import Rotation
from src.utils.solver_backend import SolverBackend, build_pulp_solver


@dataclass
class CostBreakdown:
    """成本分解，对应论文 Table 9 的列。"""
    Z: float = 0.0       # 总目标值
    Q: float = 0.0       # 总收入
    c_v: float = 0.0     # 船舶 TC 费用
    c_b: float = 0.0     # 燃油费用
    c_p: float = 0.0     # 港口费用
    c_c: float = 0.0     # 运河费用
    c_m: float = 0.0     # 装卸费用
    c_t: float = 0.0     # 转运费用
    L_v: float = 0.0     # 未使用船舶价值 (charter out revenue)
    F: float = 0.0       # 被拒绝运量 (FFE)


@dataclass
class MIPResult:
    """MIP 模型求解结果。"""
    status: str                              # 求解状态
    objective: float                         # 目标函数值
    cost_breakdown: CostBreakdown            # 成本分解
    vessel_assignments: Dict[str, int]       # rotation_id -> num_vessels
    flow_on_edges: Dict                      # 边流量
    rejected_demand: Dict[Tuple[str, str], float]  # (o,d) -> rejected FFE
    transported_demand: Dict[Tuple[str, str], float]  # (o,d) -> transported FFE
    active_backend: str = "cbc"


class LsndpMIP:
    """
    LSNDP MIP 模型，实现论文公式 (2)-(10)。

    给定航线集合 R，求解:
    - Y_r: 每条航线的船舶分配数
    - X_r(ij)d: 边流量
    - U_rs(hi)d: 转运流量
    - V_r_od: 首次装载量
    - O_od: 拒运量
    """

    def __init__(
        self,
        rotations: List[Rotation],
        vessels: List[VesselClass],
        ports_dict: Dict[str, Port],
        demands: List[Tuple[str, str, float, float]],  # (o, d, quantity, revenue)
        dist_min: Dict[Tuple[str, str], float],
        canal_info: Dict[Tuple[str, str], Tuple[bool, bool]],
        config: CostConfig = None,
        solver_backend: SolverBackend = "auto",
    ):
        self.rotations = rotations
        self.vessels = {v.name: v for v in vessels}
        self.ports_dict = ports_dict
        self.demands = demands
        self.dist_min = dist_min
        self.canal_info = canal_info
        self.config = config or CostConfig()
        self.cost_calc = CostCalculator(self.config)
        self.solver_backend = solver_backend
        self.model = None
        self.result = None

    def build(self):
        """构建 MIP 模型。"""
        model = pulp.LpProblem("LSNDP", pulp.LpMinimize)
        R = self.rotations
        G = self.demands
        P = set()
        for rot in R:
            P.update(rot.port_calls)
        for o, d, _, _ in G:
            P.add(o)
            P.add(d)
        P = list(P)

        # ========== 预计算航线参数 ==========
        rot_params = {}
        for rot in R:
            v = rot.vessel_class
            s = rot.speed
            round_trip_days = self.cost_calc.rotation_round_trip_time(
                v, s, rot.port_calls, self.dist_min
            )
            m_r = self.cost_calc.num_round_trips(round_trip_days)

            # 每次航行的运营成本 (公式 2b 中 m_r * Y_r 内的部分)
            per_sailing_cost = 0.0
            for k in range(len(rot.port_calls)):
                i_port = rot.port_calls[k]
                j_port = rot.port_calls[(k + 1) % len(rot.port_calls)]
                dist = self.dist_min.get((i_port, j_port), 0.0)

                # 燃油: e * h(v) * p(v,j) + e * g(v,s) * l(v,i,j)
                per_sailing_cost += self.cost_calc.port_idle_fuel_cost(v)
                per_sailing_cost += self.cost_calc.sailing_fuel_cost(v, s, dist)

                # 港口费: d(v,j)
                port = self.ports_dict.get(j_port)
                if port:
                    per_sailing_cost += self.cost_calc.port_call_cost(v, port)

                # 运河费: a(v,i,j)
                is_panama, is_suez = self.canal_info.get((i_port, j_port), (False, False))
                per_sailing_cost += self.cost_calc.canal_cost(v, is_panama, is_suez)

            rot_params[rot.id] = {
                "round_trip_days": round_trip_days,
                "m_r": m_r,
                "per_sailing_cost": per_sailing_cost,
                "vessel": v,
            }

        # ========== 决策变量 ==========

        # Y_r: 分配给航线 r 的船舶数 (整数, ≥ 0)
        Y = {}
        for rot in R:
            Y[rot.id] = pulp.LpVariable(f"Y_{rot.id}", lowBound=0, cat="Integer")

        # O_od: 拒运量 (连续, ≥ 0)
        O = {}
        for o, d, k_od, _ in G:
            O[(o, d)] = pulp.LpVariable(f"O_{o}_{d}", lowBound=0)

        # V_r_od: 在航线 r 首次装载的 (o,d) 需求量 (连续, ≥ 0)
        V = {}
        for rot in R:
            for o, d, _, _ in G:
                if o in rot.port_calls:
                    V[(rot.id, o, d)] = pulp.LpVariable(
                        f"V_{rot.id}_{o}_{d}", lowBound=0
                    )

        # X_r(ij)d: 在航线 r 的边 (i,j) 上运往目的地 d 的集装箱数 (连续, ≥ 0)
        X = {}
        for rot in R:
            edges = rot.edges
            for i, j in edges:
                for _, d, _, _ in G:
                    X[(rot.id, i, j, d)] = pulp.LpVariable(
                        f"X_{rot.id}_{i}_{j}_{d}", lowBound=0
                    )

        # W_r(ij): 到达最终目的地的流量 (由 X 决定, 约束 4)
        # 作为辅助变量
        W = {}
        for rot in R:
            edges = rot.edges
            for i, j in edges:
                W[(rot.id, i, j)] = pulp.LpVariable(
                    f"W_{rot.id}_{i}_{j}", lowBound=0
                )

        # U_rs(hi)d: 从航线 r 在端口 i (经边 h->i) 转运到航线 s 的量
        U = {}
        for r_rot in R:
            for s_rot in R:
                if r_rot.id == s_rot.id:
                    continue
                # 找到共同港口
                common_ports = set(r_rot.port_calls) & set(s_rot.port_calls)
                for port in common_ports:
                    for _, d, _, _ in G:
                        if d == port:
                            continue
                        # r 的边 (h,i) 其中 i = port
                        for h, i in r_rot.edges:
                            if i == port:
                                U[(r_rot.id, s_rot.id, h, port, d)] = pulp.LpVariable(
                                    f"U_{r_rot.id}_{s_rot.id}_{h}_{port}_{d}",
                                    lowBound=0,
                                )

        # ========== 目标函数 ==========
        obj = pulp.LpAffineExpression()

        # (2a) TC 费用
        for rot in R:
            v = rot_params[rot.id]["vessel"]
            f_v = v.tc_rate * self.config.planning_horizon
            obj += f_v * Y[rot.id]

        # (2a) 未使用船舶的 charter out (收入，从目标函数中减去)
        for v_name, v in self.vessels.items():
            f_tilde = v.tc_rate * self.config.planning_horizon
            z_v = v.quantity
            deployed = pulp.lpSum(
                Y[rot.id] for rot in R if rot.vessel_class.name == v_name
            )
            obj -= f_tilde * (z_v - deployed)

        # (2b) 运营成本: m_r * Y_r * per_sailing_cost
        for rot in R:
            params = rot_params[rot.id]
            obj += params["m_r"] * params["per_sailing_cost"] * Y[rot.id]

        # (2c) 收入与拒运惩罚
        for o, d, k_od, q_od in G:
            # 拒运惩罚
            obj += self.config.reject_penalty * O[(o, d)]
            # 减去运输收入 (最小化目标中减去收入)
            for rot in R:
                if (rot.id, o, d) in V:
                    obj -= q_od * V[(rot.id, o, d)]

        # (2d) 装卸费和转运费
        # 装载费 (在起点装载)
        for rot in R:
            for o, d, _, _ in G:
                if (rot.id, o, d) in V:
                    port = self.ports_dict.get(o)
                    if port:
                        obj += port.move_cost * V[(rot.id, o, d)]

        # 卸货费 (在目的地卸货) — W_r(id)
        for rot in R:
            for i, j in rot.edges:
                port = self.ports_dict.get(j)
                if port:
                    obj += port.move_cost * W[(rot.id, i, j)]

        # 转运费
        for key, u_var in U.items():
            r_id, s_id, h, port, d = key
            port_obj = self.ports_dict.get(port)
            if port_obj:
                obj += port_obj.transship_cost * u_var

        model += obj, "Total_Cost"

        # ========== 约束 ==========

        # 约束 (5): O_od + sum_r V_r_od = k_od
        for o, d, k_od, _ in G:
            model += (
                O[(o, d)] + pulp.lpSum(
                    V[(rot.id, o, d)] for rot in R if (rot.id, o, d) in V
                ) == k_od,
                f"demand_{o}_{d}",
            )

        # 约束 (3): 流量守恒 (简化版 — 非蝴蝶航线)
        # 在非目的港 i 处: 进入流量 + 新装载 + 转入 = 流出流量 + 转出
        seen_flow = set()
        for rot in R:
            triples = rot.triples
            for t_idx, (h, i, j) in enumerate(triples):
                for _, d, _, _ in G:
                    if i == d:
                        continue
                    fkey = (rot.id, t_idx, d)
                    if fkey in seen_flow:
                        continue
                    seen_flow.add(fkey)

                    lhs = pulp.LpAffineExpression()
                    if (rot.id, h, i, d) in X:
                        lhs += X[(rot.id, h, i, d)]
                    if (rot.id, i, d) in V:
                        lhs += V[(rot.id, i, d)]
                    for s_rot in R:
                        if s_rot.id == rot.id:
                            continue
                        for k_port, i_port in s_rot.edges:
                            if i_port == i:
                                key = (s_rot.id, rot.id, k_port, i, d)
                                if key in U:
                                    lhs += U[key]

                    rhs = pulp.LpAffineExpression()
                    if (rot.id, i, j, d) in X:
                        rhs += X[(rot.id, i, j, d)]
                    for s_rot in R:
                        if s_rot.id == rot.id:
                            continue
                        key = (rot.id, s_rot.id, h, i, d)
                        if key in U:
                            rhs += U[key]

                    model += (
                        lhs == rhs,
                        f"flow_{rot.id}_t{t_idx}_{d}",
                    )

        # 约束 (4): 到达目的地 — X_r(ij)j = W_r(ij) for all (i,j) in E_r
        for rot in R:
            for i, j in rot.edges:
                if (rot.id, i, j, j) in X:
                    model += (
                        X[(rot.id, i, j, j)] == W[(rot.id, i, j)],
                        f"arrival_{rot.id}_{i}_{j}",
                    )
                else:
                    model += (
                        W[(rot.id, i, j)] == 0,
                        f"arrival_zero_{rot.id}_{i}_{j}",
                    )

        # 约束 (6): 边容量 — sum_d X_r(ij)d <= c_vr * m_r * Y_r
        for rot in R:
            params = rot_params[rot.id]
            for i, j in rot.edges:
                model += (
                    pulp.lpSum(
                        X[(rot.id, i, j, d)]
                        for _, d, _, _ in G
                        if (rot.id, i, j, d) in X
                    ) <= params["vessel"].capacity * params["m_r"] * Y[rot.id],
                    f"cap_{rot.id}_{i}_{j}",
                )

        # 约束 (7): 船舶数量 — sum_{r: vr=v} Y_r <= z_v
        for v_name, v in self.vessels.items():
            model += (
                pulp.lpSum(
                    Y[rot.id] for rot in R if rot.vessel_class.name == v_name
                ) <= v.quantity,
                f"fleet_{v_name}",
            )

        self.model = model
        self._Y = Y
        self._O = O
        self._V = V
        self._X = X
        self._W = W
        self._U = U
        self._rot_params = rot_params

    def solve(self, time_limit: int = 300, msg: bool = True) -> MIPResult:
        """
        求解 MIP 模型。

        参数:
        - time_limit: 最大求解时间 (秒)
        - msg: 是否输出求解日志

        返回:
        - MIPResult
        """
        if self.model is None:
            self.build()

        solver, selection = build_pulp_solver(
            requested=self.solver_backend,
            time_limit=time_limit,
            msg=msg,
        )
        self.model.solve(solver)

        status = pulp.LpStatus[self.model.status]
        objective = pulp.value(self.model.objective) if self.model.status == 1 else float("inf")

        # 提取结果
        vessel_assignments = {}
        for rot in self.rotations:
            val = pulp.value(self._Y[rot.id])
            vessel_assignments[rot.id] = int(round(val)) if val else 0

        rejected = {}
        transported = {}
        for o, d, k_od, _ in self.demands:
            o_val = pulp.value(self._O[(o, d)])
            rejected[(o, d)] = o_val if o_val else 0.0
            transported[(o, d)] = k_od - (o_val if o_val else 0.0)

        flow = {}
        for key, var in self._X.items():
            val = pulp.value(var)
            if val and val > 0.001:
                flow[key] = val

        # 计算成本分解
        breakdown = self._compute_breakdown(vessel_assignments, rejected, transported)

        result = MIPResult(
            status=status,
            objective=objective,
            cost_breakdown=breakdown,
            vessel_assignments=vessel_assignments,
            flow_on_edges=flow,
            rejected_demand=rejected,
            transported_demand=transported,
            active_backend=selection.active,
        )
        self.result = result
        return result

    def _compute_breakdown(self, assignments, rejected, transported) -> CostBreakdown:
        """计算成本分解，对应论文 Table 9。"""
        bd = CostBreakdown()

        # c_v: TC 费用 (部署的船舶)
        for rot in self.rotations:
            n = assignments.get(rot.id, 0)
            if n > 0:
                bd.c_v += rot.vessel_class.tc_rate * self.config.planning_horizon * n

        # L_v: 未使用船舶 (charter out revenue)
        vessel_used = {}
        for rot in self.rotations:
            vn = rot.vessel_class.name
            vessel_used[vn] = vessel_used.get(vn, 0) + assignments.get(rot.id, 0)
        for v_name, v in self.vessels.items():
            unused = v.quantity - vessel_used.get(v_name, 0)
            if unused > 0:
                bd.L_v += v.tc_rate * self.config.planning_horizon * unused

        # c_b, c_p, c_c: 运营成本
        for rot in self.rotations:
            n = assignments.get(rot.id, 0)
            if n <= 0:
                continue
            params = self._rot_params[rot.id]
            m_r = params["m_r"]
            v = rot.vessel_class
            s = rot.speed

            for k in range(len(rot.port_calls)):
                i_p = rot.port_calls[k]
                j_p = rot.port_calls[(k + 1) % len(rot.port_calls)]
                dist = self.dist_min.get((i_p, j_p), 0.0)

                bd.c_b += m_r * n * (
                    self.cost_calc.sailing_fuel_cost(v, s, dist) +
                    self.cost_calc.port_idle_fuel_cost(v)
                )

                port = self.ports_dict.get(j_p)
                if port:
                    bd.c_p += m_r * n * self.cost_calc.port_call_cost(v, port)

                is_panama, is_suez = self.canal_info.get((i_p, j_p), (False, False))
                bd.c_c += m_r * n * self.cost_calc.canal_cost(v, is_panama, is_suez)

        # Q: 收入, c_m: 装卸费
        for o, d, k_od, q_od in self.demands:
            t = transported.get((o, d), 0.0)
            bd.Q += q_od * t
            # 装卸费: 装载 + 卸货
            port_o = self.ports_dict.get(o)
            port_d = self.ports_dict.get(d)
            if port_o:
                bd.c_m += port_o.move_cost * t
            if port_d:
                bd.c_m += port_d.move_cost * t

        # c_t: 转运费
        for key, var in self._U.items():
            val = pulp.value(var)
            if val and val > 0.001:
                _, _, _, port_code, _ = key
                port = self.ports_dict.get(port_code)
                if port:
                    bd.c_t += port.transship_cost * val

        # F: 被拒绝运量
        bd.F = sum(rejected.values())

        # Z: 总目标值 = costs - revenue
        bd.Z = bd.c_v + bd.c_b + bd.c_p + bd.c_c + bd.c_m + bd.c_t - bd.Q - bd.L_v + self.config.reject_penalty * bd.F

        return bd
