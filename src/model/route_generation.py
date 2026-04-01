"""
路径生成模型 AUX(v, s, η) — 实现论文 §6.3 公式 (11)-(38)。

给定船型 v、速度 s 和部署船舶数 η，生成一条使净收入最大化的航线。
这是元启发式中"启发式列生成"的核心组件。

参考: Brouer et al. (2014), Transportation Science 48(2), pp. 301-303
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pulp

from src.utils.cost_calculator import VesselClass, Port, CostCalculator, CostConfig
from src.utils.network_builder import Rotation


@dataclass
class AUXConfig:
    """AUX 模型配置参数。"""
    max_ports: int = 20              # 子问题最大港口数
    alpha: float = 1.5               # β: 蝴蝶节点额外流量系数
    suppress_subtour: bool = True     # 是否启用子环消除 (前10轮可关闭)
    time_limit: int = 60             # 单个 AUX 求解时限 (秒)


class RouteGenerator:
    """
    路径生成器，实现 AUX(v, s, η) 模型。

    对给定的船型、速度和船舶数，通过 MIP 求解生成最优航线。
    """

    def __init__(
        self,
        ports: List[Port],
        port_codes: List[str],
        vessel: VesselClass,
        residual_demand: Dict[Tuple[str, str], float],
        demand_revenue: Dict[Tuple[str, str], float],
        dist_min: Dict[Tuple[str, str], float],
        canal_info: Dict[Tuple[str, str], Tuple[bool, bool]],
        ports_dict: Dict[str, Port],
        config: CostConfig = None,
        aux_config: AUXConfig = None,
    ):
        self.ports = ports
        self.port_codes = port_codes
        self.vessel = vessel
        self.residual_demand = residual_demand
        self.demand_revenue = demand_revenue
        self.dist_min = dist_min
        self.canal_info = canal_info
        self.ports_dict = ports_dict
        self.config = config or CostConfig()
        self.aux_config = aux_config or AUXConfig()
        self.cost_calc = CostCalculator(self.config)

    def solve(self, speed: float, num_vessels: int,
              rotation_id: str = "new") -> Optional[Rotation]:
        """
        求解 AUX(v, s, η) 模型。

        参数:
        - speed: 航行速度
        - num_vessels: 部署船舶数 η
        - rotation_id: 生成的航线 ID

        返回:
        - Rotation 对象 (如果找到可行解), 否则 None
        """
        v = self.vessel
        P = self.port_codes
        n_ports = len(P)
        e = self.config.fuel_price
        T = self.config.planning_horizon

        # 需求集合 G: 仅保留有残差需求的 OD 对
        G = [(o, d) for (o, d), k in self.residual_demand.items()
             if k > 0 and o in P and d in P]

        if not G:
            return None

        # ========== 构建 MIP ==========
        model = pulp.LpProblem(f"AUX_{rotation_id}", pulp.LpMaximize)

        # --- 决策变量 ---

        # N_j: 港口 j 是否在航线中 (二进制)
        N = {j: pulp.LpVariable(f"N_{j}", cat="Binary") for j in P}

        # B_j: 港口 j 是否为蝴蝶节点 (二进制)
        B = {j: pulp.LpVariable(f"B_{j}", cat="Binary") for j in P}

        # C_j: 港口 j 是否为主港口 (二进制)
        C = {j: pulp.LpVariable(f"C_{j}", cat="Binary") for j in P}

        # A_ij: 边 (i,j) 是否在航线中 (二进制)
        A = {}
        for i in P:
            for j in P:
                if i != j:
                    A[(i, j)] = pulp.LpVariable(f"A_{i}_{j}", cat="Binary")

        # Q_od: 每次航行运载的 FFE 数 (连续, ≥ 0)
        Q = {}
        for o, d in G:
            Q[(o, d)] = pulp.LpVariable(f"Q_{o}_{d}", lowBound=0)

        # I_j: 序列号 (连续, ≥ 0), 用于消除子环
        I = {j: pulp.LpVariable(f"I_{j}", lowBound=0) for j in P}

        # W1, W2: 周频/双周频 (二进制)
        W1 = pulp.LpVariable("W1", cat="Binary")
        W2 = pulp.LpVariable("W2", cat="Binary")

        # τ: 每次航行时间的倒数 (连续, ≥ 0)
        tau = pulp.LpVariable("tau", lowBound=0)

        # Φ: 每次航行的估计净收入 (连续)
        phi = pulp.LpVariable("phi")

        # --- 目标函数 (11): maximize Φ ---
        model += phi, "maximize_revenue"

        # --- 约束 (12): 定义 Φ ---
        # Φ = (f_tilde - f_v) * τ
        #     - sum_{(i,j)} (e*h*p + e*g*l + d_j + a_ij) * A_ij
        #     + sum_{(o,d)} (q_od - u_o - u_d + q_tilde) * Q_od

        f_v = v.tc_rate * T
        f_tilde = v.tc_rate * T  # 论文: f_v = f_tilde (deployed = charter out value)

        # 运营成本项
        op_cost = pulp.LpAffineExpression()
        for i in P:
            for j in P:
                if i == j:
                    continue
                if (i, j) not in A:
                    continue
                dist = self.dist_min.get((i, j), 0.0)
                if dist == 0.0:
                    continue

                # e * h(v) * p(v,j): 停泊燃油
                idle_cost = e * v.fuel_idle * (self.config.port_stay_hours / 24.0)
                # e * g(v,s) * l(v,i,j): 航行燃油
                fuel_per_mile = v.fuel_consumption_per_mile(speed)
                sail_cost = e * fuel_per_mile * dist
                # d(v,j): 港口费
                port_j = self.ports_dict.get(j)
                port_cost = 0.0
                if port_j:
                    port_cost = port_j.fixed_call_cost + port_j.var_call_cost * v.capacity
                # a(v,i,j): 运河费
                is_panama, is_suez = self.canal_info.get((i, j), (False, False))
                canal_cost = self.cost_calc.canal_cost(v, is_panama, is_suez)

                leg_cost = idle_cost + sail_cost + port_cost + canal_cost
                op_cost += leg_cost * A[(i, j)]

        # 收入项
        revenue = pulp.LpAffineExpression()
        for o, d in G:
            q_od = self.demand_revenue.get((o, d), 0.0)
            port_o = self.ports_dict.get(o)
            port_d = self.ports_dict.get(d)
            u_o = port_o.move_cost if port_o else 0.0
            u_d = port_d.move_cost if port_d else 0.0
            net_rev = q_od - u_o - u_d + self.config.reject_penalty
            revenue += net_rev * Q[(o, d)]

        # Φ = (f_tilde - f_v) * τ - op_cost + revenue
        # 注意: f_tilde - f_v = 0 (论文假设), 所以第一项为 0
        model += (
            phi == -op_cost + revenue,
            "define_phi",
        )

        # --- 约束 (13): 航行时间 ---
        # 24 * T * τ = sum_{(i,j)} (p_j + l_ij/s) * A_ij
        time_expr = pulp.LpAffineExpression()
        for i in P:
            for j in P:
                if i == j or (i, j) not in A:
                    continue
                dist = self.dist_min.get((i, j), 0.0)
                sailing_hours = dist / speed if speed > 0 else 0.0
                leg_time = self.config.port_stay_hours + sailing_hours
                time_expr += leg_time * A[(i, j)]

        model += (24.0 * T * tau == time_expr, "trip_time")

        # --- 约束 (14): 流量平衡 ---
        for n_port in P:
            model += (
                pulp.lpSum(A[(j, n_port)] for j in P if j != n_port and (j, n_port) in A)
                ==
                pulp.lpSum(A[(n_port, j)] for j in P if j != n_port and (n_port, j) in A),
                f"balance_{n_port}",
            )

        # --- 约束 (15): 边激活 ---
        for i in P:
            for j in P:
                if i == j or (i, j) not in A:
                    continue
                model += (A[(i, j)] <= (N[i] + N[j]) / 2.0, f"edge_act_{i}_{j}")

        # --- 约束 (16): 最多一个蝴蝶节点 ---
        model += (pulp.lpSum(B[j] for j in P) <= 1, "max_butterfly")

        # --- 约束 (17)(18): 入/出边限制 ---
        for j in P:
            model += (
                pulp.lpSum(A[(i, j)] for i in P if i != j and (i, j) in A) <= N[j] + B[j],
                f"in_degree_{j}",
            )
            model += (
                pulp.lpSum(A[(j, i)] for i in P if i != j and (j, i) in A) <= N[j] + B[j],
                f"out_degree_{j}",
            )

        # --- 约束 (19)(20)(21): 货物量上界 ---
        for o, d in G:
            k_hat = self.residual_demand.get((o, d), 0.0)
            # (19): Q_od <= k_hat / η
            model += (Q[(o, d)] <= k_hat / num_vessels, f"demand_cap_{o}_{d}")
            # (20): Q_od <= k_hat * N_o
            model += (Q[(o, d)] <= k_hat * N[o], f"origin_active_{o}_{d}")
            # (21): Q_od <= k_hat * N_d
            model += (Q[(o, d)] <= k_hat * N[d], f"dest_active_{o}_{d}")

        # --- 约束 (22)(23): 装卸比例 ---
        # 论文 φ^out / φ^in: 每港口单次航行最大装卸比例
        # 使用船舶容量作为上界（论文中这些参数用于加速求解，非严格约束）
        for port_code in P:
            # (22): sum_d Q_od <= c_v * (N_o + B_o)  (单次航行装载不超过船舶容量)
            out_demand = [Q[(o, d)] for o, d in G if o == port_code]
            if out_demand:
                model += (
                    pulp.lpSum(out_demand) <= v.capacity * (N[port_code] + B[port_code]),
                    f"out_cap_{port_code}",
                )

            # (23): sum_o Q_od <= c_v * (N_d + B_d)  (单次航行卸载不超过船舶容量)
            in_demand = [Q[(o, d)] for o, d in G if d == port_code]
            if in_demand:
                model += (
                    pulp.lpSum(in_demand) <= v.capacity * (N[port_code] + B[port_code]),
                    f"in_cap_{port_code}",
                )

        # --- 约束 (24): FFE-miles 平衡 ---
        # sum_{(o,d)} l_od * Q_od <= sum_{(i,j)} A_ij * l_ij * c_v
        lhs_miles = pulp.LpAffineExpression()
        for o, d in G:
            dist_od = self.dist_min.get((o, d), 0.0)
            lhs_miles += dist_od * Q[(o, d)]

        rhs_miles = pulp.LpAffineExpression()
        for i in P:
            for j in P:
                if i == j or (i, j) not in A:
                    continue
                dist = self.dist_min.get((i, j), 0.0)
                rhs_miles += A[(i, j)] * dist * v.capacity

        model += (lhs_miles <= rhs_miles, "ffe_miles")

        # --- 约束 (25)(26)(27)(28): 子环消除 ---
        # (25): 恰好一个主港口
        model += (pulp.lpSum(C[j] for j in P) == 1, "one_master")

        # (26): B_j <= C_j <= N_j
        for j in P:
            model += (B[j] <= C[j], f"butterfly_master_{j}")
            model += (C[j] <= N[j], f"master_active_{j}")

        # (27): N_j <= I_j <= |P| * N_j
        for j in P:
            model += (N[j] <= I[j], f"seq_lb_{j}")
            model += (I[j] <= n_ports * N[j], f"seq_ub_{j}")

        # (28): 子环消除 (MTZ 格式)
        if self.aux_config.suppress_subtour:
            for i in P:
                for j in P:
                    if i == j or (i, j) not in A:
                        continue
                    model += (
                        1 + I[i] - n_ports * C[j] - n_ports * (1 - A[(i, j)]) <= I[j],
                        f"subtour_{i}_{j}",
                    )

        # --- 约束 (29)(30): 频率 ---
        # (29): W1 + W2 = 1
        model += (W1 + W2 == 1, "freq_choice")

        # (30): 大船 (>=1200 FFE) 必须周频
        if v.capacity >= 1200:
            model += (W2 == 0, "weekly_required")

        # --- 约束 (31)(32): 频率与时间关系 ---
        # κ = num_vessels (论文中的 η)
        kappa = num_vessels
        # (31): 周频: (W1-1) + 0.91 * 7κ/T <= τ <= 7κ/T + (1-W1)
        model += (
            (W1 - 1) + 0.91 * 7.0 * kappa / T <= tau,
            "weekly_lb",
        )
        model += (
            tau <= 7.0 * kappa / T + (1 - W1),
            "weekly_ub",
        )

        # (32): 双周频: (W2-1) + 0.91 * 14κ/T <= τ <= 14κ/T + (1-W2)
        model += (
            (W2 - 1) + 0.91 * 14.0 * kappa / T <= tau,
            "biweekly_lb",
        )
        model += (
            tau <= 14.0 * kappa / T + (1 - W2),
            "biweekly_ub",
        )

        # ========== 求解 ==========
        solver = pulp.COIN_CMD(
            path="/opt/homebrew/bin/cbc",
            timeLimit=self.aux_config.time_limit, msg=False
        )
        model.solve(solver)

        if model.status != 1:
            return None

        # ========== 提取结果 ==========
        # 提取被选中的港口
        active_ports = [j for j in P if pulp.value(N[j]) and pulp.value(N[j]) > 0.5]
        if len(active_ports) < 2:
            return None

        # 提取被选中的边
        active_edges = []
        for i in P:
            for j in P:
                if i != j and (i, j) in A:
                    val = pulp.value(A[(i, j)])
                    if val and val > 0.5:
                        active_edges.append((i, j))

        if not active_edges:
            return None

        # 从边重建港口调用序列
        port_sequence = self._reconstruct_sequence(active_edges, active_ports)
        if not port_sequence:
            return None

        # 检测蝴蝶节点
        butterfly_port = None
        for j in P:
            b_val = pulp.value(B[j])
            if b_val and b_val > 0.5:
                butterfly_port = j
                break

        # 计算实际需要的船舶数
        round_trip_days = self.cost_calc.rotation_round_trip_time(
            v, speed, port_sequence, self.dist_min
        )

        return Rotation(
            id=rotation_id,
            vessel_class=v,
            speed=speed,
            num_vessels=num_vessels,
            port_calls=port_sequence,
            is_butterfly=butterfly_port is not None,
            butterfly_port=butterfly_port,
        )

    def _reconstruct_sequence(self, edges: List[Tuple[str, str]],
                               ports: List[str]) -> Optional[List[str]]:
        """从边列表重建港口调用序列。"""
        if not edges:
            return None

        adj = {}
        for i, j in edges:
            if i not in adj:
                adj[i] = []
            adj[i].append(j)

        # 从第一条边开始
        start = edges[0][0]
        sequence = [start]
        visited_edges = set()
        current = start

        for _ in range(len(edges) + 1):
            nexts = adj.get(current, [])
            moved = False
            for n in nexts:
                if (current, n) not in visited_edges:
                    visited_edges.add((current, n))
                    sequence.append(n)
                    current = n
                    moved = True
                    break
            if not moved:
                break

        # 如果最后一个等于第一个，移除最后一个 (循环)
        if len(sequence) > 1 and sequence[-1] == sequence[0]:
            sequence = sequence[:-1]

        return sequence if len(sequence) >= 2 else None

    def generate_candidates(
        self,
        speed_range: Optional[List[float]] = None,
        vessel_counts: Optional[List[int]] = None,
        prefix: str = "rot",
    ) -> List[Rotation]:
        """
        生成多条候选航线。

        对不同速度和船舶数组合调用 AUX 模型。

        参数:
        - speed_range: 候选速度列表 (默认从 min_speed 到 max_speed, 步长 1)
        - vessel_counts: 候选船舶数列表 (默认 [1, 2, 3])
        - prefix: 航线 ID 前缀

        返回:
        - 候选航线列表
        """
        v = self.vessel
        if speed_range is None:
            speed_range = list(range(int(v.min_speed), int(v.max_speed) + 1))
        if vessel_counts is None:
            vessel_counts = [1, 2, 3]

        candidates = []
        idx = 0
        for s in speed_range:
            for eta in vessel_counts:
                rot_id = f"{prefix}_{v.name}_{s}kn_{eta}v_{idx}"
                result = self.solve(speed=float(s), num_vessels=eta, rotation_id=rot_id)
                if result is not None:
                    candidates.append(result)
                    idx += 1

        return candidates


def create_port_clusters(
    ports: List[str],
    demands: Dict[Tuple[str, str], float],
    dist_min: Dict[Tuple[str, str], float],
    max_cluster_size: int = 20,
) -> List[List[str]]:
    """
    将港口按贸易量聚类，用于限制 AUX 子问题规模。

    论文 §6.3: "create clusters of ports that are tightly linked both
    geographically and by trade volumes"
    """
    # 计算每个港口的总贸易量
    port_trade = {}
    for (o, d), k in demands.items():
        port_trade[o] = port_trade.get(o, 0.0) + k
        port_trade[d] = port_trade.get(d, 0.0) + k

    # 按贸易量降序排序
    sorted_ports = sorted(ports, key=lambda p: port_trade.get(p, 0.0), reverse=True)

    # 简单贪心聚类: 从最大贸易量港口开始，加入距离最近的港口
    used = set()
    clusters = []

    for seed in sorted_ports:
        if seed in used:
            continue
        cluster = [seed]
        used.add(seed)

        # 找到与种子港口有最大贸易往来的港口
        neighbors = []
        for (o, d), k in demands.items():
            if o == seed and d not in used:
                neighbors.append((d, k))
            elif d == seed and o not in used:
                neighbors.append((o, k))

        neighbors.sort(key=lambda x: x[1], reverse=True)

        for port, _ in neighbors:
            if len(cluster) >= max_cluster_size:
                break
            if port not in used:
                cluster.append(port)
                used.add(port)

        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters
