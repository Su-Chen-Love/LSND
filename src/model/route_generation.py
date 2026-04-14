"""
Route generation AUX(v, s, eta) for the liner shipping metaheuristic.

This module now focuses on two practical goals:
1. generate multiple integer candidate routes per AUX solve via no-good cuts
2. extract/validate butterfly routes from active edges deterministically
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pulp

from src.utils.cost_calculator import CostCalculator, CostConfig, Port, VesselClass
from src.utils.network_builder import Rotation
from src.utils.solver_backend import SolverBackend, build_pulp_solver


@dataclass
class AUXConfig:
    """AUX model configuration."""

    max_ports: int = 20
    alpha: float = 1.5
    suppress_subtour: bool = True
    time_limit: int = 60
    solver_backend: SolverBackend = "auto"
    num_solutions: int = 5


class RouteGenerator:
    """
    Generate candidate routes by solving AUX(v, s, eta).
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

    def solve(self, speed: float, num_vessels: int, rotation_id: str = "new") -> Optional[Rotation]:
        """Return the best feasible route for compatibility with existing callers."""
        solutions = self.solve_many(speed=speed, num_vessels=num_vessels, rotation_prefix=rotation_id, limit=1)
        return solutions[0] if solutions else None

    def solve_many(
        self,
        speed: float,
        num_vessels: int,
        rotation_prefix: str = "new",
        limit: Optional[int] = None,
    ) -> List[Rotation]:
        """Solve AUX and enumerate multiple integer candidate routes via no-good cuts."""
        limit = limit or self.aux_config.num_solutions
        model_data = self._build_model(speed=speed, num_vessels=num_vessels, model_name=rotation_prefix)
        if model_data is None:
            return []

        model, A, B, N = model_data
        solver, _ = build_pulp_solver(
            requested=self.aux_config.solver_backend,
            time_limit=self.aux_config.time_limit,
            msg=False,
        )

        seen_signatures = set()
        routes: List[Rotation] = []

        for solution_idx in range(limit):
            model.solve(solver)
            if model.status != 1:
                break

            active_edges = [
                (i, j)
                for (i, j), var in A.items()
                if (pulp.value(var) or 0.0) > 0.5
            ]
            if not active_edges:
                break

            extracted = self.extract_route_from_active_edges(
                active_edges=active_edges,
                fallback_butterfly={
                    port_code for port_code, var in B.items()
                    if (pulp.value(var) or 0.0) > 0.5
                },
            )

            if extracted is not None:
                port_sequence, butterfly_port = extracted
                canonical_calls = self.canonicalize_port_calls(port_sequence)
                signature = (
                    self.vessel.name,
                    round(speed, 4),
                    num_vessels,
                    canonical_calls,
                    butterfly_port,
                )
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    routes.append(
                        Rotation(
                            id=f"{rotation_prefix}_{solution_idx}",
                            vessel_class=self.vessel,
                            speed=speed,
                            num_vessels=num_vessels,
                            port_calls=list(port_sequence),
                            is_butterfly=butterfly_port is not None,
                            butterfly_port=butterfly_port,
                        )
                    )

            active_vars = [A[edge] for edge in active_edges]
            model += (
                pulp.lpSum(active_vars) <= len(active_edges) - 1,
                f"nogood_{rotation_prefix}_{solution_idx}",
            )

        return routes

    def _build_model(
        self,
        speed: float,
        num_vessels: int,
        model_name: str,
    ) -> Optional[Tuple[pulp.LpProblem, Dict[Tuple[str, str], pulp.LpVariable], Dict[str, pulp.LpVariable], Dict[str, pulp.LpVariable]]]:
        v = self.vessel
        P = self.port_codes
        n_ports = len(P)
        e = self.config.fuel_price
        T = self.config.planning_horizon

        G = [(o, d) for (o, d), k in self.residual_demand.items() if k > 0 and o in P and d in P]
        if not G:
            return None

        model = pulp.LpProblem(f"AUX_{model_name}", pulp.LpMaximize)

        N = {j: pulp.LpVariable(f"N_{j}", cat="Binary") for j in P}
        B = {j: pulp.LpVariable(f"B_{j}", cat="Binary") for j in P}
        C = {j: pulp.LpVariable(f"C_{j}", cat="Binary") for j in P}
        A = {(i, j): pulp.LpVariable(f"A_{i}_{j}", cat="Binary") for i in P for j in P if i != j}
        Q = {(o, d): pulp.LpVariable(f"Q_{o}_{d}", lowBound=0) for o, d in G}
        I = {j: pulp.LpVariable(f"I_{j}", lowBound=0) for j in P}
        W1 = pulp.LpVariable("W1", cat="Binary")
        W2 = pulp.LpVariable("W2", cat="Binary")
        tau = pulp.LpVariable("tau", lowBound=0)
        phi = pulp.LpVariable("phi")

        model += phi, "maximize_revenue"

        op_cost = pulp.LpAffineExpression()
        for (i, j), var in A.items():
            dist = self.dist_min.get((i, j), 0.0)
            if dist <= 0.0:
                model += (var == 0, f"invalid_edge_{i}_{j}")
                continue

            idle_cost = e * v.fuel_idle * (self.config.port_stay_hours / 24.0)
            sail_cost = e * v.fuel_consumption_per_mile(speed) * dist
            port_j = self.ports_dict.get(j)
            port_cost = 0.0
            if port_j:
                port_cost = port_j.fixed_call_cost + port_j.var_call_cost * v.capacity
            is_panama, is_suez = self.canal_info.get((i, j), (False, False))
            canal_cost = self.cost_calc.canal_cost(v, is_panama, is_suez)
            op_cost += (idle_cost + sail_cost + port_cost + canal_cost) * var

        revenue = pulp.LpAffineExpression()
        for o, d in G:
            q_od = self.demand_revenue.get((o, d), 0.0)
            port_o = self.ports_dict.get(o)
            port_d = self.ports_dict.get(d)
            u_o = port_o.move_cost if port_o else 0.0
            u_d = port_d.move_cost if port_d else 0.0
            revenue += (q_od - u_o - u_d + self.config.reject_penalty) * Q[(o, d)]

        model += (phi == -op_cost + revenue, "define_phi")

        time_expr = pulp.LpAffineExpression()
        for (i, j), var in A.items():
            dist = self.dist_min.get((i, j), 0.0)
            if dist <= 0.0:
                continue
            time_expr += (self.config.port_stay_hours + dist / max(speed, 1.0)) * var
        model += (24.0 * T * tau == time_expr, "trip_time")

        for n_port in P:
            model += (
                pulp.lpSum(A[(j, n_port)] for j in P if j != n_port) ==
                pulp.lpSum(A[(n_port, j)] for j in P if j != n_port),
                f"balance_{n_port}",
            )

        for (i, j), var in A.items():
            model += (var <= (N[i] + N[j]) / 2.0, f"edge_act_{i}_{j}")

        model += (pulp.lpSum(B[j] for j in P) <= 1, "max_butterfly")

        for j in P:
            model += (
                pulp.lpSum(A[(i, j)] for i in P if i != j) <= N[j] + B[j],
                f"in_degree_{j}",
            )
            model += (
                pulp.lpSum(A[(j, i)] for i in P if i != j) <= N[j] + B[j],
                f"out_degree_{j}",
            )

        for o, d in G:
            k_hat = self.residual_demand.get((o, d), 0.0)
            model += (Q[(o, d)] <= tau * k_hat / num_vessels, f"demand_cap_{o}_{d}")
            model += (Q[(o, d)] <= k_hat * N[o], f"origin_active_{o}_{d}")
            model += (Q[(o, d)] <= k_hat * N[d], f"dest_active_{o}_{d}")

        delta = self.aux_config.alpha
        total_out = defaultdict(float)
        total_in = defaultdict(float)
        for (o, d), k in self.residual_demand.items():
            if k > 0 and o in P and d in P:
                total_out[o] += k
                total_in[d] += k
        sum_out = sum(total_out.values()) or 1.0
        sum_in = sum(total_in.values()) or 1.0
        phi_out = {p: total_out[p] / sum_out for p in P}
        phi_in = {p: total_in[p] / sum_in for p in P}

        for port_code in P:
            out_demand = [Q[(o, d)] for o, d in G if o == port_code]
            if out_demand:
                model += (
                    pulp.lpSum(out_demand) <= phi_out[port_code] * v.capacity * (N[port_code] + delta * B[port_code]),
                    f"out_cap_{port_code}",
                )
            in_demand = [Q[(o, d)] for o, d in G if d == port_code]
            if in_demand:
                model += (
                    pulp.lpSum(in_demand) <= phi_in[port_code] * v.capacity * (N[port_code] + delta * B[port_code]),
                    f"in_cap_{port_code}",
                )

        lhs_miles = pulp.lpSum(self.dist_min.get((o, d), 0.0) * Q[(o, d)] for o, d in G)
        rhs_miles = pulp.lpSum(self.dist_min.get((i, j), 0.0) * v.capacity * A[(i, j)] for i, j in A)
        model += (lhs_miles <= rhs_miles, "ffe_miles")

        model += (pulp.lpSum(C[j] for j in P) == 1, "one_master")
        for j in P:
            model += (B[j] <= C[j], f"butterfly_master_{j}")
            model += (C[j] <= N[j], f"master_active_{j}")
            model += (N[j] <= I[j], f"seq_lb_{j}")
            model += (I[j] <= n_ports * N[j], f"seq_ub_{j}")

        if self.aux_config.suppress_subtour:
            for (i, j), var in A.items():
                model += (
                    1 + I[i] - n_ports * C[j] - n_ports * (1 - var) <= I[j],
                    f"subtour_{i}_{j}",
                )

        model += (W1 + W2 == 1, "freq_choice")
        if v.capacity >= 1200:
            model += (W2 == 0, "weekly_required")
        if v.capacity >= 4200:
            model += (tau >= 28.0 / T, "large_vessel_min_rotation")

        kappa = num_vessels
        model += ((W1 - 1) + 0.91 * 7.0 * kappa / T <= tau, "weekly_lb")
        model += (tau <= 7.0 * kappa / T + (1 - W1), "weekly_ub")
        model += ((W2 - 1) + 0.91 * 14.0 * kappa / T <= tau, "biweekly_lb")
        model += (tau <= 14.0 * kappa / T + (1 - W2), "biweekly_ub")

        return model, A, B, N

    @staticmethod
    def canonicalize_port_calls(port_calls: List[str]) -> Tuple[str, ...]:
        """Canonicalize a cyclic route to remove duplicate representations from different start ports."""
        if not port_calls:
            return tuple()
        n = len(port_calls)
        rotations = [tuple(port_calls[i:] + port_calls[:i]) for i in range(n)]
        return min(rotations)

    @staticmethod
    def extract_route_from_active_edges(
        active_edges: List[Tuple[str, str]],
        fallback_butterfly: Optional[set[str]] = None,
    ) -> Optional[Tuple[List[str], Optional[str]]]:
        """Build a route sequence from active edges and validate butterfly structure."""
        if not active_edges:
            return None

        out_degree = Counter(i for i, _ in active_edges)
        in_degree = Counter(j for _, j in active_edges)
        nodes = set(out_degree) | set(in_degree)

        if any(out_degree[n] != in_degree[n] for n in nodes):
            return None

        actual_butterflies = [n for n in nodes if out_degree[n] == 2 and in_degree[n] == 2]
        if len(actual_butterflies) > 1:
            return None
        butterfly_port = actual_butterflies[0] if actual_butterflies else None

        start = butterfly_port or min(nodes)
        adjacency = defaultdict(list)
        for edge_idx, (i, j) in enumerate(sorted(active_edges)):
            adjacency[i].append((j, edge_idx))

        circuit = RouteGenerator._hierholzer(adjacency, start=start)
        if circuit is None or len(circuit) != len(active_edges) + 1:
            return None

        used_edges = Counter(zip(circuit[:-1], circuit[1:]))
        if used_edges != Counter(active_edges):
            return None

        port_sequence = circuit[:-1]
        repeated_ports = {port for port, count in Counter(port_sequence).items() if count > 1}
        if butterfly_port:
            if repeated_ports - {butterfly_port}:
                return None
        elif repeated_ports:
            return None

        if butterfly_port:
            occurrences = [idx for idx, port in enumerate(port_sequence) if port == butterfly_port]
            if len(occurrences) != 2:
                return None
            first = occurrences[0]
            port_sequence = port_sequence[first:] + port_sequence[:first]
        else:
            min_idx = min(range(len(port_sequence)), key=lambda idx: tuple(port_sequence[idx:] + port_sequence[:idx]))
            port_sequence = port_sequence[min_idx:] + port_sequence[:min_idx]

        return port_sequence, butterfly_port

    @staticmethod
    def _hierholzer(adjacency: Dict[str, List[Tuple[str, int]]], start: str) -> Optional[List[str]]:
        stack = [start]
        path: List[str] = []
        local_adj = {node: list(reversed(edges)) for node, edges in adjacency.items()}

        while stack:
            node = stack[-1]
            if local_adj.get(node):
                next_node, _ = local_adj[node].pop()
                stack.append(next_node)
            else:
                path.append(stack.pop())
        path.reverse()
        return path if len(path) >= 2 else None

    def generate_candidates(
        self,
        speed_range: Optional[List[float]] = None,
        vessel_counts: Optional[List[int]] = None,
        prefix: str = "rot",
        limit_per_combination: Optional[int] = None,
    ) -> List[Rotation]:
        """Generate multiple candidates across speed and vessel-count combinations."""
        v = self.vessel
        if speed_range is None:
            speed_range = list(range(int(v.min_speed), int(v.max_speed) + 1))
        if vessel_counts is None:
            vessel_counts = [1, 2, 3]

        candidates = []
        seen = set()
        idx = 0
        for s in speed_range:
            for eta in vessel_counts:
                rot_id = f"{prefix}_{v.name}_{s}kn_{eta}v_{idx}"
                for result in self.solve_many(
                    speed=float(s),
                    num_vessels=eta,
                    rotation_prefix=rot_id,
                    limit=limit_per_combination,
                ):
                    signature = (
                        result.vessel_class.name,
                        result.speed,
                        result.num_vessels,
                        tuple(result.port_calls),
                        result.butterfly_port,
                    )
                    if signature not in seen:
                        seen.add(signature)
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
    Create coarse port clusters used to limit AUX problem size.
    """
    port_trade = defaultdict(float)
    for (o, d), k in demands.items():
        port_trade[o] += k
        port_trade[d] += k

    sorted_ports = sorted(ports, key=lambda p: port_trade.get(p, 0.0), reverse=True)
    used = set()
    clusters = []

    for seed in sorted_ports:
        if seed in used:
            continue
        cluster = [seed]
        used.add(seed)

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
