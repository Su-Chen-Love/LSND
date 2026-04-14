"""
Multicommodity flow problem for evaluating a fixed set of rotations.

This implementation follows the terminal-node + port-call-node graph
described in Brouer et al. (2014), Figure 7, instead of the earlier
non-butterfly approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pulp

from src.utils.cost_calculator import CostCalculator, CostConfig, Port
from src.utils.network_builder import Rotation, build_mcf_network
from src.utils.solver_backend import SolverBackend, build_pulp_solver


@dataclass
class MCFPResult:
    """MCFP solve result."""

    status: str
    objective: float
    total_revenue: float
    total_handling_cost: float
    total_transship_cost: float
    total_reject_penalty: float
    flow: Dict[str, Dict[Tuple[str, str], float]]
    leg_flow: Dict[str, Dict[Tuple[str, str], float]]
    rejected: Dict[Tuple[str, str], float]
    residual_demand: Dict[Tuple[str, str], float]
    active_backend: str = "cbc"


class MCFPSolver:
    """
    Evaluate a network of rotations by routing demand over the Figure 7 graph.
    """

    def __init__(
        self,
        rotations: List[Rotation],
        ports_dict: Dict[str, Port],
        demands: List[Tuple[str, str, float, float]],
        dist_min: Dict[Tuple[str, str], float],
        config: CostConfig = None,
        solver_backend: SolverBackend = "auto",
    ):
        self.rotations = rotations
        self.ports_dict = ports_dict
        self.demands = demands
        self.dist_min = dist_min
        self.config = config or CostConfig()
        self.cost_calc = CostCalculator(self.config)
        self.solver_backend = solver_backend

    def solve(
        self,
        vessel_assignments: Optional[Dict[str, int]] = None,
        time_limit: int = 120,
    ) -> MCFPResult:
        """
        Solve the multicommodity flow LP for a fixed set of rotations.
        """
        rotations = self._apply_assignments(vessel_assignments)
        network = build_mcf_network(
            rotations=rotations,
            ports_dict=self.ports_dict,
            demands=self.demands,
            distances=self.dist_min,
            cost_calc=self.cost_calc,
        )

        model = pulp.LpProblem("MCFP", pulp.LpMinimize)

        sail_edges = [e for e in network.edges if e.edge_type == "sail"]
        transship_edges = [e for e in network.edges if e.edge_type == "transship"]
        load_edges = [e for e in network.edges if e.edge_type == "load"]
        unload_edges = [e for e in network.edges if e.edge_type == "unload"]
        omit_edges = {e.edge_id: e for e in network.edges if e.edge_type == "omit"}

        revenue_by_od = {(o, d): q for o, d, _, q in self.demands}
        quantity_by_od = {(o, d): k for o, d, k, _ in self.demands}

        # Edge variables by commodity.
        X = {}
        U = {}
        V = {}
        W = {}
        O = {}

        node_incoming: Dict[Tuple[str, str, str], List[pulp.LpVariable]] = {}
        node_outgoing: Dict[Tuple[str, str, str], List[pulp.LpVariable]] = {}

        obj = pulp.LpAffineExpression()

        for o, d, _, _ in self.demands:
            source = network.terminal_nodes[o]
            sink = network.terminal_nodes[d]

            # Load edges are only available from the commodity origin terminal.
            for edge in load_edges:
                if edge.source != source:
                    continue
                var = pulp.LpVariable(f"V_{edge.edge_id}_{o}_{d}", lowBound=0)
                V[(edge.edge_id, o, d)] = var
                obj += (edge.cost - revenue_by_od[(o, d)]) * var
                self._append_arc(node_outgoing, node_incoming, edge.source, edge.target, o, d, var)

            # Sailing edges are commodity-specific across all call nodes.
            for edge in sail_edges:
                var = pulp.LpVariable(f"X_{edge.edge_id}_{o}_{d}", lowBound=0)
                X[(edge.edge_id, o, d)] = var
                self._append_arc(node_outgoing, node_incoming, edge.source, edge.target, o, d, var)

            # Transshipment edges are commodity-specific as well.
            for edge in transship_edges:
                var = pulp.LpVariable(f"U_{edge.edge_id}_{o}_{d}", lowBound=0)
                U[(edge.edge_id, o, d)] = var
                obj += edge.cost * var
                self._append_arc(node_outgoing, node_incoming, edge.source, edge.target, o, d, var)

            # Unload edges are only available into the commodity destination terminal.
            for edge in unload_edges:
                if edge.target != sink:
                    continue
                var = pulp.LpVariable(f"W_{edge.edge_id}_{o}_{d}", lowBound=0)
                W[(edge.edge_id, o, d)] = var
                obj += edge.cost * var
                self._append_arc(node_outgoing, node_incoming, edge.source, edge.target, o, d, var)

            omit_edge = omit_edges[f"omit:{o}:{d}"]
            omit_var = pulp.LpVariable(f"O_{o}_{d}", lowBound=0, upBound=quantity_by_od[(o, d)])
            O[(o, d)] = omit_var
            obj += omit_edge.cost * omit_var
            self._append_arc(node_outgoing, node_incoming, omit_edge.source, omit_edge.target, o, d, omit_var)

            # Source and sink balance. Source pushes the full demand into either
            # the network or the omission arc; sink receives all serviced or
            # omitted cargo.
            model += (
                pulp.lpSum(node_outgoing.get((source, o, d), []))
                - pulp.lpSum(node_incoming.get((source, o, d), []))
                == quantity_by_od[(o, d)],
                f"source_{o}_{d}",
            )
            model += (
                pulp.lpSum(node_incoming.get((sink, o, d), []))
                - pulp.lpSum(node_outgoing.get((sink, o, d), []))
                == quantity_by_od[(o, d)],
                f"sink_{o}_{d}",
            )

        model += obj, "MCFP_objective"

        # Flow conservation on all call nodes for every commodity.
        for node in network.nodes:
            if not node.startswith("call:"):
                continue
            for o, d, _, _ in self.demands:
                inflow = pulp.lpSum(node_incoming.get((node, o, d), []))
                outflow = pulp.lpSum(node_outgoing.get((node, o, d), []))
                model += (inflow == outflow, f"flow_{node.replace(':', '_')}_{o}_{d}")

        # Capacity constraints across commodities.
        for edge in sail_edges:
            model += (
                pulp.lpSum(X[(edge.edge_id, o, d)] for o, d, _, _ in self.demands) <= edge.capacity,
                f"cap_sail_{edge.edge_id.replace(':', '_')}",
            )
        for edge in transship_edges:
            model += (
                pulp.lpSum(U[(edge.edge_id, o, d)] for o, d, _, _ in self.demands) <= edge.capacity,
                f"cap_trans_{edge.edge_id.replace(':', '_')}",
            )
        for edge in load_edges:
            relevant = [V[(edge.edge_id, o, d)] for o, d, _, _ in self.demands if (edge.edge_id, o, d) in V]
            if relevant:
                model += (
                    pulp.lpSum(relevant) <= edge.capacity,
                    f"cap_load_{edge.edge_id.replace(':', '_')}",
                )
        for edge in unload_edges:
            relevant = [W[(edge.edge_id, o, d)] for o, d, _, _ in self.demands if (edge.edge_id, o, d) in W]
            if relevant:
                model += (
                    pulp.lpSum(relevant) <= edge.capacity,
                    f"cap_unload_{edge.edge_id.replace(':', '_')}",
                )

        solver, selection = build_pulp_solver(
            requested=self.solver_backend,
            time_limit=time_limit,
            msg=False,
        )
        model.solve(solver)

        status = pulp.LpStatus[model.status]
        objective = pulp.value(model.objective) if model.status == 1 else float("inf")

        rejected = {}
        residual = {}
        flow = {}
        leg_flow = {}
        total_revenue = 0.0
        total_handling = 0.0
        total_transship = 0.0
        total_penalty = 0.0

        for o, d, k_od, q_od in self.demands:
            rejected_val = pulp.value(O[(o, d)]) or 0.0
            rejected[(o, d)] = rejected_val
            residual[(o, d)] = rejected_val

            transported = k_od - rejected_val
            total_revenue += q_od * transported
            total_penalty += self.config.reject_penalty * rejected_val

        # Count load/unload cost using actual chosen flows rather than
        # reconstructing from transported totals.
        for (edge_id, o, d), var in V.items():
            val = pulp.value(var) or 0.0
            if val <= 0.001:
                continue
            edge = next(e for e in load_edges if e.edge_id == edge_id)
            total_handling += edge.cost * val
            flow.setdefault(edge.rotation_id, {})[(o, d)] = flow.setdefault(edge.rotation_id, {}).get((o, d), 0.0) + val

        for (edge_id, o, d), var in W.items():
            val = pulp.value(var) or 0.0
            if val <= 0.001:
                continue
            edge = next(e for e in unload_edges if e.edge_id == edge_id)
            total_handling += edge.cost * val

        for (edge_id, o, d), var in U.items():
            val = pulp.value(var) or 0.0
            if val <= 0.001:
                continue
            edge = next(e for e in transship_edges if e.edge_id == edge_id)
            total_transship += edge.cost * val

        for (edge_id, o, d), var in X.items():
            val = pulp.value(var) or 0.0
            if val <= 0.001:
                continue
            edge = next(e for e in sail_edges if e.edge_id == edge_id)
            leg_flow.setdefault(edge.rotation_id, {})[(o, d)] = leg_flow.setdefault(edge.rotation_id, {}).get((o, d), 0.0) + val

        return MCFPResult(
            status=status,
            objective=objective,
            total_revenue=total_revenue,
            total_handling_cost=total_handling,
            total_transship_cost=total_transship,
            total_reject_penalty=total_penalty,
            flow=flow,
            leg_flow=leg_flow,
            rejected=rejected,
            residual_demand=residual,
            active_backend=selection.active,
        )

    @staticmethod
    def _append_arc(store_out, store_in, source: str, target: str, o: str, d: str, var):
        store_out.setdefault((source, o, d), []).append(var)
        store_in.setdefault((target, o, d), []).append(var)

    def _apply_assignments(self, vessel_assignments: Optional[Dict[str, int]]) -> List[Rotation]:
        if not vessel_assignments:
            return self.rotations

        adjusted = []
        for rot in self.rotations:
            assigned = vessel_assignments.get(rot.id, rot.num_vessels)
            adjusted.append(
                Rotation(
                    id=rot.id,
                    vessel_class=rot.vessel_class,
                    speed=rot.speed,
                    num_vessels=assigned,
                    port_calls=list(rot.port_calls),
                    is_butterfly=rot.is_butterfly,
                    butterfly_port=rot.butterfly_port,
                )
            )
        return adjusted
