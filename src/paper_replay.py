"""
Paper-derived replay fixtures used as regression guardrails.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from src.model.mcfp import MCFPSolver
from src.utils.cost_calculator import CostCalculator, CostConfig, Port, VesselClass
from src.utils.network_builder import Rotation


def baltic_base_reference_rotations(vessels_by_name: Dict[str, VesselClass]) -> List[Rotation]:
    """Return the three Baltic Base services from the bundled paper log."""
    return [
        Rotation(
            id="paper_0",
            vessel_class=vessels_by_name["Feeder_450"],
            speed=11.1944,
            num_vessels=3,
            port_calls=["RULED", "FIKTK", "DEBRV", "RUKGD", "PLGDY", "DEBRV"],
            is_butterfly=True,
            butterfly_port="DEBRV",
        ),
        Rotation(
            id="paper_1",
            vessel_class=vessels_by_name["Feeder_800"],
            speed=15.4954,
            num_vessels=2,
            port_calls=["RULED", "DEBRV", "NOSVG", "SEGOT", "DEBRV"],
            is_butterfly=True,
            butterfly_port="DEBRV",
        ),
        Rotation(
            id="paper_2",
            vessel_class=vessels_by_name["Feeder_450"],
            speed=10.0,
            num_vessels=1,
            port_calls=["DEBRV", "DKAAR"],
            is_butterfly=False,
            butterfly_port=None,
        ),
    ]


def evaluate_baltic_base_replay(
    vessels_by_name: Dict[str, VesselClass],
    ports_dict: Dict[str, Port],
    demands: List[Tuple[str, str, float, float]],
    dist_min: Dict[Tuple[str, str], float],
    canal_info: Dict[Tuple[str, str], Tuple[bool, bool]],
    cost_config: CostConfig,
    solver_backend: str = "auto",
) -> dict:
    """Replay the Baltic Base paper solution through the current MCFP/cost stack."""
    rotations = baltic_base_reference_rotations(vessels_by_name)
    solver = MCFPSolver(
        rotations=rotations,
        ports_dict=ports_dict,
        demands=demands,
        dist_min=dist_min,
        config=cost_config,
        solver_backend=solver_backend,
    )
    result = solver.solve(time_limit=120)

    calc = CostCalculator(cost_config)
    fixed_cost = 0.0
    vessel_used = {}
    for rot in rotations:
        v = rot.vessel_class
        n = rot.num_vessels
        vessel_used[v.name] = vessel_used.get(v.name, 0) + n
        fixed_cost += v.tc_rate * cost_config.planning_horizon * n
        round_trip = calc.rotation_round_trip_time(v, rot.speed, rot.port_calls, dist_min)
        m_r = calc.num_round_trips(round_trip)
        for idx in range(len(rot.port_calls)):
            i_p = rot.port_calls[idx]
            j_p = rot.port_calls[(idx + 1) % len(rot.port_calls)]
            dist = dist_min.get((i_p, j_p), 0.0)
            fixed_cost += m_r * n * (
                calc.sailing_fuel_cost(v, rot.speed, dist) +
                calc.port_idle_fuel_cost(v)
            )
            port = ports_dict.get(j_p)
            if port:
                fixed_cost += m_r * n * calc.port_call_cost(v, port)
            is_panama, is_suez = canal_info.get((i_p, j_p), (False, False))
            fixed_cost += m_r * n * calc.canal_cost(v, is_panama, is_suez)

    charter_out = 0.0
    for vessel in vessels_by_name.values():
        charter_out += max(vessel.quantity - vessel_used.get(vessel.name, 0), 0) * vessel.tc_rate * cost_config.planning_horizon

    total_demand = sum(k for _, _, k, _ in demands)
    rejected = sum(result.rejected.values())
    full_objective = fixed_cost + result.objective - charter_out

    return {
        "source": "data/LinerLib/results/BrouerDesaulniersPisinger2014/Baltic_best_base.log",
        "status": result.status,
        "objective": full_objective,
        "mcf_objective": result.objective,
        "service_rate": (total_demand - rejected) / max(total_demand, 1.0),
        "rejected_ffe": rejected,
        "costs": {
            "Q": result.total_revenue,
            "c_m": result.total_handling_cost,
            "c_t": result.total_transship_cost,
            "penalty": result.total_reject_penalty,
        },
        "rotations": [
            {
                "id": rot.id,
                "vessel_class": rot.vessel_class.name,
                "num_vessels": rot.num_vessels,
                "speed": rot.speed,
                "port_calls": rot.port_calls,
                "is_butterfly": rot.is_butterfly,
            }
            for rot in rotations
        ],
    }
