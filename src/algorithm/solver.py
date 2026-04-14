"""
统一求解器接口 — 封装 MIP 和元启发式两种求解方法。

提供简洁的 API 用于加载实例数据、配置参数、运行求解、提取结果。
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import json

import pandas as pd

from src.data_reader import load_instance
from src.utils.cost_calculator import (
    VesselClass, Port, CostCalculator, CostConfig,
    build_vessels_from_data, build_ports_from_data,
    build_distance_dict,
)
from src.utils.solver_backend import SolverBackend
from src.utils.network_builder import Rotation
from src.model.mip_model import LsndpMIP, MIPResult, CostBreakdown
from src.model.mcfp import MCFPSolver
from src.model.route_generation import RouteGenerator, AUXConfig
from src.paper_replay import evaluate_baltic_base_replay
from src.algorithm.metaheuristic import (
    LsndMetaheuristic, MetaheuristicConfig, MetaheuristicResult,
)
from src.algorithm.tabu_search import TabuConfig

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """求解器配置。"""
    method: str = "metaheuristic"      # "metaheuristic" or "mip"
    fuel_price: float = 600.0          # USD/ton
    planning_horizon: int = 180        # days
    reject_penalty: float = 1000.0     # USD/FFE
    port_stay_hours: float = 24.0      # hours
    max_time: int = 300                # seconds
    max_iterations: int = 300           # for metaheuristic
    mip_time_limit: int = 300          # for MIP direct solve
    verbose: bool = True
    solver_backend: SolverBackend = "auto"
    random_seed: int = 0
    scenario: str = "base"
    benchmark_mode: bool = False
    collect_diagnostics: bool = False


@dataclass
class SolverResult:
    """求解结果。"""
    instance_name: str
    method: str
    status: str
    objective: float
    cost_breakdown: CostBreakdown
    rotations: List[Rotation]
    rotation_details: List[dict]       # 每条航线的详细信息
    flow_summary: Dict[str, float]     # 流量统计
    rejected_demand: Dict[Tuple[str, str], float]
    fleet_usage: Dict[str, dict]       # 船型使用情况
    solve_time: float
    iterations: int
    solver_backend: str = "cbc"
    seed: int = 0
    scenario: str = "base"
    columns_evaluated: int = 0
    unique_columns: int = 0
    mcf_evaluations: int = 0
    accepted_columns: int = 0
    same_class_swap_count: int = 0
    backtrack_count: int = 0
    paper_gap_pct: Optional[float] = None
    diagnostics_path: Optional[str] = None


class LsndSolver:
    """
    LSNDP 统一求解器。

    用法:
        solver = LsndSolver("Baltic")
        result = solver.solve()
        print(result.cost_breakdown)
    """

    def __init__(
        self,
        instance_name: str,
        config: SolverConfig = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.instance_name = instance_name
        self.config = config or SolverConfig()
        self.progress_callback = progress_callback

        # 加载数据
        data = load_instance(instance_name, scenario=self.config.scenario)
        self.data = data

        # 构建数据结构
        self.vessels = build_vessels_from_data(data["fleet"])
        self.ports_dict = build_ports_from_data(data["ports_all"])
        self.dist_full, self.dist_min, self.canal_info = build_distance_dict(
            self._load_distances()
        )

        # 需求列表: FFEPerWeek 需乘以 (T/7) 转换为规划期总量
        # 论文模型中所有量均为规划期 T 天内的总量
        demand_df = data["demand"]
        weeks_in_horizon = self.config.planning_horizon / 7.0
        self.demands = [
            (row["Origin"], row["Destination"],
             float(row["FFEPerWeek"]) * weeks_in_horizon, float(row["Revenue_1"]))
            for _, row in demand_df.iterrows()
        ]

        # 成本配置
        self.cost_config = CostConfig(
            fuel_price=self.config.fuel_price,
            planning_horizon=self.config.planning_horizon,
            reject_penalty=self.config.reject_penalty,
            port_stay_hours=self.config.port_stay_hours,
        )

    def _diagnostics_path(self) -> Optional[Path]:
        if not self.config.collect_diagnostics:
            return None
        diagnostics_dir = Path("results") / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        return diagnostics_dir / f"{self.instance_name}_{self.config.scenario}_{self.config.random_seed}.json"

    def _load_distances(self):
        from src.data_reader import load_distances
        return load_distances()

    def solve(self) -> SolverResult:
        """运行求解器。"""
        if self.config.method == "metaheuristic":
            return self._solve_metaheuristic()
        elif self.config.method == "mip":
            return self._solve_mip_with_generated_routes()
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _solve_metaheuristic(self) -> SolverResult:
        """使用元启发式求解。"""
        start = time.time()

        meta_config = MetaheuristicConfig(
            max_time=self.config.max_time,
            max_iterations=self.config.max_iterations,
            progress_callback=self.progress_callback,
            random_seed=self.config.random_seed,
            solver_backend=self.config.solver_backend,
            collect_diagnostics=self.config.collect_diagnostics,
        )
        tabu_config = TabuConfig()

        solver = LsndMetaheuristic(
            vessels=self.vessels,
            ports_dict=self.ports_dict,
            demands=self.demands,
            dist_min=self.dist_min,
            canal_info=self.canal_info,
            cost_config=self.cost_config,
            meta_config=meta_config,
            tabu_config=tabu_config,
        )

        result = solver.solve()

        # 构建结果
        solver_result = self._build_result(
            method="metaheuristic",
            status="Optimal" if result.best_objective < float("inf") else "No solution",
            objective=result.best_objective,
            rotations=result.best_rotations,
            mcfp_result=result.mcfp_result,
            solve_time=result.total_time,
            iterations=result.total_iterations,
            solver_backend=result.active_backend,
            columns_evaluated=result.total_columns_evaluated,
            unique_columns=result.unique_columns,
            mcf_evaluations=result.mcf_evaluations,
            accepted_columns=result.accepted_columns,
            same_class_swap_count=result.same_class_swap_count,
            backtrack_count=result.backtrack_count,
        )
        self._write_diagnostics(solver_result)
        return solver_result

    def _solve_mip_with_generated_routes(self) -> SolverResult:
        """
        先用 AUX 模型生成候选航线，再用 MIP 主模型求解。
        适用于小实例。
        """
        start = time.time()

        # Step 1: 生成候选航线
        all_ports = list(set(
            [o for o, _, _, _ in self.demands] + [d for _, d, _, _ in self.demands]
        ))
        residual_demand = {(o, d): k for o, d, k, _ in self.demands}
        demand_revenue = {(o, d): q for o, d, _, q in self.demands}

        candidates = []
        rot_counter = 0

        for v in self.vessels:
            port_objs = [self.ports_dict[p] for p in all_ports if p in self.ports_dict]

            time_per_v = (self.config.max_time * 0.5) / max(1, len(self.vessels))
            v_start = time.time()

            aux_cfg = AUXConfig(
                max_ports=min(len(all_ports), 20),
                suppress_subtour=True,
                time_limit=max(5, int(time_per_v / 10)),
                solver_backend=self.config.solver_backend,
            )

            generator = RouteGenerator(
                ports=port_objs,
                port_codes=all_ports,
                vessel=v,
                residual_demand=residual_demand,
                demand_revenue=demand_revenue,
                dist_min=self.dist_min,
                canal_info=self.canal_info,
                ports_dict=self.ports_dict,
                config=self.cost_config,
                aux_config=aux_cfg,
            )

            speeds = list(range(int(v.min_speed), int(v.max_speed) + 1, 2))
            for s in speeds:
                for eta in [1, 2, 3]:
                    if eta > v.quantity:
                        break
                    rot_id = f"rot_{rot_counter}"
                    rot_counter += 1
                    rot = generator.solve(float(s), eta, rot_id)
                    if rot:
                        candidates.append(rot)

                    if time.time() - v_start > time_per_v:
                        break
                if time.time() - v_start > time_per_v:
                    break
            if time.time() - start > self.config.max_time * 0.7:
                break

        logger.info(f"Generated {len(candidates)} candidate rotations")

        if not candidates:
            return SolverResult(
                instance_name=self.instance_name,
                method="mip",
                status="No routes generated",
                objective=float("inf"),
                cost_breakdown=CostBreakdown(),
                rotations=[],
                rotation_details=[],
                flow_summary={},
                rejected_demand={(o, d): k for o, d, k, _ in self.demands},
                fleet_usage={},
                solve_time=time.time() - start,
                iterations=0,
            )

        # Step 2: 用 MIP 主模型求解
        mip = LsndpMIP(
            rotations=candidates,
            vessels=self.vessels,
            ports_dict=self.ports_dict,
            demands=self.demands,
            dist_min=self.dist_min,
            canal_info=self.canal_info,
            config=self.cost_config,
            solver_backend=self.config.solver_backend,
        )
        mip.build()
        mip_result = mip.solve(
            time_limit=max(60, self.config.mip_time_limit),
            msg=self.config.verbose,
        )

        # 筛选被使用的航线
        used_rotations = [
            rot for rot in candidates
            if mip_result.vessel_assignments.get(rot.id, 0) > 0
        ]

        solver_result = self._build_result(
            method="mip",
            status=mip_result.status,
            objective=mip_result.objective,
            rotations=used_rotations,
            mip_result=mip_result,
            solve_time=time.time() - start,
            iterations=1,
            solver_backend=mip_result.active_backend,
        )
        self._write_diagnostics(solver_result)
        return solver_result

    def _build_result(self, method, status, objective, rotations,
                      mcfp_result=None, mip_result=None,
                      solve_time=0.0, iterations=0,
                      solver_backend="cbc",
                      columns_evaluated=0,
                      unique_columns=0,
                      mcf_evaluations=0,
                      accepted_columns=0,
                      same_class_swap_count=0,
                      backtrack_count=0) -> SolverResult:
        """构建统一的 SolverResult。"""
        cost_calc = CostCalculator(self.cost_config)

        # MIP 实际船舶分配 (如果有)
        mip_assignments = mip_result.vessel_assignments if mip_result else None

        # 航线详情
        rotation_details = []
        for rot in rotations:
            # 使用 MIP 的 Y_r (如果可用), 否则使用 AUX 建议值
            n_v = mip_assignments.get(rot.id, rot.num_vessels) if mip_assignments else rot.num_vessels
            round_trip = cost_calc.rotation_round_trip_time(
                rot.vessel_class, rot.speed, rot.port_calls, self.dist_min
            )
            rotation_details.append({
                "id": rot.id,
                "vessel_class": rot.vessel_class.name,
                "num_vessels": n_v,
                "speed": rot.speed,
                "port_calls": rot.port_calls,
                "round_trip_days": round(round_trip, 1),
                "frequency": "weekly" if round_trip <= 7 * max(n_v, 1) * 1.1 else "biweekly",
                "is_butterfly": rot.is_butterfly,
            })

        # 船队使用
        vessel_used = {}
        for rot in rotations:
            vn = rot.vessel_class.name
            n_v = mip_assignments.get(rot.id, rot.num_vessels) if mip_assignments else rot.num_vessels
            if vn not in vessel_used:
                vessel_used[vn] = {"deployed": 0, "available": 0}
            vessel_used[vn]["deployed"] += n_v
        for v in self.vessels:
            if v.name not in vessel_used:
                vessel_used[v.name] = {"deployed": 0, "available": v.quantity}
            else:
                vessel_used[v.name]["available"] = v.quantity

        # 被拒绝需求
        rejected = {}
        if mip_result:
            rejected = mip_result.rejected_demand
        elif mcfp_result:
            rejected = mcfp_result.rejected

        # 成本分解
        breakdown = CostBreakdown()
        if mip_result:
            breakdown = mip_result.cost_breakdown
        elif mcfp_result and rotations:
            # 从 MCFP 结果构建近似分解
            breakdown = self._approximate_breakdown(rotations, mcfp_result)

        # 流量统计
        total_demand = sum(k for _, _, k, _ in self.demands)
        total_rejected = sum(rejected.values())
        flow_summary = {
            "total_demand_ffe": total_demand,
            "transported_ffe": total_demand - total_rejected,
            "rejected_ffe": total_rejected,
            "service_rate": (total_demand - total_rejected) / max(total_demand, 1.0),
            "n_rotations": len(rotations),
        }

        diagnostics_path = self._diagnostics_path()

        return SolverResult(
            instance_name=self.instance_name,
            method=method,
            status=status,
            objective=objective,
            cost_breakdown=breakdown,
            rotations=rotations,
            rotation_details=rotation_details,
            flow_summary=flow_summary,
            rejected_demand=rejected,
            fleet_usage=vessel_used,
            solve_time=solve_time,
            iterations=iterations,
            solver_backend=solver_backend,
            seed=self.config.random_seed,
            scenario=self.config.scenario,
            columns_evaluated=columns_evaluated,
            unique_columns=unique_columns,
            mcf_evaluations=mcf_evaluations,
            accepted_columns=accepted_columns,
            same_class_swap_count=same_class_swap_count,
            backtrack_count=backtrack_count,
            diagnostics_path=str(diagnostics_path) if diagnostics_path else None,
        )

    def _approximate_breakdown(self, rotations, mcfp_result) -> CostBreakdown:
        """从元启发式结果近似计算成本分解。"""
        bd = CostBreakdown()
        cost_calc = CostCalculator(self.cost_config)
        T = self.cost_config.planning_horizon

        for rot in rotations:
            v = rot.vessel_class
            n = rot.num_vessels
            bd.c_v += v.tc_rate * T * n

            round_trip = cost_calc.rotation_round_trip_time(
                v, rot.speed, rot.port_calls, self.dist_min
            )
            m_r = cost_calc.num_round_trips(round_trip)

            for k in range(len(rot.port_calls)):
                i_p = rot.port_calls[k]
                j_p = rot.port_calls[(k + 1) % len(rot.port_calls)]
                dist = self.dist_min.get((i_p, j_p), 0.0)
                bd.c_b += m_r * n * (
                    cost_calc.sailing_fuel_cost(v, rot.speed, dist) +
                    cost_calc.port_idle_fuel_cost(v)
                )
                port = self.ports_dict.get(j_p)
                if port:
                    bd.c_p += m_r * n * cost_calc.port_call_cost(v, port)
                is_panama, is_suez = self.canal_info.get((i_p, j_p), (False, False))
                bd.c_c += m_r * n * cost_calc.canal_cost(v, is_panama, is_suez)

        # 未使用船舶
        vessel_used = {}
        for rot in rotations:
            vn = rot.vessel_class.name
            vessel_used[vn] = vessel_used.get(vn, 0) + rot.num_vessels
        for v in self.vessels:
            unused = v.quantity - vessel_used.get(v.name, 0)
            if unused > 0:
                bd.L_v += v.tc_rate * T * unused

        bd.Q = mcfp_result.total_revenue
        bd.c_m = mcfp_result.total_handling_cost
        bd.c_t = mcfp_result.total_transship_cost
        bd.F = sum(mcfp_result.rejected.values())
        bd.Z = bd.c_v + bd.c_b + bd.c_p + bd.c_c + bd.c_m + bd.c_t - bd.Q - bd.L_v + self.cost_config.reject_penalty * bd.F

        return bd

    def _write_diagnostics(self, result: SolverResult) -> None:
        diagnostics_path = self._diagnostics_path()
        if diagnostics_path is None:
            return

        payload = {
            "instance": result.instance_name,
            "scenario": result.scenario,
            "seed": result.seed,
            "method": result.method,
            "status": result.status,
            "objective": result.objective,
            "solver_backend": result.solver_backend,
            "solve_time": result.solve_time,
            "iterations": result.iterations,
            "columns_evaluated": result.columns_evaluated,
            "unique_columns": result.unique_columns,
            "mcf_evaluations": result.mcf_evaluations,
            "accepted_columns": result.accepted_columns,
            "same_class_swap_count": result.same_class_swap_count,
            "backtrack_count": result.backtrack_count,
            "flow_summary": result.flow_summary,
            "cost_breakdown": result.cost_breakdown.__dict__,
            "rotations": result.rotation_details,
            "selected_rotation_signatures": [
                {
                    "vessel_class": rot.vessel_class.name,
                    "speed": round(rot.speed, 4),
                    "num_vessels": rot.num_vessels,
                    "port_calls": list(rot.port_calls),
                    "butterfly_port": rot.butterfly_port,
                }
                for rot in result.rotations
            ],
        }
        if result.instance_name == "Baltic" and result.scenario == "base":
            vessels_by_name = {v.name: v for v in self.vessels}
            payload["paper_replay_comparison"] = evaluate_baltic_base_replay(
                vessels_by_name=vessels_by_name,
                ports_dict=self.ports_dict,
                demands=self.demands,
                dist_min=self.dist_min,
                canal_info=self.canal_info,
                cost_config=self.cost_config,
                solver_backend=self.config.solver_backend,
            )
        diagnostics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def solve_instance(
    instance_name: str,
    method: str = "metaheuristic",
    max_time: int = 300,
    fuel_price: float = 600.0,
    verbose: bool = True,
    solver_backend: SolverBackend = "auto",
    random_seed: int = 0,
    scenario: str = "base",
    benchmark_mode: bool = False,
    collect_diagnostics: bool = False,
) -> SolverResult:
    """
    便捷函数: 求解一个 LINERLIB 实例。

    用法:
        result = solve_instance("Baltic", method="metaheuristic", max_time=300)
        print(f"Objective: {result.objective:.0f}")
        print(f"Rotations: {len(result.rotations)}")
    """
    config = SolverConfig(
        method=method,
        fuel_price=fuel_price,
        max_time=max_time,
        verbose=verbose,
        solver_backend=solver_backend,
        random_seed=random_seed,
        scenario=scenario,
        benchmark_mode=benchmark_mode,
        collect_diagnostics=collect_diagnostics,
    )
    solver = LsndSolver(instance_name, config)
    return solver.solve()
