"""
元启发式算法 — 实现论文 §6.2 的禁忌搜索 + 启发式列生成框架。

算法流程:
1. 初始化空航线集合 R_0
2. 迭代:
   a. 用 AUX 模型生成候选航线 (列生成)
   b. 逐一评估候选航线 (MCFP)
   c. 选择使目标最优的航线加入
   d. 禁忌搜索 + 回溯
3. 终止于最大运行时间

参考: Brouer et al. (2014), Transportation Science 48(2), pp. 299-301
"""

import time
import logging
import random
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from src.utils.cost_calculator import (
    VesselClass, Port, CostCalculator, CostConfig,
)
from src.utils.network_builder import Rotation
from src.model.route_generation import (
    RouteGenerator, AUXConfig, create_port_clusters,
)
from src.model.mcfp import MCFPSolver, MCFPResult
from src.algorithm.tabu_search import (
    TabuList, BacktrackManager, TabuConfig, score_rotations,
)
from src.utils.solver_backend import SolverBackend

logger = logging.getLogger(__name__)


@dataclass
class MetaheuristicConfig:
    """元启发式算法配置。"""
    max_time: int = 300              # 最大运行时间 (秒)
    max_iterations: int = 200        # 最大迭代数
    aux_time_limit: int = 5          # AUX 子问题时限
    mcfp_time_limit: int = 60        # MCFP 时限
    max_cluster_size: int = 20       # AUX 子问题最大港口数
    speed_step: int = 1              # 速度搜索步长 (细粒度搜索)
    initial_speed_narrow: bool = False  # 从第一轮就搜索全速度范围
    suppress_subtour_iters: int = 5   # 前 N 轮不启用子环消除
    aux_num_solutions: int = 1        # 每个 AUX(v,s,eta) 默认保留的候选整数解数量
    max_candidates_per_iteration: int = 30
    progress_callback: Optional[Callable] = None  # 进度回调
    random_seed: int = 0
    solver_backend: SolverBackend = "auto"
    collect_diagnostics: bool = False


@dataclass
class IterationLog:
    """单次迭代日志。"""
    iteration: int
    n_rotations: int
    n_candidates: int
    objective: float
    best_objective: float
    improved: bool
    elapsed_sec: float


@dataclass
class MetaheuristicResult:
    """元启发式算法结果。"""
    best_rotations: List[Rotation]
    best_objective: float
    mcfp_result: Optional[MCFPResult]
    iteration_logs: List[IterationLog]
    total_time: float
    total_iterations: int
    total_columns_evaluated: int
    unique_columns: int
    mcf_evaluations: int
    active_backend: str = "cbc"
    accepted_columns: int = 0
    same_class_swap_count: int = 0
    backtrack_count: int = 0


class LsndMetaheuristic:
    """
    LSNDP 元启发式求解器。
    """

    def __init__(
        self,
        vessels: List[VesselClass],
        ports_dict: Dict[str, Port],
        demands: List[Tuple[str, str, float, float]],
        dist_min: Dict[Tuple[str, str], float],
        canal_info: Dict[Tuple[str, str], Tuple[bool, bool]],
        cost_config: CostConfig = None,
        meta_config: MetaheuristicConfig = None,
        tabu_config: TabuConfig = None,
    ):
        self.vessels = vessels
        self.ports_dict = ports_dict
        self.demands = demands
        self.dist_min = dist_min
        self.canal_info = canal_info
        self.cost_config = cost_config or CostConfig()
        self.meta_config = meta_config or MetaheuristicConfig()
        self.tabu_config = tabu_config or TabuConfig()
        self.cost_calc = CostCalculator(self.cost_config)

        # 所有港口
        self.all_ports = list(set(
            [o for o, _, _, _ in demands] + [d for _, d, _, _ in demands]
        ))

    def solve(self) -> MetaheuristicResult:
        """运行元启发式算法。"""
        start_time = time.time()
        cfg = self.meta_config
        rng = random.Random(cfg.random_seed)

        # 初始化
        current_rotations: List[Rotation] = []
        residual_demand = {(o, d): k for o, d, k, _ in self.demands}
        demand_revenue = {(o, d): q for o, d, _, q in self.demands}

        tabu_list = TabuList(self.tabu_config.tabu_tenure)
        backtrack_mgr = BacktrackManager(self.tabu_config)

        logs: List[IterationLog] = []
        total_columns = 0
        best_objective = float("inf")
        best_rotations = []
        best_mcfp = None
        current_mcfp = None
        current_objective = float("inf")
        unique_columns = set()
        mcf_evaluations = 0
        accepted_columns = 0
        same_class_swap_count = 0

        rot_counter = 0

        for iteration in range(cfg.max_iterations):
            elapsed = time.time() - start_time
            if elapsed >= cfg.max_time:
                logger.info(f"Time limit reached at iteration {iteration}")
                break

            logger.info(f"=== Iteration {iteration} === "
                        f"Rotations: {len(current_rotations)}, "
                        f"Best obj: {best_objective:.0f}")

            # 进度回调
            if cfg.progress_callback:
                cfg.progress_callback(iteration, cfg.max_iterations, best_objective)

            # --- Step 1: 计算剩余船舶 ---
            vessel_used = {}
            for rot in current_rotations:
                vn = rot.vessel_class.name
                vessel_used[vn] = vessel_used.get(vn, 0) + rot.num_vessels
            available_vessels = {
                v.name: v.quantity - vessel_used.get(v.name, 0)
                for v in self.vessels
            }

            # --- Step 2: 生成候选航线 (列生成) ---
            candidates = []
            suppress_subtour = iteration < cfg.suppress_subtour_iters

            # 端口聚类
            clusters = create_port_clusters(
                self.all_ports, residual_demand, self.dist_min,
                max_cluster_size=cfg.max_cluster_size,
            )
            if not clusters:
                clusters = [self.all_ports[:cfg.max_cluster_size]]

            for v_idx, v in enumerate(self.vessels):
                # Always generate candidates to allow swapping existing ones
                n_vessels_to_try = v.quantity

                # 使用多个聚类增加航线多样性（论文: 同时运行 16 个子问题）
                # 对排名前 n_clusters_per_vessel 的聚类都生成候选
                n_clusters_per_vessel = min(3, len(clusters))
                start_idx = rng.randrange(len(clusters))
                cluster_indices = [
                    (start_idx + iteration + v_idx + c) % len(clusters)
                    for c in range(n_clusters_per_vessel)
                ]
                # 去重
                cluster_indices = list(dict.fromkeys(cluster_indices))

                for cluster_idx in cluster_indices:
                    cluster = clusters[cluster_idx]
                    cluster_ports = [self.ports_dict[p] for p in cluster if p in self.ports_dict]
                    remaining_time = cfg.max_time - (time.time() - start_time)
                    if remaining_time <= 1.0:
                        break

                    # 速度范围
                    if cfg.initial_speed_narrow and iteration < 5:
                        speeds = [v.design_speed]
                    else:
                        speeds = list(range(int(v.min_speed), int(v.max_speed) + 1, cfg.speed_step))

                    vessel_counts = list(range(1, min(n_vessels_to_try + 1, 4)))

                    aux_cfg = AUXConfig(
                        max_ports=cfg.max_cluster_size,
                        suppress_subtour=not suppress_subtour,
                        time_limit=max(1, min(cfg.aux_time_limit, int(max(remaining_time - 1.0, 1.0)))),
                        solver_backend=cfg.solver_backend,
                    )

                    generator = RouteGenerator(
                        ports=cluster_ports,
                        port_codes=cluster,
                        vessel=v,
                        residual_demand=residual_demand,
                        demand_revenue=demand_revenue,
                        dist_min=self.dist_min,
                        canal_info=self.canal_info,
                        ports_dict=self.ports_dict,
                        config=self.cost_config,
                        aux_config=aux_cfg,
                    )

                    for s in speeds:
                        for eta in vessel_counts:
                            rot_id = f"rot_{rot_counter}"
                            rot_counter += 1
                            for result in generator.solve_many(
                                speed=float(s),
                                num_vessels=eta,
                                rotation_prefix=rot_id,
                                limit=cfg.aux_num_solutions,
                            ):
                                if self._candidate_potential_score(result, residual_demand, demand_revenue) <= 0:
                                    continue
                                candidates.append(result)
                                unique_columns.add(self._rotation_signature(result))

                            # 时间检查
                            if time.time() - start_time >= cfg.max_time:
                                break
                        if time.time() - start_time >= cfg.max_time:
                            break
                    if time.time() - start_time >= cfg.max_time:
                        break
                if time.time() - start_time >= cfg.max_time:
                    break

            total_columns += len(candidates)
            seed_candidates = self._generate_seed_routes(residual_demand)
            for seed_rot in seed_candidates:
                sig = self._rotation_signature(seed_rot)
                if sig in unique_columns:
                    continue
                candidates.append(seed_rot)
                unique_columns.add(sig)
            total_columns += len(seed_candidates)
            candidates = self._select_candidate_subset(
                candidates=candidates,
                residual_demand=residual_demand,
                demand_revenue=demand_revenue,
                limit=cfg.max_candidates_per_iteration,
            )
            logger.info(f"  Generated {len(candidates)} candidates")

            # --- Step 3: 评估候选航线 ---
            best_candidate = None
            best_candidate_obj = float("inf")
            best_candidate_mcfp = None

            best_candidate_removed = []

            for cand in candidates:
                eval_scenarios = self._build_eval_scenarios(
                    candidate=cand,
                    current_rotations=current_rotations,
                    current_mcfp=current_mcfp,
                )

                for test_rotations, removals in eval_scenarios:
                    remaining_time = cfg.max_time - (time.time() - start_time)
                    if remaining_time <= 1.0:
                        break
                    mcfp = MCFPSolver(
                        rotations=test_rotations,
                        ports_dict=self.ports_dict,
                        demands=self.demands,
                        dist_min=self.dist_min,
                        config=self.cost_config,
                        solver_backend=cfg.solver_backend,
                    )

                    try:
                        mcfp_result = mcfp.solve(
                            time_limit=max(1, min(cfg.mcfp_time_limit, int(max(remaining_time - 0.5, 1.0))))
                        )
                        mcf_evaluations += 1
                    except Exception as e:
                        logger.warning(f"  MCFP failed for {cand.id}: {e}")
                        continue

                    if mcfp_result.status != "Optimal":
                        continue

                    # 计算完整目标值 (包括航线固定成本)
                    obj = self._compute_full_objective(test_rotations, mcfp_result)

                    if obj < best_candidate_obj:
                        best_candidate_obj = obj
                        best_candidate = cand
                        best_candidate_mcfp = mcfp_result
                        best_candidate_removed = removals

                # 时间检查
                if time.time() - start_time >= cfg.max_time:
                    break

            # --- Step 4: 更新航线集合 ---
            improved = False
            accepted = False
            if best_candidate is not None and best_candidate_obj < current_objective:
                for r_rm in best_candidate_removed:
                    current_rotations.remove(r_rm)
                    logger.info(f"  SWAPPED OUT: {r_rm.id}")
                current_rotations.append(best_candidate)
                tabu_list.add(best_candidate.id, iteration)
                current_objective = best_candidate_obj
                current_mcfp = best_candidate_mcfp
                accepted = True
                accepted_columns += 1
                if best_candidate_removed:
                    same_class_swap_count += 1

                # 更新残差需求
                if best_candidate_mcfp:
                    residual_demand = dict(best_candidate_mcfp.residual_demand)

                logger.info(f"  ACCEPTED: obj={current_objective:.0f}, "
                            f"added {best_candidate.id}")
            elif best_candidate is not None:
                # 未改进但有候选: 有条件接受 (禁忌搜索的邻域探索)
                # 论文策略: 只在目标值恶化不超过 5% 时接受，且限制连续非改进接受次数
                tolerance = 0.05  # 允许 5% 的恶化
                if current_objective != float("inf") and current_objective != 0:
                    degradation = (best_candidate_obj - current_objective) / abs(current_objective)
                else:
                    degradation = 0.0
                if degradation <= tolerance:
                    for r_rm in best_candidate_removed:
                        current_rotations.remove(r_rm)
                        logger.info(f"  SWAPPED OUT (non-improving, {degradation:.1%}): {r_rm.id}")
                    current_rotations.append(best_candidate)
                    tabu_list.add(best_candidate.id, iteration)
                    current_objective = best_candidate_obj
                    current_mcfp = best_candidate_mcfp
                    accepted = True
                    accepted_columns += 1
                    if best_candidate_removed:
                        same_class_swap_count += 1
                    if best_candidate_mcfp:
                        residual_demand = dict(best_candidate_mcfp.residual_demand)
                else:
                    logger.info(f"  Rejected non-improving candidate ({degradation:.1%} degradation)")

            if accepted and current_objective < best_objective:
                best_objective = current_objective
                best_rotations = list(current_rotations)
                best_mcfp = current_mcfp
                improved = True

            backtrack_mgr.update_best(best_rotations, best_objective)

            # --- Step 5: 禁忌搜索 - 回溯 ---
            if backtrack_mgr.should_backtrack() and best_rotations:
                logger.info("  BACKTRACKING")
                # 计算航线评分
                flow = best_mcfp.leg_flow if best_mcfp else {}
                caps = {rot.id: rot.vessel_class.capacity for rot in current_rotations}
                scores = score_rotations(current_rotations, flow, caps)
                current_rotations = backtrack_mgr.backtrack(scores)
                if current_rotations:
                    recomputed_mcfp = MCFPSolver(
                        rotations=current_rotations,
                        ports_dict=self.ports_dict,
                        demands=self.demands,
                        dist_min=self.dist_min,
                        config=self.cost_config,
                        solver_backend=cfg.solver_backend,
                    )
                    remaining_time = cfg.max_time - (time.time() - start_time)
                    recomputed_result = recomputed_mcfp.solve(
                        time_limit=max(1, min(cfg.mcfp_time_limit, int(max(remaining_time - 0.5, 1.0))))
                    )
                    mcf_evaluations += 1
                    if recomputed_result.status == "Optimal":
                        current_mcfp = recomputed_result
                        current_objective = self._compute_full_objective(current_rotations, recomputed_result)
                        residual_demand = dict(recomputed_result.residual_demand)
                    else:
                        current_objective = best_objective
                        current_mcfp = best_mcfp
                        residual_demand = dict(best_mcfp.residual_demand) if best_mcfp else {
                            (o, d): k for o, d, k, _ in self.demands
                        }
                else:
                    current_objective = float("inf")
                    current_mcfp = None
                    residual_demand = {(o, d): k for o, d, k, _ in self.demands}

            # 清理禁忌列表
            tabu_list.cleanup(iteration)

            # 无候选航线时删除一些航线
            if not candidates and current_rotations:
                # 删除最近添加的非禁忌航线
                to_remove = None
                for rot in reversed(current_rotations):
                    if not tabu_list.is_tabu(rot.id, iteration):
                        to_remove = rot
                        break
                if to_remove:
                    current_rotations.remove(to_remove)
                    logger.info(f"  Removed {to_remove.id} (no candidates)")

            logs.append(IterationLog(
                iteration=iteration,
                n_rotations=len(current_rotations),
                n_candidates=len(candidates),
                objective=best_candidate_obj if best_candidate else float("inf"),
                best_objective=best_objective,
                improved=improved,
                elapsed_sec=time.time() - start_time,
            ))

        total_time = time.time() - start_time

        return MetaheuristicResult(
            best_rotations=best_rotations,
            best_objective=best_objective,
            mcfp_result=best_mcfp,
            iteration_logs=logs,
            total_time=total_time,
            total_iterations=len(logs),
            total_columns_evaluated=total_columns,
            unique_columns=len(unique_columns),
            mcf_evaluations=mcf_evaluations,
            active_backend=best_mcfp.active_backend if best_mcfp else "cbc",
            accepted_columns=accepted_columns,
            same_class_swap_count=same_class_swap_count,
            backtrack_count=backtrack_mgr.backtrack_count,
        )

    def _compute_full_objective(
        self,
        rotations: List[Rotation],
        mcfp_result: MCFPResult,
    ) -> float:
        """
        计算完整目标函数值 (航线固定成本 + MCFP 成本)。
        """
        # 航线固定运营成本
        fixed_cost = 0.0
        for rot in rotations:
            v = rot.vessel_class
            n = rot.num_vessels
            T = self.cost_config.planning_horizon

            # TC 费用
            fixed_cost += v.tc_rate * T * n

            # 运营成本
            round_trip_days = self.cost_calc.rotation_round_trip_time(
                v, rot.speed, rot.port_calls, self.dist_min
            )
            m_r = self.cost_calc.num_round_trips(round_trip_days)

            for k in range(len(rot.port_calls)):
                i_p = rot.port_calls[k]
                j_p = rot.port_calls[(k + 1) % len(rot.port_calls)]
                dist = self.dist_min.get((i_p, j_p), 0.0)

                fixed_cost += m_r * n * (
                    self.cost_calc.sailing_fuel_cost(v, rot.speed, dist) +
                    self.cost_calc.port_idle_fuel_cost(v)
                )
                port = self.ports_dict.get(j_p)
                if port:
                    fixed_cost += m_r * n * self.cost_calc.port_call_cost(v, port)

                is_panama, is_suez = self.canal_info.get((i_p, j_p), (False, False))
                fixed_cost += m_r * n * self.cost_calc.canal_cost(v, is_panama, is_suez)

        # 未使用船舶的 charter out revenue
        vessel_used = {}
        for rot in rotations:
            vn = rot.vessel_class.name
            vessel_used[vn] = vessel_used.get(vn, 0) + rot.num_vessels
        charter_out = 0.0
        T = self.cost_config.planning_horizon
        for v in self.vessels:
            unused = v.quantity - vessel_used.get(v.name, 0)
            if unused > 0:
                charter_out += v.tc_rate * T * unused

        # 完整目标 = 固定成本 + MCFP 目标 - charter out
        return fixed_cost + mcfp_result.objective - charter_out

    @staticmethod
    def _rotation_signature(rotation: Rotation) -> Tuple:
        return (
            rotation.vessel_class.name,
            round(rotation.speed, 2),
            rotation.num_vessels,
            tuple(rotation.port_calls),
            rotation.is_butterfly,
            rotation.butterfly_port,
        )

    def _build_eval_scenarios(
        self,
        candidate: Rotation,
        current_rotations: List[Rotation],
        current_mcfp: Optional[MCFPResult],
    ) -> List[Tuple[List[Rotation], List[Rotation]]]:
        """Build route-removal scenarios for evaluating a candidate column."""
        cand_v = candidate.vessel_class.name
        current_used = sum(r.num_vessels for r in current_rotations if r.vessel_class.name == cand_v)
        needed = candidate.num_vessels
        if current_used + needed <= candidate.vessel_class.quantity:
            return [(current_rotations + [candidate], [])]

        same_class = [r for r in current_rotations if r.vessel_class.name == cand_v]
        if not same_class:
            return []

        utilization = self._route_utilization_map(current_rotations, current_mcfp)
        ranked_same_class = sorted(
            same_class,
            key=lambda rot: (utilization.get(rot.id, 0.0), rot.num_vessels, rot.id),
        )

        combos = []
        seen = set()
        for size in (1, 2):
            if size > len(ranked_same_class):
                continue
            for combo in combinations(ranked_same_class, size):
                freed = sum(rot.num_vessels for rot in combo)
                if current_used - freed + needed > candidate.vessel_class.quantity:
                    continue
                combo_ids = tuple(sorted(rot.id for rot in combo))
                if combo_ids in seen:
                    continue
                seen.add(combo_ids)
                avg_util = sum(utilization.get(rot.id, 0.0) for rot in combo) / len(combo)
                combos.append((avg_util, len(combo), list(combo)))

        combos.sort(key=lambda item: (item[0], item[1], tuple(rot.id for rot in item[2])))
        return [
            ([rot for rot in current_rotations if rot not in removals] + [candidate], removals)
            for _, _, removals in combos
        ]

    @staticmethod
    def _route_utilization_map(
        rotations: List[Rotation],
        current_mcfp: Optional[MCFPResult],
    ) -> Dict[str, float]:
        if not current_mcfp:
            return {rot.id: 0.0 for rot in rotations}
        utilization = {}
        for rot in rotations:
            leg_flow = current_mcfp.leg_flow.get(rot.id, {})
            total_leg_flow = sum(leg_flow.values())
            cap = max(rot.vessel_class.capacity * max(rot.num_vessels, 1) * max(len(rot.port_calls), 1), 1.0)
            utilization[rot.id] = min(total_leg_flow / cap, 1.0)
        return utilization

    def _candidate_potential_score(
        self,
        candidate: Rotation,
        residual_demand: Dict[Tuple[str, str], float],
        demand_revenue: Dict[Tuple[str, str], float],
    ) -> float:
        ports_in_route = set(candidate.port_calls)
        score = 0.0
        for (o, d), qty in residual_demand.items():
            if qty <= 0 or o not in ports_in_route or d not in ports_in_route:
                continue
            if o == d:
                continue
            score += qty * max(demand_revenue.get((o, d), 0.0), 0.0)
        return score

    def _select_candidate_subset(
        self,
        candidates: List[Rotation],
        residual_demand: Dict[Tuple[str, str], float],
        demand_revenue: Dict[Tuple[str, str], float],
        limit: int,
    ) -> List[Rotation]:
        """Keep a diverse, high-value subset of candidates for MCF evaluation."""
        if len(candidates) <= limit:
            return candidates

        potential_sorted = sorted(
            candidates,
            key=lambda rot: self._candidate_potential_score(rot, residual_demand, demand_revenue),
            reverse=True,
        )
        structural_sorted = sorted(
            candidates,
            key=lambda rot: (
                1 if rot.is_butterfly else 0,
                len(set(rot.port_calls)),
                len(rot.port_calls),
                self._candidate_potential_score(rot, residual_demand, demand_revenue),
            ),
            reverse=True,
        )

        selected = []
        seen = set()
        half = max(1, limit // 2)
        for pool in (potential_sorted[:half], structural_sorted[: limit - half]):
            for rot in pool:
                sig = self._rotation_signature(rot)
                if sig in seen:
                    continue
                selected.append(rot)
                seen.add(sig)
                if len(selected) >= limit:
                    return selected

        for rot in potential_sorted:
            sig = self._rotation_signature(rot)
            if sig in seen:
                continue
            selected.append(rot)
            seen.add(sig)
            if len(selected) >= limit:
                break
        return selected

    def _generate_seed_routes(
        self,
        residual_demand: Dict[Tuple[str, str], float],
    ) -> List[Rotation]:
        """
        Generate hub-oriented seed routes that complement AUX candidates.

        These seeded routes are intentionally simple: they expose promising
        hub/butterfly topologies to the metaheuristic when AUX tends to return
        local short cycles.
        """
        port_totals = {}
        bilateral = {}
        for (o, d), qty in residual_demand.items():
            if qty <= 0:
                continue
            port_totals[o] = port_totals.get(o, 0.0) + qty
            port_totals[d] = port_totals.get(d, 0.0) + qty
            bilateral[(o, d)] = bilateral.get((o, d), 0.0) + qty
            bilateral[(d, o)] = bilateral.get((d, o), 0.0) + qty

        hubs = [port for port, _ in sorted(port_totals.items(), key=lambda item: item[1], reverse=True)[:2]]
        seeds = []
        seen = set()

        for vessel in self.vessels:
            speed = float(round(vessel.design_speed))
            max_eta = min(vessel.quantity, 3)
            for eta in range(1, max_eta + 1):
                for hub in hubs:
                    spokes = [
                        port for port, _ in sorted(
                            ((port, bilateral.get((hub, port), 0.0)) for port in self.all_ports if port != hub),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                        if port != hub and bilateral.get((hub, port), 0.0) > 0
                    ][:4]

                    candidate_paths = []
                    if len(spokes) >= 2:
                        candidate_paths.append([hub, spokes[0], spokes[1]])
                    if len(spokes) >= 3:
                        candidate_paths.append([hub, spokes[0], spokes[1], spokes[2]])
                    if len(spokes) >= 4:
                        candidate_paths.append([hub, spokes[0], spokes[1], hub, spokes[2], spokes[3]])

                    for path in candidate_paths:
                        if not self._frequency_feasible(vessel, speed, eta, path):
                            continue
                        butterfly_port = hub if path.count(hub) > 1 else None
                        rot = Rotation(
                            id=f"seed_{vessel.name}_{eta}_{hub}_{len(path)}",
                            vessel_class=vessel,
                            speed=speed,
                            num_vessels=eta,
                            port_calls=path,
                            is_butterfly=butterfly_port is not None,
                            butterfly_port=butterfly_port,
                        )
                        sig = self._rotation_signature(rot)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        seeds.append(rot)
        return seeds

    def _frequency_feasible(
        self,
        vessel: VesselClass,
        speed: float,
        num_vessels: int,
        port_calls: List[str],
    ) -> bool:
        if len(port_calls) < 2:
            return False
        round_trip = self.cost_calc.rotation_round_trip_time(vessel, speed, port_calls, self.dist_min)
        weekly_upper = 7.0 * num_vessels * 1.1
        biweekly_upper = 14.0 * num_vessels * 1.1
        if vessel.capacity >= 1200:
            return round_trip <= weekly_upper
        return round_trip <= biweekly_upper
