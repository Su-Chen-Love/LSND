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

logger = logging.getLogger(__name__)


@dataclass
class MetaheuristicConfig:
    """元启发式算法配置。"""
    max_time: int = 300              # 最大运行时间 (秒)
    max_iterations: int = 200        # 最大迭代数
    aux_time_limit: int = 15         # AUX 子问题时限 (CBC 较慢，控制单个时限)
    mcfp_time_limit: int = 60        # MCFP 时限
    max_cluster_size: int = 20       # AUX 子问题最大港口数
    speed_step: int = 1              # 速度搜索步长 (细粒度搜索)
    initial_speed_narrow: bool = False  # 从第一轮就搜索全速度范围
    suppress_subtour_iters: int = 5   # 前 N 轮不启用子环消除
    progress_callback: Optional[Callable] = None  # 进度回调


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
                n_avail = available_vessels.get(v.name, 0)
                if n_avail <= 0:
                    continue

                # 循环使用不同聚类，增加航线多样性
                cluster_idx = (iteration + v_idx) % len(clusters)
                cluster = clusters[cluster_idx]
                cluster_ports = [self.ports_dict[p] for p in cluster if p in self.ports_dict]

                # 速度范围
                if cfg.initial_speed_narrow and iteration < 5:
                    speeds = [v.design_speed]
                else:
                    speeds = list(range(int(v.min_speed), int(v.max_speed) + 1, cfg.speed_step))

                vessel_counts = list(range(1, min(n_avail + 1, 4)))

                aux_cfg = AUXConfig(
                    max_ports=cfg.max_cluster_size,
                    suppress_subtour=not suppress_subtour,
                    time_limit=cfg.aux_time_limit,
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
                        result = generator.solve(
                            speed=float(s), num_vessels=eta, rotation_id=rot_id
                        )
                        if result is not None:
                            candidates.append(result)

                        # 时间检查
                        if time.time() - start_time >= cfg.max_time:
                            break
                    if time.time() - start_time >= cfg.max_time:
                        break
                if time.time() - start_time >= cfg.max_time:
                    break

            total_columns += len(candidates)
            logger.info(f"  Generated {len(candidates)} candidates")

            # --- Step 3: 评估候选航线 ---
            best_candidate = None
            best_candidate_obj = float("inf")
            best_candidate_mcfp = None

            for cand in candidates:
                # R_{u+1} = R_u ∪ {cand}
                test_rotations = current_rotations + [cand]

                mcfp = MCFPSolver(
                    rotations=test_rotations,
                    ports_dict=self.ports_dict,
                    demands=self.demands,
                    dist_min=self.dist_min,
                    config=self.cost_config,
                )

                try:
                    mcfp_result = mcfp.solve(time_limit=cfg.mcfp_time_limit)
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

                # 时间检查
                if time.time() - start_time >= cfg.max_time:
                    break

            # --- Step 4: 更新航线集合 ---
            improved = False
            if best_candidate is not None and best_candidate_obj < best_objective:
                current_rotations.append(best_candidate)
                tabu_list.add(best_candidate.id, iteration)
                best_objective = best_candidate_obj
                best_rotations = list(current_rotations)
                best_mcfp = best_candidate_mcfp
                improved = True

                # 更新残差需求
                if best_candidate_mcfp:
                    residual_demand = dict(best_candidate_mcfp.residual_demand)

                logger.info(f"  IMPROVED: obj={best_objective:.0f}, "
                            f"added {best_candidate.id}")
            elif best_candidate is not None:
                # 未改进但有候选: 仍然可能接受 (禁忌搜索的邻域探索)
                current_rotations.append(best_candidate)
                tabu_list.add(best_candidate.id, iteration)
                if best_candidate_mcfp:
                    residual_demand = dict(best_candidate_mcfp.residual_demand)

            backtrack_mgr.update_best(best_rotations, best_objective)

            # --- Step 5: 禁忌搜索 - 回溯 ---
            if backtrack_mgr.should_backtrack() and best_rotations:
                logger.info("  BACKTRACKING")
                # 计算航线评分
                flow = best_candidate_mcfp.flow if best_candidate_mcfp else {}
                caps = {rot.id: rot.vessel_class.capacity for rot in current_rotations}
                scores = score_rotations(current_rotations, flow, caps)
                current_rotations = backtrack_mgr.backtrack(scores)

                # 重新计算残差需求
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
