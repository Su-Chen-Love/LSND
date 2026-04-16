"""
禁忌搜索组件 — 实现论文 §6.2 中的禁忌搜索策略。

管理禁忌列表、回溯机制和航线删除策略。

参考: Brouer et al. (2014), Transportation Science 48(2), pp. 299-301
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from src.utils.network_builder import Rotation


@dataclass
class TabuConfig:
    """禁忌搜索配置。"""
    tabu_tenure: int = 10                 # 禁忌期限 (迭代数)
    max_non_improving: int = 5            # 最大无改进迭代数，触发回溯
    backtrack_delete_base: int = 1        # 回溯时删除的基础航线数
    backtrack_delete_growth: float = 0.5  # 回溯次数增长时的额外删除比例


class TabuList:
    """禁忌列表，防止最近添加的航线被删除。"""

    def __init__(self, tenure: int = 10):
        self.tenure = tenure
        self._entries: Dict[str, int] = {}  # rotation_id -> expiry_iteration

    def add(self, rotation_id: str, current_iter: int):
        self._entries[rotation_id] = current_iter + self.tenure

    def is_tabu(self, rotation_id: str, current_iter: int) -> bool:
        if rotation_id not in self._entries:
            return False
        return current_iter < self._entries[rotation_id]

    def cleanup(self, current_iter: int):
        expired = [k for k, v in self._entries.items() if v <= current_iter]
        for k in expired:
            del self._entries[k]


class BacktrackManager:
    """
    回溯管理器。

    论文策略:
    - 若连续 max_non_improving 次无改进，回溯到最优解 R*
    - 从 R* 中删除部分航线 (根据三个标准)
    - 删除数量随回溯到同一 R* 的次数增加而增加
    """

    def __init__(self, config: TabuConfig = None):
        self.config = config or TabuConfig()
        self.best_rotations: List[Rotation] = []
        self.best_objective: float = float("inf")
        self.non_improving_count: int = 0
        self.backtrack_count: int = 0

    def update_best(self, rotations: List[Rotation], objective: float) -> bool:
        """更新最优解。返回是否有改进。"""
        if objective < self.best_objective:
            self.best_rotations = list(rotations)
            self.best_objective = objective
            self.non_improving_count = 0
            self.backtrack_count = 0
            return True
        else:
            self.non_improving_count += 1
            return False

    def should_backtrack(self) -> bool:
        return self.non_improving_count >= self.config.max_non_improving

    def backtrack(self, rotation_scores: Dict[str, float]) -> List[Rotation]:
        """
        执行回溯: 返回到 R* 并删除评分最差的航线。

        参数:
        - rotation_scores: 航线评分 (越高越差)
          评分标准: 空 FFE-miles、低利用率、MIP gap

        返回:
        - 回溯后的航线集合
        """
        self.backtrack_count += 1
        self.non_improving_count = 0

        # 确定删除数量
        n_delete = self.config.backtrack_delete_base + int(
            self.backtrack_count * self.config.backtrack_delete_growth
        )
        n_delete = min(n_delete, max(1, len(self.best_rotations) - 1))

        # 按评分排序，删除最差的
        scored = [(rot, rotation_scores.get(rot.id, 0.0)) for rot in self.best_rotations]
        scored.sort(key=lambda x: x[1], reverse=True)

        remaining = [rot for rot, _ in scored[n_delete:]]
        return remaining


def score_rotations(
    rotations: List[Rotation],
    flow: Dict[str, Dict],
    capacities: Dict[str, float],
    sail_edge_totals: Dict[str, float] = None,
    distances: Dict = None,
) -> Dict[str, float]:
    """
    对航线评分 (用于回溯删除决策)。

    论文三个标准:
    1. 空 FFE-miles (高 = 差) — 航线的空载里程占总里程的比例
    2. 低利用率腿 (低利用率 = 差) — 利用率低于阈值的腿的比例
    3. MIP gap (高 = 差) — 近似为航线运营成本与收入的比值

    返回: rotation_id -> score (越高越差，越应该被删除)
    """
    scores = {}
    sail_edge_totals = sail_edge_totals or {}

    for rot in rotations:
        rot_flow = flow.get(rot.id, {})
        total_flow = sum(rot_flow.values())
        cap = max(capacities.get(rot.id, rot.vessel_class.capacity), 1.0)
        n_legs = max(len(rot.port_calls), 1)

        per_leg_utils = []
        for leg_idx in range(n_legs):
            edge_id = f"sail:{rot.id}:{leg_idx}"
            leg_total = sail_edge_totals.get(edge_id, 0.0)
            per_leg_utils.append(min(leg_total / cap, 1.0))

        if per_leg_utils:
            avg_util = sum(per_leg_utils) / len(per_leg_utils)
            empty_leg_ratio = sum(1.0 - u for u in per_leg_utils) / len(per_leg_utils)
            low_util_share = sum(1 for u in per_leg_utils if u < 0.25) / len(per_leg_utils)
        else:
            avg_util = 0.0
            empty_leg_ratio = 1.0
            low_util_share = 1.0

        route_length_penalty = (n_legs / 6.0) if avg_util < 0.35 else 0.0
        vessel_cost_score = min((rot.num_vessels * cap) / max(total_flow, 1.0), 3.0) / 3.0

        scores[rot.id] = (
            0.45 * empty_leg_ratio
            + 0.35 * low_util_share
            + 0.10 * min(route_length_penalty, 1.0)
            + 0.10 * vessel_cost_score
        )

    return scores
