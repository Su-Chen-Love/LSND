"""
多商品流网络图构建模块 — 实现论文 §6.2 中描述的 MCF 图结构。

构建 G=(N, A):
- N = N_T ∪ N_C^r  (港口终端节点 + 航线停靠节点)
- A = A_l ∪ A_u ∪ A_t ∪ A_s ∪ A_o  (装货/卸货/转运/航行/拒绝边)

参考: Brouer et al. (2014), Transportation Science 48(2), Figure 7
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional
from src.utils.cost_calculator import VesselClass, Port, CostCalculator, CostConfig


@dataclass
class Rotation:
    """
    一条航线 (rotation)，对应论文中的 r ∈ R。
    """
    id: str                          # 航线标识
    vessel_class: VesselClass        # vr: 使用的船型
    speed: float                     # sr: 航行速度
    num_vessels: int                 # Yr: 分配的船舶数
    port_calls: List[str]            # 有序的港口调用序列
    is_butterfly: bool = False       # 是否为蝴蝶航线
    butterfly_port: Optional[str] = None  # 蝴蝶节点

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """航线中的所有边 (i,j)，即 E_r。"""
        edges = []
        for k in range(len(self.port_calls)):
            i = self.port_calls[k]
            j = self.port_calls[(k + 1) % len(self.port_calls)]
            edges.append((i, j))
        return edges

    @property
    def triples(self) -> List[Tuple[str, str, str]]:
        """
        航线中的有序港口三元组 (h,i,j)，即 Δ_r。
        用于蝴蝶航线的流量守恒。
        """
        triples = []
        n = len(self.port_calls)
        for k in range(n):
            h = self.port_calls[(k - 1) % n]
            i = self.port_calls[k]
            j = self.port_calls[(k + 1) % n]
            triples.append((h, i, j))
        return triples


@dataclass
class MCFEdge:
    """多商品流图中的一条边。"""
    source: str          # 源节点 ID
    target: str          # 目标节点 ID
    edge_type: str       # 'load', 'unload', 'transship', 'sail', 'omit'
    cost: float          # 边成本 (每 FFE)
    capacity: float      # 边容量
    rotation_id: str = ""     # 所属航线 (仅 sail/load/unload)
    port: str = ""            # 对应的港口代码


class MCFNetwork:
    """
    多商品流网络图，对应论文 Figure 7。

    节点类型:
    - terminal:{port}  — 港口终端节点 N_T
    - call:{rotation}:{port}:{seq}  — 航线停靠节点 N_C^r

    边类型:
    - load: 从终端节点到航线节点 (A_l)
    - unload: 从航线节点到终端节点 (A_u)
    - transship: 航线间转运 (A_t)
    - sail: 航线内航行边 (A_s)
    - omit: 拒绝需求 (A_o)
    """

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: List[MCFEdge] = []
        self.terminal_nodes: Dict[str, str] = {}  # port_code -> node_id
        self.call_nodes: Dict[str, List[str]] = {}  # port_code -> [node_ids]

    def add_terminal_node(self, port_code: str) -> str:
        node_id = f"terminal:{port_code}"
        self.nodes.add(node_id)
        self.terminal_nodes[port_code] = node_id
        if port_code not in self.call_nodes:
            self.call_nodes[port_code] = []
        return node_id

    def add_call_node(self, rotation_id: str, port_code: str, seq: int) -> str:
        node_id = f"call:{rotation_id}:{port_code}:{seq}"
        self.nodes.add(node_id)
        if port_code not in self.call_nodes:
            self.call_nodes[port_code] = []
        self.call_nodes[port_code].append(node_id)
        return node_id

    def add_edge(self, edge: MCFEdge):
        self.edges.append(edge)


def build_mcf_network(
    rotations: List[Rotation],
    ports_dict: Dict[str, Port],
    demands: List[Tuple[str, str, float, float]],
    distances: Dict[Tuple[str, str], float],
    cost_calc: CostCalculator,
) -> MCFNetwork:
    """
    根据当前航线集合 R_u 构建多商品流图。

    参数:
    - rotations: 当前航线集合
    - ports_dict: 港口字典 (code -> Port)
    - demands: 需求列表 [(origin, dest, quantity, revenue), ...]
    - distances: 距离字典 (from, to) -> distance
    - cost_calc: 成本计算器

    返回:
    - MCFNetwork: 构建好的网络图
    """
    network = MCFNetwork()

    # 1. 创建所有涉及港口的终端节点
    all_ports = set()
    for rot in rotations:
        all_ports.update(rot.port_calls)
    for o, d, _, _ in demands:
        all_ports.add(o)
        all_ports.add(d)

    for port_code in all_ports:
        network.add_terminal_node(port_code)

    # 2. 为每条航线创建停靠节点和航行边
    for rot in rotations:
        call_node_ids = []
        for seq, port_code in enumerate(rot.port_calls):
            node_id = network.add_call_node(rot.id, port_code, seq)
            call_node_ids.append((node_id, port_code))

        # 航行边 A_s: 连接航线内相邻停靠节点 (成本为 0，因为运营成本计入固定成本)
        n = len(call_node_ids)
        for k in range(n):
            src_id, src_port = call_node_ids[k]
            tgt_id, tgt_port = call_node_ids[(k + 1) % n]
            capacity = rot.vessel_class.capacity * rot.num_vessels
            round_trip_days = cost_calc.rotation_round_trip_time(
                rot.vessel_class, rot.speed, rot.port_calls, distances
            )
            m_r = cost_calc.num_round_trips(round_trip_days)
            total_capacity = rot.vessel_class.capacity * m_r * rot.num_vessels

            network.add_edge(MCFEdge(
                source=src_id, target=tgt_id,
                edge_type="sail", cost=0.0,
                capacity=total_capacity,
                rotation_id=rot.id, port=tgt_port,
            ))

        # 装货边 A_l 和 卸货边 A_u: 连接终端节点和航线停靠节点
        for node_id, port_code in call_node_ids:
            port = ports_dict.get(port_code)
            if not port:
                continue
            terminal_id = network.terminal_nodes[port_code]
            capacity = rot.vessel_class.capacity

            # 装货边: terminal -> call (成本 = 装货费 u_j)
            network.add_edge(MCFEdge(
                source=terminal_id, target=node_id,
                edge_type="load", cost=port.move_cost,
                capacity=capacity,
                rotation_id=rot.id, port=port_code,
            ))

            # 卸货边: call -> terminal (成本 = 卸货费 u_j)
            network.add_edge(MCFEdge(
                source=node_id, target=terminal_id,
                edge_type="unload", cost=port.move_cost,
                capacity=capacity,
                rotation_id=rot.id, port=port_code,
            ))

    # 3. 转运边 A_t: 连接同一港口不同航线的停靠节点
    for port_code, call_nodes in network.call_nodes.items():
        port = ports_dict.get(port_code)
        if not port:
            continue
        for i, src_node in enumerate(call_nodes):
            for j, tgt_node in enumerate(call_nodes):
                if i != j:
                    # 提取航线 ID
                    src_rot = src_node.split(":")[1]
                    tgt_rot = tgt_node.split(":")[1]
                    # 转运边 (也包括蝴蝶航线内的 r=s 情况)
                    network.add_edge(MCFEdge(
                        source=src_node, target=tgt_node,
                        edge_type="transship", cost=port.transship_cost,
                        capacity=float("inf"),
                        rotation_id=f"{src_rot}->{tgt_rot}",
                        port=port_code,
                    ))

    # 4. 拒绝边 A_o: 为每个需求创建直接拒绝边
    for o, d, quantity, revenue in demands:
        src = network.terminal_nodes.get(o, f"terminal:{o}")
        tgt = network.terminal_nodes.get(d, f"terminal:{d}")
        network.add_edge(MCFEdge(
            source=src, target=tgt,
            edge_type="omit",
            cost=cost_calc.config.reject_penalty,
            capacity=quantity,
        ))

    return network
