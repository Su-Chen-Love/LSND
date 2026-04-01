"""
成本计算模块 — 严格实现 Brouer et al. (2014) 论文中的所有成本公式。

参考: Transportation Science 48(2), pp. 281-312
"""

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional


@dataclass
class VesselClass:
    """船型类，对应论文 Table 3 的船型规格。"""
    name: str
    capacity: int           # cF: FFE 容量
    tc_rate: float          # TF: 日租金 (USD/day)
    draft: float            # deltaF: 吃水
    min_speed: float        # sF_min: 最低平均速度 (knots)
    max_speed: float        # sF_max: 最高平均速度 (knots)
    design_speed: float     # vF*: 设计速度 (knots)
    fuel_design: float      # fF*: 设计速度下日燃油消耗 (ton/day)
    fuel_idle: float        # fF0: 停泊时日燃油消耗 (ton/day)
    panama_fee: float       # pF: 巴拿马运河费
    suez_fee: float         # sF: 苏伊士运河费
    quantity: int           # qF: 可用船舶数量

    def fuel_consumption_per_day(self, speed: float) -> float:
        """
        计算给定速度下的日燃油消耗量 (ton/day)。
        论文公式 (1): F(s) = (s / v*)^3 * f*
        """
        return (speed / self.design_speed) ** 3 * self.fuel_design

    def fuel_consumption_per_mile(self, speed: float) -> float:
        """
        计算给定速度下每海里的燃油消耗量 (ton/nm)。
        g(v,s) = F(s) / (s * 24)  [将 ton/day 转为 ton/nm]
        """
        daily = self.fuel_consumption_per_day(speed)
        return daily / (speed * 24.0)


@dataclass
class Port:
    """港口，对应论文 Table 2 的港口数据。"""
    code: str               # UNLOCODE
    name: str
    longitude: float
    latitude: float
    draft: float            # 最大吃水
    move_cost: float        # mP: 装卸费 (USD/FFE)
    transship_cost: float   # tP: 转运费 (USD/FFE)
    fixed_call_cost: float  # fP: 固定靠港费 (USD)
    var_call_cost: float    # vP: 变动靠港费 (USD/FFE capacity)


@dataclass
class CostConfig:
    """成本参数配置。"""
    fuel_price: float = 600.0           # e: 燃油价格 (USD/ton)
    planning_horizon: int = 180         # T: 规划期 (天)
    reject_penalty: float = 1000.0      # q_tilde: 拒运惩罚 (USD/FFE)
    port_stay_hours: float = 24.0       # pv_j: 停泊时间 (hours)，论文设为所有港口和船型均为24h


class CostCalculator:
    """
    成本计算器，实现论文 §6.1 中目标函数的所有成本分项。
    """

    def __init__(self, config: CostConfig = None):
        self.config = config or CostConfig()

    def tc_cost(self, vessel: VesselClass, num_vessels: int) -> float:
        """
        船舶 TC 费用 (目标函数 2a 的第一项)。
        f_v * Y_r = TC_rate * T * Y_r
        """
        return vessel.tc_rate * self.config.planning_horizon * num_vessels

    def charter_out_revenue(self, vessel: VesselClass, unused_vessels: int) -> float:
        """
        未使用船舶的租出收入 (目标函数 2a 的第二项)。
        f_tilde_v * (z_v - sum Y_r)
        论文假设: 未使用船舶收入 = TC_rate * T (即 f_v = f_tilde_v)
        """
        return vessel.tc_rate * self.config.planning_horizon * unused_vessels

    def sailing_fuel_cost(self, vessel: VesselClass, speed: float,
                          distance: float) -> float:
        """
        航行燃油成本 (目标函数 2b 部分)。
        e * g(v,s) * l(v,i,j)
        """
        fuel_per_mile = vessel.fuel_consumption_per_mile(speed)
        return self.config.fuel_price * fuel_per_mile * distance

    def port_idle_fuel_cost(self, vessel: VesselClass) -> float:
        """
        停泊燃油成本 (目标函数 2b 部分)。
        e * h(v) * p(v,j)
        h(v) = idle consumption (ton/day), p(v,j) = 24h = 1 day
        """
        port_stay_days = self.config.port_stay_hours / 24.0
        return self.config.fuel_price * vessel.fuel_idle * port_stay_days

    def port_call_cost(self, vessel: VesselClass, port: Port) -> float:
        """
        靠港费用 (目标函数 2b 部分)。
        d(v,j) = FixedPortCallCost + VarPortCallCost * capacity
        """
        return port.fixed_call_cost + port.var_call_cost * vessel.capacity

    def canal_cost(self, vessel: VesselClass,
                   is_panama: bool, is_suez: bool) -> float:
        """
        运河费用 (目标函数 2b 部分)。
        a(v,i,j) = panama_fee (if Panama) + suez_fee (if Suez)
        """
        cost = 0.0
        if is_panama and not math.isnan(vessel.panama_fee):
            cost += vessel.panama_fee
        if is_suez and not math.isnan(vessel.suez_fee):
            cost += vessel.suez_fee
        return cost

    def move_cost(self, port: Port, num_ffe: float) -> float:
        """
        装卸费用 (目标函数 2d 的 u_j 部分)。
        u_j * num_ffe
        """
        return port.move_cost * num_ffe

    def transshipment_cost(self, port: Port, num_ffe: float) -> float:
        """
        转运费用 (目标函数 2d 的 t_j 部分)。
        t_j * num_ffe
        """
        return port.transship_cost * num_ffe

    def reject_penalty(self, num_ffe: float) -> float:
        """
        拒运惩罚 (目标函数 2c 的 q_tilde 部分)。
        q_tilde * O_od
        """
        return self.config.reject_penalty * num_ffe

    def rotation_round_trip_time(self, vessel: VesselClass, speed: float,
                                 port_calls: list,
                                 distances: Dict[Tuple[str, str], float]) -> float:
        """
        计算航线一个完整轮次的时间 (天)。
        T_rot = sum(l(v,i,j) / (s * 24) + p(v,j) / 24) for each leg
        """
        total_hours = 0.0
        for k in range(len(port_calls)):
            i = port_calls[k]
            j = port_calls[(k + 1) % len(port_calls)]
            dist = distances.get((i, j), 0.0)
            sailing_hours = dist / speed
            total_hours += sailing_hours + self.config.port_stay_hours
        return total_hours / 24.0

    def num_round_trips(self, round_trip_days: float) -> float:
        """
        规划期内的轮次数。
        m_r = T / round_trip_days
        论文不要求整数轮次。
        """
        if round_trip_days <= 0:
            return 0.0
        return self.config.planning_horizon / round_trip_days

    def num_vessels_for_weekly(self, round_trip_days: float) -> int:
        """
        满足周频所需的船舶数。
        Y_r = ceil(round_trip_days / 7)
        """
        return max(1, math.ceil(round_trip_days / 7.0))

    def num_vessels_for_biweekly(self, round_trip_days: float) -> int:
        """
        满足双周频所需的船舶数。
        Y_r = ceil(round_trip_days / 14)
        """
        return max(1, math.ceil(round_trip_days / 14.0))

    def rotation_operating_cost(self, vessel: VesselClass, speed: float,
                                port_sequence: list,
                                ports_dict: Dict[str, Port],
                                distances: Dict[Tuple[str, str], float],
                                canal_info: Dict[Tuple[str, str], Tuple[bool, bool]]) -> dict:
        """
        计算一条航线单次航行的全部运营成本。

        返回各项成本分解:
        - fuel_sailing: 航行燃油
        - fuel_idle: 停泊燃油
        - port_call: 靠港费
        - canal: 运河费
        """
        fuel_sailing = 0.0
        fuel_idle = 0.0
        port_call_total = 0.0
        canal_total = 0.0

        for k in range(len(port_sequence)):
            i = port_sequence[k]
            j = port_sequence[(k + 1) % len(port_sequence)]

            dist = distances.get((i, j), 0.0)
            fuel_sailing += self.sailing_fuel_cost(vessel, speed, dist)

            port = ports_dict.get(j)
            if port:
                fuel_idle += self.port_idle_fuel_cost(vessel)
                port_call_total += self.port_call_cost(vessel, port)

            is_panama, is_suez = canal_info.get((i, j), (False, False))
            canal_total += self.canal_cost(vessel, is_panama, is_suez)

        return {
            "fuel_sailing": fuel_sailing,
            "fuel_idle": fuel_idle,
            "port_call": port_call_total,
            "canal": canal_total,
            "total": fuel_sailing + fuel_idle + port_call_total + canal_total,
        }


def build_vessels_from_data(fleet_full: pd.DataFrame) -> list:
    """从 DataFrame 构建 VesselClass 对象列表。"""
    vessels = []
    for _, row in fleet_full.iterrows():
        v = VesselClass(
            name=row["Vessel class"],
            capacity=int(row["Capacity FFE"]),
            tc_rate=float(row["TC rate daily (fixed Cost)"]),
            draft=float(row["draft"]),
            min_speed=float(row["minSpeed"]),
            max_speed=float(row["maxSpeed"]),
            design_speed=float(row["designSpeed"]),
            fuel_design=float(row["Bunker ton per day at designSpeed"]),
            fuel_idle=float(row["Idle Consumption ton/day"]),
            panama_fee=float(row.get("panamaFee", float("nan"))),
            suez_fee=float(row.get("suezFee", float("nan"))),
            quantity=int(row["Quantity"]),
        )
        vessels.append(v)
    return vessels


def build_ports_from_data(ports_df: pd.DataFrame) -> Dict[str, Port]:
    """从 DataFrame 构建 Port 对象字典 (code -> Port)。"""
    ports = {}
    for _, row in ports_df.iterrows():
        p = Port(
            code=row["UNLocode"],
            name=row["name"],
            longitude=float(row["Longitude"]),
            latitude=float(row["Latitude"]),
            draft=float(row["Draft"]),
            move_cost=float(row["CostPerFULL"]),
            transship_cost=float(row["CostPerFULLTrnsf"]),
            fixed_call_cost=float(row["PortCallCostFixed"]),
            var_call_cost=float(row["PortCallCostPerFFE"]),
        )
        ports[p.code] = p
    return ports


def build_distance_dict(dist_df: pd.DataFrame) -> Tuple[
    Dict[Tuple[str, str, float], float],
    Dict[Tuple[str, str], float],
    Dict[Tuple[str, str], Tuple[bool, bool]],
]:
    """
    从距离 DataFrame 构建查找字典。

    返回:
    - dist_full: (from, to, draft) -> distance  完整距离字典
    - dist_min: (from, to) -> min_distance  每对港口的最短距离
    - canal_info: (from, to) -> (is_panama, is_suez)  基于最短路径的运河信息
    """
    dist_full = {}
    dist_min = {}
    canal_info = {}

    for _, row in dist_df.iterrows():
        from_port = row["FromUNLOCODE"]
        to_port = row["ToUNLOCODE"]
        distance = row["Distance"]
        draft = row.get("Draft", float("inf"))
        is_panama = bool(row.get("IsPanama", 0))
        is_suez = bool(row.get("IsSuez", 0))

        if pd.isna(draft):
            draft = float("inf")

        dist_full[(from_port, to_port, draft)] = distance

        key = (from_port, to_port)
        if key not in dist_min or distance < dist_min[key]:
            dist_min[key] = distance
            canal_info[key] = (is_panama, is_suez)

    return dist_full, dist_min, canal_info


def get_distance_for_vessel(dist_full: dict, from_port: str, to_port: str,
                            vessel_draft: float) -> Tuple[float, bool, bool]:
    """
    获取特定船型（受吃水限制）从 from_port 到 to_port 的最短距离。
    返回 (distance, is_panama, is_suez)。
    """
    best_dist = float("inf")
    best_panama = False
    best_suez = False

    for (fp, tp, draft), dist in dist_full.items():
        if fp == from_port and tp == to_port:
            if draft >= vessel_draft or draft == float("inf"):
                if dist < best_dist:
                    best_dist = dist
                    # 查找对应的运河信息
                    best_panama = False
                    best_suez = False

    # 如果使用 dist_full 找不到，退化到无吃水限制
    if best_dist == float("inf"):
        for (fp, tp, draft), dist in dist_full.items():
            if fp == from_port and tp == to_port:
                if dist < best_dist:
                    best_dist = dist

    return best_dist, best_panama, best_suez
