"""
LINERLIB 班轮航运网络数据读取模块
读取 LINERLIB benchmark suite 中的港口、距离、船队和需求数据。
"""

import os
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "LinerLib"

INSTANCES = ["Baltic", "EuropeAsia", "Mediterranean", "Pacific", "WAF", "WorldLarge", "WorldSmall"]


def load_ports() -> pd.DataFrame:
    """读取港口数据，包含坐标、成本等信息。"""
    df = pd.read_csv(DATA_DIR / "ports.csv", sep="\t")
    df.columns = df.columns.str.strip()
    for col in ["Longitude", "Latitude", "Draft"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_distances() -> pd.DataFrame:
    """读取港口间距离矩阵 (dist_dense.csv)。"""
    df = pd.read_csv(DATA_DIR / "dist_dense.csv", sep="\t")
    df.columns = df.columns.str.strip()
    df.rename(columns={"fromUNLOCODe": "FromUNLOCODE"}, inplace=True)
    df["Distance"] = pd.to_numeric(df["Distance"], errors="coerce")
    return df


def load_fleet_data() -> pd.DataFrame:
    """读取船型参数 (6 种船型的详细规格)。"""
    df = pd.read_csv(DATA_DIR / "fleet_data.csv", sep="\t")
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    return df


def load_fleet(instance: str) -> pd.DataFrame:
    """读取指定实例的可用船队配置。"""
    path = DATA_DIR / f"fleet_{instance}.csv"
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")
    return df


def load_demand(instance: str) -> pd.DataFrame:
    """读取指定实例的需求数据 (OD 对、运量、收入、时限)。"""
    path = DATA_DIR / f"Demand_{instance}.csv"
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip()
    df["FFEPerWeek"] = pd.to_numeric(df["FFEPerWeek"], errors="coerce")
    return df


def load_instance(instance: str) -> dict:
    """加载一个完整实例的所有数据。"""
    ports = load_ports()
    demand = load_demand(instance)
    fleet = load_fleet(instance)
    fleet_specs = load_fleet_data()

    fleet_full = fleet.merge(fleet_specs, on="Vessel class", how="left")

    involved_codes = set(demand["Origin"].unique()) | set(demand["Destination"].unique())
    ports_involved = ports[ports["UNLocode"].isin(involved_codes)].copy()

    return {
        "name": instance,
        "ports_all": ports,
        "ports": ports_involved,
        "demand": demand,
        "fleet": fleet_full,
        "fleet_config": fleet,
        "fleet_specs": fleet_specs,
    }


if __name__ == "__main__":
    for inst in INSTANCES:
        data = load_instance(inst)
        print(f"[{inst}] 港口: {len(data['ports'])}, "
              f"OD需求对: {len(data['demand'])}, "
              f"船型: {len(data['fleet'])}")
