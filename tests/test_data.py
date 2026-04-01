"""
数据读取与可视化模块测试脚本
运行: python -m tests.test_data
"""

import sys
from pathlib import Path

def test_imports():
    import pandas as pd
    import plotly.graph_objects as go
    print("  [OK] pandas, plotly 导入成功")

def test_all_instances():
    from src.data_reader import load_instance, load_distances, INSTANCES

    print("\n[测试] 读取所有实例...")
    for name in INSTANCES:
        data = load_instance(name)
        ports = data["ports"]
        demand = data["demand"]
        fleet = data["fleet"]

        assert len(ports) > 0, f"{name}: 港口数据为空"
        assert len(demand) > 0, f"{name}: 需求数据为空"
        assert len(fleet) > 0, f"{name}: 船队数据为空"
        assert "Longitude" in ports.columns, f"{name}: 缺少 Longitude 列"
        assert "Latitude" in ports.columns, f"{name}: 缺少 Latitude 列"
        assert "FFEPerWeek" in demand.columns, f"{name}: 缺少 FFEPerWeek 列"
        assert "Capacity FFE" in fleet.columns, f"{name}: 缺少 Capacity FFE 列"

        total_ffe = demand["FFEPerWeek"].sum()
        total_cap = (fleet["Capacity FFE"] * fleet["Quantity"]).sum()
        print(f"  [OK] {name:15s} | 港口:{len(ports):3d} | OD对:{len(demand):5d} | "
              f"总需求:{total_ffe:8,.0f} FFE/周 | 总运力:{total_cap:8,.0f} FFE | "
              f"供需比:{total_cap/total_ffe:.2f}")

def test_distances():
    from src.data_reader import load_distances
    print("\n[测试] 读取距离矩阵...")
    dist = load_distances()
    assert len(dist) > 0
    assert "FromUNLOCODE" in dist.columns
    assert "Distance" in dist.columns
    print(f"  [OK] 距离记录数: {len(dist):,}, 最远距离: {dist['Distance'].max():.0f} 海里")

def test_plotly_figure():
    from src.data_reader import load_instance
    import plotly.graph_objects as go
    print("\n[测试] 生成 Plotly 地图图形...")
    data = load_instance("Baltic")
    ports = data["ports"]
    fig = go.Figure(go.Scattergeo(
        lon=ports["Longitude"], lat=ports["Latitude"],
        mode="markers", text=ports["name"],
    ))
    fig.update_geos(projection_type="natural earth")
    fig_dict = fig.to_dict()
    assert "data" in fig_dict
    print(f"  [OK] Baltic 实例地图生成成功，包含 {len(ports)} 个港口标记")

if __name__ == "__main__":
    print("=" * 60)
    print("LSND 数据读取与可视化 — 功能测试")
    print("=" * 60)

    failures = []
    tests = [
        ("依赖库导入", test_imports),
        ("所有实例读取", test_all_instances),
        ("距离矩阵读取", test_distances),
        ("Plotly 图形生成", test_plotly_figure),
    ]

    for name, fn in tests:
        print(f"\n[测试] {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {e}")
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"失败: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("全部测试通过！")
        print("=" * 60)
