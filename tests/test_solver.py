"""
求解器测试 — 验证模型和算法在 Baltic 实例上的正确性。
运行: python -m tests.test_solver
"""

import sys
import time


def test_cost_calculator():
    """测试成本计算公式。"""
    from src.utils.cost_calculator import VesselClass, CostCalculator, CostConfig

    v = VesselClass(
        name="Feeder_450", capacity=450, tc_rate=5000, draft=8.0,
        min_speed=10, max_speed=14, design_speed=12.0,
        fuel_design=18.8, fuel_idle=2.4,
        panama_fee=64800, suez_fee=175769, quantity=38,
    )
    calc = CostCalculator(CostConfig(fuel_price=600))

    # 公式 (1): F(s) = (s/v*)^3 * f*
    assert abs(v.fuel_consumption_per_day(12.0) - 18.8) < 0.01, "设计速度燃油不正确"
    assert abs(v.fuel_consumption_per_day(10.0) - 18.8 * (10/12)**3) < 0.01, "慢速燃油不正确"
    assert abs(v.fuel_consumption_per_day(14.0) - 18.8 * (14/12)**3) < 0.01, "快速燃油不正确"

    # TC 费用
    tc = calc.tc_cost(v, 3)
    assert tc == 5000 * 180 * 3, "TC 费用不正确"

    # 拒运惩罚
    penalty = calc.reject_penalty(100)
    assert penalty == 100000, "拒运惩罚不正确"

    print("  [OK] 成本计算公式验证通过")


def test_data_structures():
    """测试数据结构构建。"""
    from src.data_reader import load_instance, load_distances
    from src.utils.cost_calculator import build_vessels_from_data, build_ports_from_data, build_distance_dict

    data = load_instance("Baltic")
    vessels = build_vessels_from_data(data["fleet"])
    ports = build_ports_from_data(data["ports_all"])
    dist_df = load_distances()
    dist_full, dist_min, canal_info = build_distance_dict(dist_df)

    assert len(vessels) == 2, f"Baltic 应有 2 种船型, 实际 {len(vessels)}"
    assert vessels[0].name == "Feeder_450"
    assert vessels[1].name == "Feeder_800"
    assert len(ports) > 0, "港口字典为空"
    assert "DEBRV" in ports, "Bremerhaven 不在港口字典中"
    assert len(dist_min) > 0, "距离字典为空"

    print(f"  [OK] Baltic: {len(vessels)} 船型, {len(ports)} 港口, {len(dist_min)} 距离对")


def test_rotation():
    """测试航线数据结构。"""
    from src.utils.cost_calculator import VesselClass
    from src.utils.network_builder import Rotation

    v = VesselClass(
        name="Test", capacity=450, tc_rate=5000, draft=8.0,
        min_speed=10, max_speed=14, design_speed=12.0,
        fuel_design=18.8, fuel_idle=2.4,
        panama_fee=0, suez_fee=0, quantity=10,
    )
    rot = Rotation(id="r1", vessel_class=v, speed=12.0, num_vessels=2,
                   port_calls=["A", "B", "C"])

    edges = rot.edges
    assert edges == [("A", "B"), ("B", "C"), ("C", "A")], f"边列表不正确: {edges}"

    triples = rot.triples
    assert triples == [("C", "A", "B"), ("A", "B", "C"), ("B", "C", "A")], f"三元组不正确: {triples}"

    print("  [OK] Rotation 数据结构正确")


def test_solver_baltic():
    """测试在 Baltic 实例上的完整求解。"""
    from src.algorithm.solver import solve_instance

    print("  正在求解 Baltic 实例 (60s)...")
    result = solve_instance("Baltic", method="metaheuristic", max_time=60, verbose=False)

    assert result.status != "Error", f"求解出错: {result.status}"
    assert result.solve_time > 0, "求解时间为 0"

    bd = result.cost_breakdown
    print(f"  Status: {result.status}")
    print(f"  Objective: {result.objective:,.0f}")
    print(f"  Rotations: {len(result.rotations)}")
    print(f"  Service rate: {result.flow_summary.get('service_rate', 0):.1%}")
    print(f"  Z={bd.Z:,.0f}, Q={bd.Q:,.0f}, c_v={bd.c_v:,.0f}, c_b={bd.c_b:,.0f}")
    print(f"  c_p={bd.c_p:,.0f}, c_c={bd.c_c:,.0f}, c_m={bd.c_m:,.0f}, c_t={bd.c_t:,.0f}")
    print(f"  L_v={bd.L_v:,.0f}, F={bd.F:,.0f} FFE")
    print(f"  Solve time: {result.solve_time:.1f}s")

    # 论文 Table 9 Baltic Base 参考值:
    # Z = -8,365 (目标为负, 表示盈利)
    # Q = 98,310 (收入约 98k)
    print(f"\n  论文参考值 (Baltic Base): Z=-8,365, Q=98,310")
    print(f"  当前结果: Z={bd.Z:,.0f}")

    for rd in result.rotation_details:
        print(f"    Route {rd['id']}: {rd['vessel_class']}, {rd['speed']}kn, "
              f"{rd['num_vessels']}v, {' -> '.join(rd['port_calls'])}, "
              f"{rd['round_trip_days']}d ({rd['frequency']})")

    print("  [OK] Baltic 求解完成")


if __name__ == "__main__":
    print("=" * 60)
    print("LSND 求解器测试")
    print("=" * 60)

    failures = []
    tests = [
        ("成本计算公式", test_cost_calculator),
        ("数据结构构建", test_data_structures),
        ("Rotation 数据结构", test_rotation),
        ("Baltic 实例求解", test_solver_baltic),
    ]

    for name, fn in tests:
        print(f"\n[测试] {name}...")
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {e}")
            import traceback
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 60)
    if failures:
        print(f"失败: {', '.join(failures)}")
        sys.exit(1)
    else:
        print("全部测试通过!")
    print("=" * 60)
