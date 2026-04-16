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


def test_scenario_scaling():
    """测试 Low/Base/High 场景缩放规则。"""
    from src.data_reader import load_instance

    base = load_instance("Baltic", scenario="base")["fleet"]
    low = load_instance("Baltic", scenario="low")["fleet"]
    high = load_instance("Baltic", scenario="high")["fleet"]

    assert list(base["Quantity"]) == [4, 2]
    assert list(low["Quantity"]) == [3, 2], f"low quantity mismatch: {list(low['Quantity'])}"
    assert list(high["Quantity"]) == [5, 2], f"high quantity mismatch: {list(high['Quantity'])}"
    assert low["TC rate daily (fixed Cost)"].ge(base["TC rate daily (fixed Cost)"]).all()
    assert high["TC rate daily (fixed Cost)"].le(base["TC rate daily (fixed Cost)"]).all()

    print("  [OK] 场景缩放规则正确")


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


def test_mcfp_small_fixture():
    """测试图式 MCFP 在简单单航线网络上的流量守恒。"""
    from src.model.mcfp import MCFPSolver
    from src.utils.cost_calculator import VesselClass, Port, CostConfig
    from src.utils.network_builder import Rotation

    vessel = VesselClass(
        name="Fixture", capacity=100, tc_rate=5000, draft=8.0,
        min_speed=10, max_speed=14, design_speed=12.0,
        fuel_design=18.8, fuel_idle=2.4,
        panama_fee=0, suez_fee=0, quantity=2,
    )
    ports = {
        "A": Port("A", "A", 0.0, 0.0, 10.0, 10.0, 20.0, 100.0, 0.0),
        "B": Port("B", "B", 1.0, 1.0, 10.0, 15.0, 25.0, 100.0, 0.0),
    }
    rotation = Rotation(id="r1", vessel_class=vessel, speed=12.0, num_vessels=1, port_calls=["A", "B"])
    solver = MCFPSolver(
        rotations=[rotation],
        ports_dict=ports,
        demands=[("A", "B", 50.0, 100.0)],
        dist_min={("A", "B"): 100.0, ("B", "A"): 100.0},
        config=CostConfig(planning_horizon=180, reject_penalty=1000.0),
    )
    result = solver.solve(time_limit=30)

    assert result.status == "Optimal", result.status
    assert abs(result.rejected[("A", "B")]) < 1e-6
    assert result.total_revenue == 5000.0
    assert abs(result.total_handling_cost - (10.0 + 15.0) * 50.0) < 1e-3
    assert result.flow["r1"][("A", "B")] > 0

    print("  [OK] 简单 MCFP 夹具验证通过")


def test_baltic_paper_replay():
    """论文 Baltic Base 服务回放应保持接近论文日志。"""
    from src.data_reader import load_instance, load_distances
    from src.paper_replay import evaluate_baltic_base_replay
    from src.utils.cost_calculator import build_vessels_from_data, build_ports_from_data, build_distance_dict, CostConfig

    data = load_instance("Baltic", scenario="base")
    vessels = {v.name: v for v in build_vessels_from_data(data["fleet"])}
    ports = build_ports_from_data(data["ports_all"])
    _, dist_min, canal_info = build_distance_dict(load_distances())
    weeks = 180 / 7.0
    demands = [
        (row["Origin"], row["Destination"], float(row["FFEPerWeek"]) * weeks, float(row["Revenue_1"]))
        for _, row in data["demand"].iterrows()
    ]

    replay = evaluate_baltic_base_replay(
        vessels_by_name=vessels,
        ports_dict=ports,
        demands=demands,
        dist_min=dist_min,
        canal_info=canal_info,
        cost_config=CostConfig(),
    )

    assert replay["status"] == "Optimal"
    assert replay["service_rate"] >= 0.91, replay["service_rate"]
    assert replay["objective"] < -5_500_000, replay["objective"]
    assert replay["rejected_ffe"] < 12_000, replay["rejected_ffe"]

    print("  [OK] 论文 Baltic 服务回放保持有效")


def test_aux_butterfly_extraction():
    """蝴蝶边集应被确定性提取为完整回路。"""
    from src.model.route_generation import RouteGenerator

    edges = [
        ("RULED", "FIKTK"),
        ("FIKTK", "DEBRV"),
        ("DEBRV", "RUKGD"),
        ("RUKGD", "PLGDY"),
        ("PLGDY", "DEBRV"),
        ("DEBRV", "RULED"),
    ]
    extracted = RouteGenerator.extract_route_from_active_edges(edges, fallback_butterfly={"DEBRV"})
    assert extracted is not None
    port_calls, butterfly_port = extracted
    assert butterfly_port == "DEBRV"
    assert len(port_calls) == len(edges)
    assert port_calls.count("DEBRV") == 2

    edge_multiset = {}
    for idx in range(len(port_calls)):
        i_p = port_calls[idx]
        j_p = port_calls[(idx + 1) % len(port_calls)]
        edge_multiset[(i_p, j_p)] = edge_multiset.get((i_p, j_p), 0) + 1
    for edge in edges:
        assert edge_multiset.get(edge, 0) == 1, f"missing edge {edge} in {port_calls}"

    print("  [OK] AUX 蝴蝶提取逻辑正确")


def test_aux_multiple_candidates():
    """单个 AUX 子问题应能枚举出多个去重候选。"""
    from src.data_reader import load_instance, load_distances
    from src.model.route_generation import AUXConfig, RouteGenerator
    from src.utils.cost_calculator import CostConfig, build_distance_dict, build_ports_from_data, build_vessels_from_data

    data = load_instance("Baltic", scenario="base")
    vessels = build_vessels_from_data(data["fleet"])
    ports_dict = build_ports_from_data(data["ports_all"])
    _, dist_min, canal_info = build_distance_dict(load_distances())
    residual = {}
    revenue = {}
    weeks = 180 / 7.0
    for _, row in data["demand"].iterrows():
        residual[(row["Origin"], row["Destination"])] = float(row["FFEPerWeek"]) * weeks
        revenue[(row["Origin"], row["Destination"])] = float(row["Revenue_1"])

    from src.model.route_generation import create_port_clusters

    all_ports = sorted(set([o for o, _ in residual] + [d for _, d in residual]))
    cluster = create_port_clusters(all_ports, residual, dist_min, max_cluster_size=20)[0]
    generator = RouteGenerator(
        ports=[ports_dict[p] for p in cluster],
        port_codes=cluster,
        vessel=vessels[0],
        residual_demand=residual,
        demand_revenue=revenue,
        dist_min=dist_min,
        canal_info=canal_info,
        ports_dict=ports_dict,
        config=CostConfig(),
        aux_config=AUXConfig(time_limit=10, num_solutions=3),
    )

    candidates = generator.solve_many(speed=10.0, num_vessels=1, rotation_prefix="test_aux", limit=5)
    signatures = {tuple(rot.port_calls) for rot in candidates}
    assert len(candidates) >= 2, len(candidates)
    assert len(signatures) == len(candidates)

    print("  [OK] AUX 多候选枚举正常")


def test_seed_route_id_uniqueness():
    """seed route 应有稳定唯一的 id，且签名去重后不重复。"""
    from src.data_reader import load_instance, load_distances
    from src.algorithm.metaheuristic import LsndMetaheuristic, MetaheuristicConfig
    from src.algorithm.tabu_search import TabuConfig
    from src.utils.cost_calculator import CostConfig, build_distance_dict, build_ports_from_data, build_vessels_from_data

    data = load_instance("Baltic", scenario="base")
    vessels = build_vessels_from_data(data["fleet"])
    ports_dict = build_ports_from_data(data["ports_all"])
    _, dist_min, canal_info = build_distance_dict(load_distances())
    weeks = 180 / 7.0
    demands = [
        (row["Origin"], row["Destination"], float(row["FFEPerWeek"]) * weeks, float(row["Revenue_1"]))
        for _, row in data["demand"].iterrows()
    ]
    solver = LsndMetaheuristic(
        vessels=vessels,
        ports_dict=ports_dict,
        demands=demands,
        dist_min=dist_min,
        canal_info=canal_info,
        cost_config=CostConfig(),
        meta_config=MetaheuristicConfig(max_time=5),
        tabu_config=TabuConfig(),
    )
    residual = {(o, d): k for o, d, k, _ in demands}
    pools = solver._generate_seed_routes(residual, iteration=3, prefer_butterfly=True)
    seeds = pools["seed_simple"] + pools["seed_butterfly"]

    ids = [rot.id for rot in seeds]
    signatures = [solver._rotation_signature(rot) for rot in seeds]
    assert len(ids) == len(set(ids))
    assert len(signatures) == len(set(signatures))

    print("  [OK] seed route 唯一性正常")


def test_mcfp_constraint_naming_uniqueness():
    """相同形状但不同 rotation id 的网络应能稳定求解，不出现约束名冲突。"""
    from src.model.mcfp import MCFPSolver
    from src.utils.cost_calculator import VesselClass, Port, CostConfig
    from src.utils.network_builder import Rotation

    vessel = VesselClass(
        name="Fixture", capacity=100, tc_rate=5000, draft=8.0,
        min_speed=10, max_speed=14, design_speed=12.0,
        fuel_design=18.8, fuel_idle=2.4,
        panama_fee=0, suez_fee=0, quantity=4,
    )
    ports = {
        "A": Port("A", "A", 0.0, 0.0, 10.0, 10.0, 20.0, 100.0, 0.0),
        "B": Port("B", "B", 1.0, 1.0, 10.0, 15.0, 25.0, 100.0, 0.0),
        "C": Port("C", "C", 2.0, 2.0, 10.0, 15.0, 25.0, 100.0, 0.0),
    }
    rotations = [
        Rotation(id="seed_alpha", vessel_class=vessel, speed=12.0, num_vessels=1, port_calls=["A", "B", "C"]),
        Rotation(id="seed_beta", vessel_class=vessel, speed=12.0, num_vessels=1, port_calls=["A", "B", "C"]),
    ]
    solver = MCFPSolver(
        rotations=rotations,
        ports_dict=ports,
        demands=[("A", "C", 40.0, 100.0), ("B", "A", 10.0, 80.0)],
        dist_min={
            ("A", "B"): 100.0, ("B", "C"): 120.0, ("C", "A"): 140.0,
            ("B", "A"): 100.0, ("C", "B"): 120.0, ("A", "C"): 140.0,
        },
        config=CostConfig(planning_horizon=180, reject_penalty=1000.0),
    )
    result = solver.solve(time_limit=30)

    assert result.status == "Optimal", result.status
    assert isinstance(result.sail_edge_totals, dict)

    print("  [OK] MCFP 约束命名唯一性正常")


def test_pair_move_scenarios():
    """pair 邻域应能为同船型双列加入生成最小删除组合。"""
    from src.algorithm.metaheuristic import LsndMetaheuristic, MetaheuristicConfig
    from src.algorithm.tabu_search import TabuConfig
    from src.utils.cost_calculator import VesselClass, Port, CostConfig
    from src.utils.network_builder import Rotation

    vessel = VesselClass(
        name="Fixture", capacity=100, tc_rate=5000, draft=8.0,
        min_speed=10, max_speed=14, design_speed=12.0,
        fuel_design=18.8, fuel_idle=2.4,
        panama_fee=0, suez_fee=0, quantity=3,
    )
    ports = {
        "A": Port("A", "A", 0.0, 0.0, 10.0, 10.0, 20.0, 100.0, 0.0),
        "B": Port("B", "B", 1.0, 1.0, 10.0, 15.0, 25.0, 100.0, 0.0),
        "C": Port("C", "C", 2.0, 2.0, 10.0, 15.0, 25.0, 100.0, 0.0),
        "D": Port("D", "D", 3.0, 3.0, 10.0, 15.0, 25.0, 100.0, 0.0),
    }
    demands = [("A", "C", 50.0, 100.0), ("B", "D", 50.0, 100.0)]
    solver = LsndMetaheuristic(
        vessels=[vessel],
        ports_dict=ports,
        demands=demands,
        dist_min={
            ("A", "B"): 100.0, ("B", "C"): 100.0, ("C", "A"): 100.0,
            ("A", "D"): 100.0, ("D", "B"): 100.0, ("B", "A"): 100.0,
            ("C", "D"): 100.0, ("D", "A"): 100.0, ("C", "B"): 100.0,
        },
        canal_info={},
        cost_config=CostConfig(),
        meta_config=MetaheuristicConfig(max_time=5),
        tabu_config=TabuConfig(),
    )
    current = [
        Rotation(id="r1", vessel_class=vessel, speed=12.0, num_vessels=1, port_calls=["A", "B", "C"]),
        Rotation(id="r2", vessel_class=vessel, speed=12.0, num_vessels=1, port_calls=["A", "D", "B"]),
    ]
    additions = [
        Rotation(id="n1", vessel_class=vessel, speed=12.0, num_vessels=1, port_calls=["A", "C", "D"]),
        Rotation(id="n2", vessel_class=vessel, speed=12.0, num_vessels=1, port_calls=["B", "D", "C"]),
    ]

    scenarios = solver._build_eval_scenarios(additions, current, current_mcfp=None)
    assert scenarios, "pair scenarios should not be empty"
    assert any(len(removals) == 1 for _, removals in scenarios), scenarios

    print("  [OK] pair 邻域场景生成正常")


def test_solver_baltic():
    """测试在 Baltic 实例上的完整求解。"""
    from src.algorithm.solver import solve_instance

    print("  正在求解 Baltic 实例 (20s)...")
    result = solve_instance(
        "Baltic",
        method="metaheuristic",
        max_time=20,
        verbose=False,
        random_seed=7,
        collect_diagnostics=True,
    )

    assert result.status != "Error", f"求解出错: {result.status}"
    assert result.solve_time > 0, "求解时间为 0"
    assert result.columns_evaluated >= result.unique_columns
    assert result.diagnostics_path is not None

    bd = result.cost_breakdown
    print(f"  Status: {result.status}")
    print(f"  Objective: {result.objective:,.0f}")
    print(f"  Rotations: {len(result.rotations)}")
    print(f"  Backend: {result.solver_backend}")
    print(f"  Columns: {result.columns_evaluated}, unique: {result.unique_columns}, MCF eval: {result.mcf_evaluations}")
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
        ("场景缩放规则", test_scenario_scaling),
        ("Rotation 数据结构", test_rotation),
        ("MCFP 简单夹具", test_mcfp_small_fixture),
        ("Baltic 论文回放", test_baltic_paper_replay),
        ("AUX 蝴蝶提取", test_aux_butterfly_extraction),
        ("AUX 多候选枚举", test_aux_multiple_candidates),
        ("seed route 唯一性", test_seed_route_id_uniqueness),
        ("MCFP 约束命名唯一性", test_mcfp_constraint_naming_uniqueness),
        ("pair 邻域场景", test_pair_move_scenarios),
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
