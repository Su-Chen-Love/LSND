"""
基准验证脚本 — 对比论文 Table 9 结果。

用法:
  快速测试 (Baltic, 120s):
    python -m tests.run_benchmark --quick

  完整测试 (所有实例, 论文推荐时间):
    python -m tests.run_benchmark --full

  指定实例和时间:
    python -m tests.run_benchmark --instance Baltic --time 300
"""

import argparse
import sys
import time
import logging

from src.algorithm.solver import solve_instance

logging.basicConfig(level=logging.WARNING)

# 论文 Table 9 参考值 (Base 配置)
# 注意: 论文 Table 9 中成本值单位为 $1000 USD，此处已转换为 USD
# Z: 目标值, Q: 收入, |R|: 航线数, Rej%: 拒绝率, F: 拒运 FFE
PAPER_BENCHMARKS = {
    "Baltic": {
        "time": 300,
        "Z_best": -6_044_000,
        "Z_median": 858_000,
        "Q_best": 96_750_000,
        "rotations_best": 5,
        "reject_pct_best": 8.6,
        "F_best": 10_850,
    },
    "WAF": {
        "time": 900,
        "Z_best": -115_361_000,
        "Z_median": -109_470_000,
        "Q_best": 345_800_000,
        "rotations_best": 7,
        "reject_pct_best": 11.7,
        "F_best": 25_710,
    },
    "Mediterranean": {
        "time": 1200,
        "Z_best": -218_980_000,
        "Z_median": -165_440_000,
        "Q_best": 460_400_000,
        "rotations_best": 8,
        "reject_pct_best": 33.5,
        "F_best": 65_210,
    },
    "Pacific": {
        "time": 3600,
        "Z_best": -2_321_770_000,
        "Z_median": -2_051_750_000,
        "Q_best": 3_456_000_000,
        "rotations_best": 17,
        "reject_pct_best": 27.8,
        "F_best": 327_530,
    },
}

# 论文中使用的时间与实例映射 (Table 7)
FULL_TEST_TIMES = {
    "Baltic": 300,
    "WAF": 900,
    "Mediterranean": 1200,
    "Pacific": 3600,
    "EuropeAsia": 14400,
    "WorldSmall": 10800,
}


def run_benchmark(instance: str, max_time: int, method: str = "metaheuristic"):
    """运行单个实例基准测试。"""
    print(f"\n{'='*70}")
    print(f" 实例: {instance} | 方法: {method} | 时限: {max_time}s")
    print(f"{'='*70}")

    result = solve_instance(
        instance, method=method, max_time=max_time, verbose=False
    )

    bd = result.cost_breakdown
    fs = result.flow_summary
    total_demand = fs.get("total_demand_ffe", 0)
    rejected = fs.get("rejected_ffe", 0)
    reject_pct = (rejected / total_demand * 100) if total_demand > 0 else 0

    print(f"\n  状态: {result.status}")
    print(f"  求解时间: {result.solve_time:.1f}s")
    print(f"  迭代次数: {result.iterations}")
    print(f"\n  === 成本分解 (对应论文 Table 9) ===")
    print(f"  {'指标':<8} {'本次结果':>15} {'单位'}")
    print(f"  {'─'*40}")
    print(f"  {'Z':.<8} {bd.Z:>15,.0f}  USD")
    print(f"  {'Q':.<8} {bd.Q:>15,.0f}  USD (收入)")
    print(f"  {'c_v':.<8} {bd.c_v:>15,.0f}  USD (TC)")
    print(f"  {'c_b':.<8} {bd.c_b:>15,.0f}  USD (燃油)")
    print(f"  {'c_p':.<8} {bd.c_p:>15,.0f}  USD (港口)")
    print(f"  {'c_c':.<8} {bd.c_c:>15,.0f}  USD (运河)")
    print(f"  {'c_m':.<8} {bd.c_m:>15,.0f}  USD (装卸)")
    print(f"  {'c_t':.<8} {bd.c_t:>15,.0f}  USD (转运)")
    print(f"  {'L_v':.<8} {bd.L_v:>15,.0f}  USD (租出)")
    print(f"  {'F':.<8} {bd.F:>15,.0f}  FFE (拒运)")

    print(f"\n  === 网络指标 ===")
    print(f"  航线数 |R|: {len(result.rotations)}")
    print(f"  服务率: {fs.get('service_rate', 0):.1%}")
    print(f"  拒绝率: {reject_pct:.1f}%")
    print(f"  总需求: {total_demand:,.0f} FFE/week")

    # 航线详情
    if result.rotation_details:
        print(f"\n  === 航线详情 ===")
        for rd in result.rotation_details:
            ports_str = " → ".join(rd["port_calls"])
            print(f"  {rd['id']}: {rd['vessel_class']}, {rd['speed']}kn, "
                  f"{rd['num_vessels']}v, {rd['round_trip_days']}d "
                  f"({rd['frequency']})")
            print(f"    路径: {ports_str}")

    # 与论文对比
    ref = PAPER_BENCHMARKS.get(instance)
    if ref:
        print(f"\n  === 与论文对比 ===")
        print(f"  {'指标':<15} {'本次':>12} {'论文 Best':>12} {'论文 Median':>12}")
        print(f"  {'─'*55}")
        print(f"  {'Z':<15} {bd.Z:>12,.0f} {ref['Z_best']:>12,.0f} {ref['Z_median']:>12,.0f}")
        print(f"  {'|R|':<15} {len(result.rotations):>12} {ref['rotations_best']:>12}")
        print(f"  {'拒绝率%':<13} {reject_pct:>12.1f} {ref['reject_pct_best']:>12.1f}")

        # 简单评估
        if bd.Z <= ref["Z_median"]:
            verdict = "优于或接近论文中位数"
        elif bd.Z <= ref["Z_median"] * 1.5:
            verdict = "在合理范围内"
        else:
            verdict = "偏差较大，建议增加时间或检查参数"
        print(f"\n  评估: {verdict}")

    return result


def main():
    parser = argparse.ArgumentParser(description="LSND 基准验证脚本")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试: Baltic 120s")
    parser.add_argument("--full", action="store_true",
                        help="完整测试: 所有实例，论文推荐时间")
    parser.add_argument("--instance", type=str, default=None,
                        help="指定实例名称")
    parser.add_argument("--time", type=int, default=120,
                        help="最大求解时间 (秒)")
    parser.add_argument("--method", type=str, default="metaheuristic",
                        choices=["metaheuristic", "mip"],
                        help="求解方法")
    args = parser.parse_args()

    start_all = time.time()

    if args.quick:
        run_benchmark("Baltic", max_time=120, method=args.method)
    elif args.full:
        for inst, t in FULL_TEST_TIMES.items():
            run_benchmark(inst, max_time=t, method="metaheuristic")
    elif args.instance:
        run_benchmark(args.instance, max_time=args.time, method=args.method)
    else:
        # 默认: 快速 Baltic 测试
        run_benchmark("Baltic", max_time=120, method=args.method)

    total = time.time() - start_all
    print(f"\n{'='*70}")
    print(f" 总耗时: {total:.1f}s ({total/60:.1f} 分钟)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
