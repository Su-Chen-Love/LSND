"""
Benchmark runner for comparing LSND results against paper baselines.

Examples:
  python -m tests.run_benchmark --instance Baltic --time 120
  python -m tests.run_benchmark --instance Baltic --time 300 --runs 10 --seed-base 42
  python -m tests.run_benchmark --quick
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, List

from src.algorithm.solver import SolverResult, solve_instance
from src.paper_replay import evaluate_baltic_base_replay
from src.data_reader import load_instance, load_distances
from src.utils.cost_calculator import build_vessels_from_data, build_ports_from_data, build_distance_dict, CostConfig


PAPER_BENCHMARKS = {
    "Baltic": {
        "base": {"time": 300, "Z_best": -8_365_000, "Z_median": -6_582_000, "Q_best": 98_310_000, "rotations_best": 3, "reject_pct_best": 5.0, "source": "Baltic_best_base.log"},
        "low": {"time": 300, "Z_best": -6_104_400, "Z_median": 83_800, "Q_best": 96_175_000, "rotations_best": 4, "reject_pct_best": 8.2},
        "high": {"time": 300, "Z_best": -15_167_800, "Z_median": -11_150_900, "Q_best": 103_150_000, "rotations_best": 4, "reject_pct_best": 0.1},
    },
    "WAF": {
        "base": {"time": 900, "Z_best": -1_431_110_000, "Z_median": -1_371_681_000, "Q_best": 3_701_600_000, "rotations_best": 11, "reject_pct_best": 4.7},
    },
    "Mediterranean": {
        "base": {"time": 1200, "Z_best": 12_209_000, "Z_median": 24_934_000, "Q_best": 136_800_000, "rotations_best": 7, "reject_pct_best": 1.2},
    },
    "Pacific": {
        "base": {"time": 3600, "Z_best": -54_087_000, "Z_median": -12_774_000, "Q_best": 1_197_000_000, "rotations_best": 21, "reject_pct_best": 3.1},
    },
}


FULL_TEST_TIMES = {
    "Baltic": 300,
    "WAF": 900,
    "Mediterranean": 1200,
    "Pacific": 3600,
    "EuropeAsia": 14400,
    "WorldSmall": 10800,
}


def paper_gap_pct(result: SolverResult, reference: Dict) -> float | None:
    if not reference:
        return None
    denom = max(abs(reference["Z_best"]), 1.0)
    return (result.objective - reference["Z_best"]) / denom * 100.0


def run_single(
    instance: str,
    max_time: int,
    method: str = "metaheuristic",
    scenario: str = "base",
    backend: str = "auto",
    seed: int = 0,
    collect_diagnostics: bool = False,
) -> SolverResult:
    result = solve_instance(
        instance,
        method=method,
        max_time=max_time,
        verbose=False,
        solver_backend=backend,
        random_seed=seed,
        scenario=scenario,
        benchmark_mode=True,
        collect_diagnostics=collect_diagnostics,
    )

    ref = PAPER_BENCHMARKS.get(instance, {}).get(scenario)
    result.paper_gap_pct = paper_gap_pct(result, ref)
    return result


def baltic_paper_replay_snapshot() -> dict:
    """Evaluate the bundled Baltic paper solution with the current cost/MCFP stack."""
    data = load_instance("Baltic", scenario="base")
    vessels = {v.name: v for v in build_vessels_from_data(data["fleet"])}
    ports = build_ports_from_data(data["ports_all"])
    _, dist_min, canal_info = build_distance_dict(load_distances())
    weeks = 180 / 7.0
    demands = [
        (row["Origin"], row["Destination"], float(row["FFEPerWeek"]) * weeks, float(row["Revenue_1"]))
        for _, row in data["demand"].iterrows()
    ]
    return evaluate_baltic_base_replay(
        vessels_by_name=vessels,
        ports_dict=ports,
        demands=demands,
        dist_min=dist_min,
        canal_info=canal_info,
        cost_config=CostConfig(),
        solver_backend="auto",
    )


def print_single(result: SolverResult):
    bd = result.cost_breakdown
    fs = result.flow_summary
    total_demand = fs.get("total_demand_ffe", 0.0)
    rejected = fs.get("rejected_ffe", 0.0)
    reject_pct = rejected / total_demand * 100 if total_demand else 0.0
    ref = PAPER_BENCHMARKS.get(result.instance_name, {}).get(result.scenario)

    print(f"\n{'=' * 78}")
    print(
        f" 实例: {result.instance_name} | 场景: {result.scenario} | 方法: {result.method}"
        f" | 后端: {result.solver_backend} | seed: {result.seed}"
    )
    print(f"{'=' * 78}")
    print(f"  状态: {result.status}")
    print(f"  求解时间: {result.solve_time:.1f}s")
    print(f"  迭代次数: {result.iterations}")
    print(f"  候选列: {result.columns_evaluated} | 唯一列: {result.unique_columns} | MCF评估: {result.mcf_evaluations}")
    print(f"  接受列: {result.accepted_columns} | same-class swaps: {result.same_class_swap_count} | backtracks: {result.backtrack_count}")
    print(f"  pair moves: {result.pair_moves_evaluated} | accepted move: {result.accepted_move_type} | plateau: {result.plateau_triggered}")
    if result.candidate_pool_counts:
        print(f"  candidate pools: {result.candidate_pool_counts}")
    if result.diagnostics_path:
        print(f"  诊断文件: {result.diagnostics_path}")

    print(f"\n  === 成本分解 (论文 Table 9 对齐) ===")
    print(f"  Z={bd.Z:,.0f}  Q={bd.Q:,.0f}  c_v={bd.c_v:,.0f}  c_b={bd.c_b:,.0f}")
    print(f"  c_p={bd.c_p:,.0f}  c_c={bd.c_c:,.0f}  c_m={bd.c_m:,.0f}  c_t={bd.c_t:,.0f}")
    print(f"  L_v={bd.L_v:,.0f}  F={bd.F:,.0f}")

    print(f"\n  === 网络指标 ===")
    print(f"  航线数 |R|: {len(result.rotations)}")
    print(f"  服务率: {fs.get('service_rate', 0):.1%}")
    print(f"  拒绝率: {reject_pct:.1f}%")
    print(f"  总需求(规划期): {total_demand:,.0f} FFE")

    if ref:
        print(f"\n  === 与论文对比 ===")
        print(f"  论文 best Z: {ref['Z_best']:,.0f}")
        print(f"  论文 median Z: {ref['Z_median']:,.0f}")
        print(f"  论文 best |R|: {ref['rotations_best']}")
        print(f"  论文 best 拒绝率: {ref['reject_pct_best']:.1f}%")
        if "source" in ref:
            print(f"  论文参考来源: {ref['source']}")
        if result.paper_gap_pct is not None:
            print(f"  相对论文 best 目标差距: {result.paper_gap_pct:.1f}%")
        if result.instance_name == "Baltic" and result.scenario == "base":
            replay = baltic_paper_replay_snapshot()
            print(f"\n  === 当前模型下的论文服务回放 ===")
            print(f"  replay objective: {replay['objective']:,.0f}")
            print(f"  replay service rate: {replay['service_rate']:.1%}")
            print(f"  replay rejected: {replay['rejected_ffe']:,.0f} FFE")


def print_summary(results: List[SolverResult], instance: str, scenario: str):
    objectives = [r.objective for r in results]
    reject_pcts = [
        (r.flow_summary.get("rejected_ffe", 0.0) / max(r.flow_summary.get("total_demand_ffe", 1.0), 1.0)) * 100.0
        for r in results
    ]
    ref = PAPER_BENCHMARKS.get(instance, {}).get(scenario)
    sorted_by_obj = sorted(results, key=lambda r: r.objective)
    median_result = sorted_by_obj[len(sorted_by_obj) // 2]

    print(f"\n{'=' * 78}")
    print(f" 汇总: {instance} / {scenario} / {len(results)} runs")
    print(f"{'=' * 78}")
    print(f"  最佳目标: {min(objectives):,.0f}")
    print(f"  中位目标: {statistics.median(objectives):,.0f}")
    print(f"  平均目标: {statistics.mean(objectives):,.0f}")
    print(f"  中位拒绝率: {statistics.median(reject_pcts):.1f}%")
    print(f"  最佳航线数: {len(sorted_by_obj[0].rotations)}")
    print(f"  中位航线数: {len(median_result.rotations)}")
    print(f"  最佳后端: {sorted_by_obj[0].solver_backend}")
    print(f"  中位候选列: {median_result.columns_evaluated} | 唯一列: {median_result.unique_columns} | MCF评估: {median_result.mcf_evaluations}")
    print(f"  中位接受列: {median_result.accepted_columns} | same-class swaps: {median_result.same_class_swap_count} | backtracks: {median_result.backtrack_count}")
    print(f"  中位 pair moves: {median_result.pair_moves_evaluated} | accepted move: {median_result.accepted_move_type} | plateau: {median_result.plateau_triggered}")
    if median_result.candidate_pool_counts:
        print(f"  中位 candidate pools: {median_result.candidate_pool_counts}")
    if ref:
        best_gap = paper_gap_pct(sorted_by_obj[0], ref)
        med_gap = paper_gap_pct(median_result, ref)
        print(f"  与论文 best 差距(best run): {best_gap:.1f}%")
        print(f"  与论文 best 差距(median run): {med_gap:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="LSND benchmark runner")
    parser.add_argument("--quick", action="store_true", help="Run Baltic/base for 120s")
    parser.add_argument("--full", action="store_true", help="Run all default instances with paper time limits")
    parser.add_argument("--instance", type=str, default=None, help="Instance name")
    parser.add_argument("--time", type=int, default=120, help="Time limit in seconds")
    parser.add_argument("--method", type=str, default="metaheuristic", choices=["metaheuristic", "mip"])
    parser.add_argument("--scenario", type=str, default="base", choices=["base", "low", "high"])
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs")
    parser.add_argument("--seed-base", type=int, default=0, help="Base random seed")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "gurobi", "cbc"])
    parser.add_argument("--diagnostics", action="store_true", help="Write diagnostics JSON files")
    args = parser.parse_args()

    start_all = time.time()

    if args.quick:
        results = [run_single("Baltic", 120, method=args.method, scenario="base", backend=args.backend, seed=args.seed_base, collect_diagnostics=args.diagnostics)]
        print_single(results[0])
    elif args.full:
        for inst, t in FULL_TEST_TIMES.items():
            result = run_single(inst, t, method="metaheuristic", scenario="base", backend=args.backend, seed=args.seed_base, collect_diagnostics=args.diagnostics)
            print_single(result)
    else:
        instance = args.instance or "Baltic"
        results = [
            run_single(
                instance,
                args.time,
                method=args.method,
                scenario=args.scenario,
                backend=args.backend,
                seed=args.seed_base + run_idx,
                collect_diagnostics=args.diagnostics,
            )
            for run_idx in range(args.runs)
        ]
        for result in results:
            print_single(result)
        if len(results) > 1:
            print_summary(results, instance, args.scenario)

    total = time.time() - start_all
    print(f"\n{'=' * 78}")
    print(f" 总耗时: {total:.1f}s ({total / 60:.1f} 分钟)")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
