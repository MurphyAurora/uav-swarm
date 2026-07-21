import argparse
import csv
import json
import statistics
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from simulation import run_simulation


DEFAULT_CASES = [
    {"scenario": "no_obstacle"},
    {"scenario": "single_pillar", "pillar_height": 3.0, "pillar_radius": 1.0},
    {"scenario": "single_pillar", "pillar_height": 5.0, "pillar_radius": 1.6},
    {"scenario": "test_side_bypass"},
    {"scenario": "forest_gap"},
    {"scenario": "forest_gap_mixed_height"},
    {"scenario": "s_curve_easy"},
    {"scenario": "s_curve_medium"},
    {"scenario": "s_curve_mixed_height"},
    {"scenario": "random_forest_sparse", "height_range": (2.0, 8.0)},
    {"scenario": "random_forest_medium", "height_range": (2.0, 8.0)},
]


CSV_COLUMNS = [
    "run_id",
    "scenario",
    "height_seed",
    "pillar_height",
    "pillar_radius",
    "success",
    "status",
    "collision_rate",
    "path_length",
    "flight_time",
    "minimum_clearance",
    "max_altitude",
    "height_variation",
    "computation_time",
    "avg_computation_time",
    "max_computation_time",
]


def _float_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isinf(value):
        return "inf"
    return float(value)


def _case_runs(cases, seeds):
    for case in cases:
        if "height_range" in case:
            for seed in seeds:
                run = dict(case)
                run["height_seed"] = seed
                yield run
        else:
            run = dict(case)
            run["height_seed"] = 0
            yield run


def _run_one(run_id, case, args, run_log_dir):
    scenario = case["scenario"]
    height_seed = int(case.get("height_seed", 0))
    pillar_height = case.get("pillar_height", args.pillar_height)
    pillar_radius = case.get("pillar_radius")
    height_range = case.get("height_range")

    run_log = run_log_dir / f"{run_id:03d}_{scenario}_seed{height_seed}.txt"
    with run_log.open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle):
            _, _, _, metrics = run_simulation(
                scenario=scenario,
                pillar_height=pillar_height,
                flight_height=args.flight_height,
                pillar_radius=pillar_radius,
                max_steps=args.max_steps,
                dt=args.dt,
                predict_time=args.predict_time,
                show_plot=False,
                log_file=None,
                summary_file=None,
                height_range=height_range,
                height_seed=height_seed,
                check_feasibility=False,
            )

    success = metrics["status"] == "goal_reached" and metrics["collision_rate"] == 0.0
    return {
        "run_id": run_id,
        "scenario": scenario,
        "height_seed": height_seed,
        "pillar_height": pillar_height,
        "pillar_radius": pillar_radius,
        "success": bool(success),
        "status": metrics["status"],
        "collision_rate": _float_or_none(metrics["collision_rate"]),
        "path_length": _float_or_none(metrics["path_length"]),
        "flight_time": _float_or_none(metrics["flight_time"]),
        "minimum_clearance": _float_or_none(metrics["minimum_clearance"]),
        "max_altitude": _float_or_none(metrics["max_altitude"]),
        "height_variation": _float_or_none(metrics["height_variation"]),
        "computation_time": _float_or_none(metrics["computation_time"]),
        "avg_computation_time": _float_or_none(metrics["avg_computation_time"]),
        "max_computation_time": _float_or_none(metrics["max_computation_time"]),
        "log_file": str(run_log),
    }


def _mean(rows, key):
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return statistics.mean(values)


def _aggregate(rows, pass_threshold):
    total = len(rows)
    successes = sum(1 for row in rows if row["success"])
    failures = [row for row in rows if not row["success"]]
    success_rate = successes / total if total else 0.0
    collision_runs = sum(1 for row in rows if isinstance(row["collision_rate"], (int, float)) and row["collision_rate"] > 0.0)

    by_scenario = {}
    for row in rows:
        bucket = by_scenario.setdefault(row["scenario"], {"total": 0, "successes": 0, "failures": 0})
        bucket["total"] += 1
        if row["success"]:
            bucket["successes"] += 1
        else:
            bucket["failures"] += 1
    for bucket in by_scenario.values():
        bucket["success_rate"] = bucket["successes"] / bucket["total"] if bucket["total"] else 0.0

    return {
        "total_runs": total,
        "successes": successes,
        "failures": len(failures),
        "success_rate": success_rate,
        "pass_threshold": pass_threshold,
        "passed": success_rate >= pass_threshold,
        "collision_runs": collision_runs,
        "mean_path_length": _mean(rows, "path_length"),
        "mean_flight_time": _mean(rows, "flight_time"),
        "mean_minimum_clearance": _mean(rows, "minimum_clearance"),
        "mean_max_altitude": _mean(rows, "max_altitude"),
        "mean_height_variation": _mean(rows, "height_variation"),
        "mean_avg_computation_time": _mean(rows, "avg_computation_time"),
        "by_scenario": by_scenario,
        "failed_runs": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Static-map success-rate benchmark for the 3D motion primitive planner.")
    parser.add_argument("--output-dir", default="logs/static_benchmark")
    parser.add_argument("--seeds", type=int, default=20, help="Height seeds for mixed-height benchmark cases.")
    parser.add_argument("--pass-threshold", type=float, default=0.90)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--predict-time", type=float, default=2.0)
    parser.add_argument("--pillar-height", type=float, default=5.0)
    parser.add_argument("--flight-height", type=float, default=3.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_log_dir = output_dir / "run_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seeds))
    rows = []
    runs = list(_case_runs(DEFAULT_CASES, seeds))
    print(f"static benchmark runs: {len(runs)}")

    for run_id, case in enumerate(runs, start=1):
        row = _run_one(run_id, case, args, run_log_dir)
        rows.append(row)
        mark = "PASS" if row["success"] else "FAIL"
        print(
            f"[{run_id:03d}/{len(runs):03d}] {mark} "
            f"scenario={row['scenario']} seed={row['height_seed']} "
            f"status={row['status']} collision={row['collision_rate']}"
        )

    csv_path = output_dir / "static_benchmark_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_COLUMNS})

    summary = _aggregate(rows, args.pass_threshold)
    summary_path = output_dir / "static_benchmark_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("summary:")
    print(f"  success_rate: {summary['success_rate']:.3f}")
    print(f"  successes: {summary['successes']}/{summary['total_runs']}")
    print(f"  collision_runs: {summary['collision_runs']}")
    print(f"  passed_90_percent: {summary['passed']}")
    print(f"  csv: {csv_path}")
    print(f"  summary: {summary_path}")


if __name__ == "__main__":
    main()
