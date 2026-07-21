import argparse
import json
from pathlib import Path

from benchmark_static import DEFAULT_CASES
from simulation import run_simulation


def _case_label(case):
    parts = [case["scenario"]]
    if case.get("pillar_height") is not None:
        parts.append(f"h={case['pillar_height']}")
    if case.get("pillar_radius") is not None:
        parts.append(f"r={case['pillar_radius']}")
    if case.get("height_range") is not None:
        parts.append(f"height_range={case['height_range']}")
    return " ".join(parts)


def _run_case(index, total, case, args, output_dir):
    scenario = case["scenario"]
    height_seed = int(case.get("height_seed", args.height_seed))
    summary_file = output_dir / f"{index:02d}_{scenario}_summary.json"
    log_file = output_dir / f"{index:02d}_{scenario}.csv" if args.write_step_logs else None

    print(f"[{index:02d}/{total:02d}] START {_case_label(case)}", flush=True)
    _, _, _, metrics = run_simulation(
        scenario=scenario,
        pillar_height=case.get("pillar_height", args.pillar_height),
        flight_height=args.flight_height,
        pillar_radius=case.get("pillar_radius"),
        max_steps=args.max_steps,
        dt=args.dt,
        predict_time=args.predict_time,
        show_plot=False,
        log_file=log_file,
        summary_file=summary_file,
        height_range=case.get("height_range"),
        height_seed=height_seed,
        check_feasibility=False,
    )

    success = metrics["status"] == "goal_reached" and metrics["collision_rate"] == 0.0
    mark = "PASS" if success else "FAIL"
    print(
        f"[{index:02d}/{total:02d}] {mark} {scenario} "
        f"status={metrics['status']} collision={metrics['collision_rate']:.3f} "
        f"clearance={metrics['minimum_clearance']:.3f} path={metrics['path_length']:.3f} "
        f"time={metrics['flight_time']:.3f} max_z={metrics['max_altitude']:.3f} "
        f"height_var={metrics['height_variation']:.3f}",
        flush=True,
    )
    return {
        "scenario": scenario,
        "label": _case_label(case),
        "success": success,
        "summary_file": str(summary_file),
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="One-pass static scene pilot checker.")
    parser.add_argument("--output-dir", default="logs/static_pilot")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--predict-time", type=float, default=2.0)
    parser.add_argument("--pillar-height", type=float, default=5.0)
    parser.add_argument("--flight-height", type=float, default=3.0)
    parser.add_argument("--height-seed", type=int, default=0)
    parser.add_argument("--write-step-logs", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, case in enumerate(DEFAULT_CASES, start=1):
        rows.append(_run_case(index, len(DEFAULT_CASES), case, args, output_dir))

    successes = sum(1 for row in rows if row["success"])
    summary = {
        "total": len(rows),
        "successes": successes,
        "failures": len(rows) - successes,
        "success_rate": successes / len(rows) if rows else 0.0,
        "failed_cases": [row for row in rows if not row["success"]],
        "cases": rows,
    }
    summary_path = output_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("pilot summary:", flush=True)
    print(f"  success_rate: {summary['success_rate']:.3f}", flush=True)
    print(f"  successes: {successes}/{len(rows)}", flush=True)
    print(f"  summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
