#!/usr/bin/env python3
"""Run the ROS-independent planner benchmark from the repository checkout."""

import argparse
import json
from pathlib import Path

from xtd2_mission.offline_benchmark import OfflineBenchmarkRunner, make_scenario
from xtd2_mission.planner_framework import EgoLikePlannerAdapter, PlannerConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="open_space",
        choices=("open_space", "single_static_column", "dynamic_crossing", "dynamic_head_on"),
    )
    parser.add_argument("--frontend-mode", default="local_astar")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--save-traj", action="store_true")
    args = parser.parse_args()

    planner = EgoLikePlannerAdapter(PlannerConfig(frontend_mode=args.frontend_mode))
    result = OfflineBenchmarkRunner(planner).run(make_scenario(args.scenario))

    if args.save_traj:
        output = Path("results") / args.scenario
        result.trajectory.save_csv(str(output / "trajectory.csv"))

    print(json.dumps(result.metrics.to_dict(), indent=2 if args.pretty else None, ensure_ascii=False))
    return 0 if result.metrics.success else 2


if __name__ == "__main__":
    raise SystemExit(main())
