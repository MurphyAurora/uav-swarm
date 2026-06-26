#!/usr/bin/env python3
"""Offline 2D planner smoke simulator.

This script runs the pure-Python planner framework without ROS, PX4, Gazebo, or
RViz.  It is intended for quick logic checks on Linux or Windows:

    python scripts/offline_planner_sim.py --case single_pillar --plot

Scenario/map definitions live in scripts/offline_scenarios/.  This runner only
loads a scenario, executes the planner loop, and writes optional CSV/plot output.

On Windows from the repository root, if imports fail:

    powershell: $env:PYTHONPATH="$PWD\\src\\xtd2_mission"
    cmd.exe:    set PYTHONPATH=%CD%\\src\\xtd2_mission
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src", "xtd2_mission")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from offline_scenarios.base import SimCase, bounds_margin, collision_margin, make_perception, obstacle_snapshots
from offline_scenarios.registry import CASE_REGISTRY, get_case
from xtd2_mission.planner_framework import EgoLikePlannerCore, PlannerConfig, PlannerState, Vec3


def simulate(case: SimCase, config: PlannerConfig, verbose_every: int) -> Tuple[List[Dict[str, object]], str]:
    planner = EgoLikePlannerCore(config)
    pos = case.start
    vel = Vec3()
    heading = math.atan2(case.goal.y - case.start.y, case.goal.x - case.start.x)
    history: List[Dict[str, object]] = []
    status = "max_steps"
    last_source = ""

    for step in range(case.steps):
        stamp = step * case.dt
        next_stamp = (step + 1) * case.dt
        state = PlannerState(drone_id=1, position=pos, velocity=vel, stamp=stamp, heading=heading)
        report = planner.plan(state, case.goal, make_perception(case, stamp))
        cmd = report["command"]
        cmd_vel = Vec3.from_mapping(cmd.get("velocity", {}))

        # Minimal kinematic closed loop: apply the command for one offline step.
        # A later dynamics test can replace this with an acceleration-limited model.
        pos = Vec3(pos.x + cmd_vel.x * case.dt, pos.y + cmd_vel.y * case.dt, pos.z + cmd_vel.z * case.dt)
        vel = cmd_vel
        if math.hypot(vel.x, vel.y) > 0.03:
            heading = math.atan2(vel.y, vel.x)

        margin = collision_margin(pos, case, config.drone_radius, next_stamp)
        b_margin = bounds_margin(pos, case, config.drone_radius)
        row = {
            "step": step,
            "t": next_stamp,
            "x": pos.x,
            "y": pos.y,
            "z": pos.z,
            "vx": vel.x,
            "vy": vel.y,
            "mode": cmd.get("mode", ""),
            "source": cmd.get("source_trajectory", ""),
            "reason": cmd.get("reason", ""),
            "safe_count": report.get("safe_count", 0),
            "feasible_count": report.get("feasible_count", 0),
            "candidate_count": report.get("candidate_count", 0),
            "collision_margin": margin,
            "bounds_margin": b_margin,
            "max_abs_y": abs(pos.y),
            "distance_to_goal": pos.distance_to(case.goal),
        }
        history.append(row)

        source_changed = row["source"] != last_source
        if verbose_every > 0 and (step % verbose_every == 0 or source_changed):
            print(
                f"step={step:04d} pos=({pos.x:5.2f},{pos.y:5.2f}) "
                f"cmd=({vel.x:5.2f},{vel.y:5.2f}) src={row['source']} "
                f"reason={row['reason']} margin={margin:5.2f} bounds={b_margin:5.2f}"
            )
        last_source = str(row["source"])

        if margin < 0.0:
            status = "collision"
            break
        if b_margin < 0.0:
            status = "out_of_bounds"
            break
        if pos.distance_to(case.goal) <= config.local_goal_reached_radius:
            status = "reached"
            break
    return history, status


def write_csv(path: str, history: Iterable[Dict[str, object]]) -> None:
    rows = list(history)
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_case(path: str, case: SimCase, history: List[Dict[str, object]], config: PlannerConfig) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --plot: pip install matplotlib") from exc

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    if case.bounds is not None:
        width = case.bounds.xmax - case.bounds.xmin
        height = case.bounds.ymax - case.bounds.ymin
        rect = Rectangle((case.bounds.xmin, case.bounds.ymin), width, height, fill=False, linestyle="--", linewidth=1.2, label="bounds")
        ax.add_patch(rect)
    ax.plot([case.start.x], [case.start.y], "go", label="start")
    ax.plot([case.goal.x], [case.goal.y], "r*", markersize=12, label="goal")
    if history:
        ax.plot([float(r["x"]) for r in history], [float(r["y"]) for r in history], "b-", linewidth=2, label="trajectory")

    final_t = float(history[-1]["t"]) if history else 0.0
    for x, y, radius, obs_id in obstacle_snapshots(case, 0.0):
        circle = plt.Circle((x, y), radius + config.drone_radius, color="tab:gray", alpha=0.25)
        ax.add_patch(circle)
        ax.plot(x, y, "ko", markersize=3)
        if obs_id:
            ax.text(x, y, obs_id, fontsize=7)
    if case.dynamic_obstacles and final_t > 0.0:
        for x, y, radius, obs_id in obstacle_snapshots(case, final_t):
            circle = plt.Circle((x, y), radius + config.drone_radius, color="tab:orange", alpha=0.18)
            ax.add_patch(circle)
            ax.plot(x, y, "kx", markersize=4)
            if obs_id:
                ax.text(x, y, f"{obs_id}@end", fontsize=7)

    ax.set_title(case.name)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[plot] wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_REGISTRY), default="single_pillar")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--csv", default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", default="")
    parser.add_argument("--strict", action="store_true", help="return a non-zero exit code when the case is not reached")
    parser.add_argument("--verbose-every", type=int, default=20)
    args = parser.parse_args()

    case = get_case(args.case)
    if args.steps is not None or args.dt is not None:
        case = replace(
            case,
            steps=args.steps if args.steps is not None else case.steps,
            dt=args.dt if args.dt is not None else case.dt,
        )

    config = PlannerConfig(
        local_goal_distance=2.2,
        horizon=3.0,
        dt=0.5,
        obstacle_margin=0.20,
        static_hard_clearance=0.25,
        static_emergency_clearance=0.45,
        goal_weight=3.0,
        progress_weight=4.0,
        lateral_penalty=3.0,
        reverse_penalty=8.0,
    )
    history, status = simulate(case, config, args.verbose_every)
    final = history[-1] if history else {}
    print(
        f"[result] case={case.name} status={status} steps={len(history)} "
        f"final=({float(final.get('x', case.start.x)):.2f},{float(final.get('y', case.start.y)):.2f}) "
        f"goal_dist={float(final.get('distance_to_goal', case.start.distance_to(case.goal))):.2f} "
        f"min_margin={min(float(r['collision_margin']) for r in history):.2f} "
        f"min_bounds={min(float(r['bounds_margin']) for r in history):.2f} "
        f"max_abs_y={max(float(r['max_abs_y']) for r in history):.2f}"
    )

    if args.csv:
        write_csv(args.csv, history)
        print(f"[csv] wrote {args.csv}")
    if args.plot:
        plot_path = args.plot_path or os.path.join(REPO_ROOT, "log", f"offline_{case.name}.png")
        plot_case(plot_path, case, history, config)
    return 0 if status == "reached" or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
