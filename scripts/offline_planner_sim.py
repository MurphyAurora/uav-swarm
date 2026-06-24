#!/usr/bin/env python3
"""Offline 2D planner smoke simulator.

This script runs the pure-Python planner framework without ROS, PX4, Gazebo, or
RViz.  It is intended for quick logic checks on Linux or Windows:

    python scripts/offline_planner_sim.py --case single_pillar --plot

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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src", "xtd2_mission")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from xtd2_mission.planner_framework import EgoLikePlannerCore, Obstacle, PerceptionData, PlannerConfig, PlannerState, Vec3


@dataclass(frozen=True)
class CircleObstacle:
    x: float
    y: float
    radius: float
    source: str = "static"
    obstacle_id: str = ""


@dataclass(frozen=True)
class SimCase:
    name: str
    start: Vec3
    goal: Vec3
    obstacles: Tuple[CircleObstacle, ...]
    steps: int = 500
    dt: float = 0.2


def build_cases() -> Dict[str, SimCase]:
    return {
        "single_pillar": SimCase(
            name="single_pillar",
            start=Vec3(0.0, 0.0, -3.0),
            goal=Vec3(12.0, 0.0, -3.0),
            obstacles=(CircleObstacle(4.0, 0.0, 0.55, obstacle_id="pillar_1"),),
        ),
        "forest_gap": SimCase(
            name="forest_gap",
            start=Vec3(0.0, 0.0, -3.0),
            goal=Vec3(16.0, 0.0, -3.0),
            obstacles=(
                CircleObstacle(3.5, -0.9, 0.45, obstacle_id="p1"),
                CircleObstacle(4.4, 0.8, 0.45, obstacle_id="p2"),
                CircleObstacle(6.2, -0.2, 0.50, obstacle_id="p3"),
                CircleObstacle(8.0, 1.0, 0.50, obstacle_id="p4"),
                CircleObstacle(9.0, -1.0, 0.50, obstacle_id="p5"),
                CircleObstacle(11.0, 0.4, 0.45, obstacle_id="p6"),
            ),
            steps=700,
        ),
        "rejoin_shell": SimCase(
            name="rejoin_shell",
            start=Vec3(0.0, 0.0, -3.0),
            goal=Vec3(8.0, 0.0, -3.0),
            obstacles=(
                CircleObstacle(1.1, -0.8, 0.50, obstacle_id="near_side"),
                CircleObstacle(3.8, 0.8, 0.45, obstacle_id="front_side"),
            ),
            steps=450,
        ),
        "narrow_corridor_easy": SimCase(
            name="narrow_corridor_easy",
            start=Vec3(0.0, 0.0, -3.0),
            goal=Vec3(13.0, 0.0, -3.0),
            obstacles=(
                # Wide corridor without a center obstacle.  This is the first
                # corridor-lock smoke test: the expected behavior is stable direct
                # traversal inside the channel, not a detour outside the rows.
                CircleObstacle(2.8, 1.75, 0.40, obstacle_id="top_1"),
                CircleObstacle(4.8, 1.75, 0.40, obstacle_id="top_2"),
                CircleObstacle(6.8, 1.75, 0.40, obstacle_id="top_3"),
                CircleObstacle(8.8, 1.75, 0.40, obstacle_id="top_4"),
                CircleObstacle(10.8, 1.75, 0.40, obstacle_id="top_5"),
                CircleObstacle(2.8, -1.75, 0.40, obstacle_id="bottom_1"),
                CircleObstacle(4.8, -1.75, 0.40, obstacle_id="bottom_2"),
                CircleObstacle(6.8, -1.75, 0.40, obstacle_id="bottom_3"),
                CircleObstacle(8.8, -1.75, 0.40, obstacle_id="bottom_4"),
                CircleObstacle(10.8, -1.75, 0.40, obstacle_id="bottom_5"),
            ),
            steps=650,
        ),
        "narrow_corridor": SimCase(
            name="narrow_corridor",
            start=Vec3(0.0, 0.0, -3.0),
            goal=Vec3(13.0, 0.0, -3.0),
            obstacles=(
                # Medium corridor with staggered rows.  Use this after the easy
                # corridor passes; it tests whether same-side memory and corridor
                # lock remain stable when the center line has mild local pressure.
                CircleObstacle(2.8, 1.60, 0.42, obstacle_id="top_1"),
                CircleObstacle(4.7, 1.55, 0.42, obstacle_id="top_2"),
                CircleObstacle(6.6, 1.60, 0.42, obstacle_id="top_3"),
                CircleObstacle(8.5, 1.55, 0.42, obstacle_id="top_4"),
                CircleObstacle(10.4, 1.60, 0.42, obstacle_id="top_5"),
                CircleObstacle(3.7, -1.60, 0.42, obstacle_id="bottom_1"),
                CircleObstacle(5.6, -1.55, 0.42, obstacle_id="bottom_2"),
                CircleObstacle(7.5, -1.60, 0.42, obstacle_id="bottom_3"),
                CircleObstacle(9.4, -1.55, 0.42, obstacle_id="bottom_4"),
                CircleObstacle(11.3, -1.60, 0.42, obstacle_id="bottom_5"),
            ),
            steps=650,
        ),
        "narrow_corridor_bias": SimCase(
            name="narrow_corridor_bias",
            start=Vec3(0.0, 0.0, -3.0),
            goal=Vec3(13.0, 0.0, -3.0),
            obstacles=(
                # Harder corridor with a small center perturbation.  This should
                # be tested only after narrow_corridor_easy and narrow_corridor pass.
                CircleObstacle(2.8, 1.55, 0.42, obstacle_id="top_1"),
                CircleObstacle(4.7, 1.55, 0.42, obstacle_id="top_2"),
                CircleObstacle(6.6, 1.55, 0.42, obstacle_id="top_3"),
                CircleObstacle(8.5, 1.55, 0.42, obstacle_id="top_4"),
                CircleObstacle(10.4, 1.55, 0.42, obstacle_id="top_5"),
                CircleObstacle(3.7, -1.55, 0.42, obstacle_id="bottom_1"),
                CircleObstacle(5.6, -1.55, 0.42, obstacle_id="bottom_2"),
                CircleObstacle(7.5, -1.55, 0.42, obstacle_id="bottom_3"),
                CircleObstacle(9.4, -1.55, 0.42, obstacle_id="bottom_4"),
                CircleObstacle(11.3, -1.55, 0.42, obstacle_id="bottom_5"),
                CircleObstacle(6.4, 0.45, 0.20, obstacle_id="center_bias"),
            ),
            steps=700,
        ),
    }


def make_perception(case: SimCase, stamp: float) -> PerceptionData:
    obstacles = [
        Obstacle(
            position=Vec3(item.x, item.y, case.start.z),
            radius=item.radius,
            source=item.source,
            obstacle_id=item.obstacle_id,
        )
        for item in case.obstacles
    ]
    return PerceptionData(
        obstacles=obstacles,
        nearest_distance=999.0,
        near_field_danger=False,
        frame_id="world",
        stamp=stamp,
    )


def collision_margin(pos: Vec3, case: SimCase, drone_radius: float) -> float:
    margin = 999.0
    for obs in case.obstacles:
        dist = math.hypot(pos.x - obs.x, pos.y - obs.y)
        margin = min(margin, dist - (drone_radius + obs.radius))
    return margin


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
        state = PlannerState(drone_id=1, position=pos, velocity=vel, stamp=stamp, heading=heading)
        report = planner.plan(state, case.goal, make_perception(case, stamp))
        cmd = report["command"]
        cmd_vel = Vec3.from_mapping(cmd.get("velocity", {}))

        pos = Vec3(pos.x + cmd_vel.x * case.dt, pos.y + cmd_vel.y * case.dt, pos.z + cmd_vel.z * case.dt)
        vel = cmd_vel
        if math.hypot(vel.x, vel.y) > 0.03:
            heading = math.atan2(vel.y, vel.x)

        margin = collision_margin(pos, case, config.drone_radius)
        row = {
            "step": step,
            "t": stamp,
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
            "collision_margin": margin,
            "distance_to_goal": pos.distance_to(case.goal),
        }
        history.append(row)

        source_changed = row["source"] != last_source
        if verbose_every > 0 and (step % verbose_every == 0 or source_changed):
            print(
                f"step={step:04d} pos=({pos.x:5.2f},{pos.y:5.2f}) "
                f"cmd=({vel.x:5.2f},{vel.y:5.2f}) src={row['source']} "
                f"reason={row['reason']} margin={margin:5.2f}"
            )
        last_source = str(row["source"])

        if margin < 0.0:
            status = "collision"
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
    except ImportError as exc:
        raise SystemExit("matplotlib is required for --plot: pip install matplotlib") from exc

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([case.start.x], [case.start.y], "go", label="start")
    ax.plot([case.goal.x], [case.goal.y], "r*", markersize=12, label="goal")
    if history:
        ax.plot([float(r["x"]) for r in history], [float(r["y"]) for r in history], "b-", linewidth=2, label="trajectory")
    for obs in case.obstacles:
        circle = plt.Circle((obs.x, obs.y), obs.radius + config.drone_radius, color="tab:gray", alpha=0.25)
        ax.add_patch(circle)
        ax.plot(obs.x, obs.y, "ko", markersize=3)
    ax.set_title(case.name)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[plot] wrote {path}")


def main() -> int:
    cases = build_cases()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(cases), default="single_pillar")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--csv", default="")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-path", default="")
    parser.add_argument("--strict", action="store_true", help="return a non-zero exit code when the case is not reached")
    parser.add_argument("--verbose-every", type=int, default=20)
    args = parser.parse_args()

    case = cases[args.case]
    if args.steps is not None or args.dt is not None:
        case = SimCase(
            name=case.name,
            start=case.start,
            goal=case.goal,
            obstacles=case.obstacles,
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
        f"min_margin={min(float(r['collision_margin']) for r in history):.2f}"
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
