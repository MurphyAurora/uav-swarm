#!/usr/bin/env python3
"""Analyze ego_like_debug_*.jsonl logs produced by ego_like_planner_framework_node.py.

Usage:
  python3 scripts/debug/analyze_ego_like_log.py ~/ws_xtd2/log/ego_like_debug_YYYYmmdd_HHMMSS.jsonl
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def fget(mapping, path, default=None):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value, digits=3):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as exc:
                print(f"skip bad json line {lineno}: {exc}")
                continue
            records.append(item)
    return records


def main():
    if len(sys.argv) < 2:
        print("usage: python3 scripts/debug/analyze_ego_like_log.py <ego_like_debug.jsonl>")
        return 2
    path = Path(sys.argv[1]).expanduser()
    records = load_records(path)
    if not records:
        print(f"no records in {path}")
        return 1

    by_drone = defaultdict(list)
    for rec in records:
        by_drone[int(rec.get("drone_id", 0))].append(rec)

    print(f"file: {path}")
    print(f"records: {len(records)}")
    print()

    for drone_id, items in sorted(by_drone.items()):
        modes = Counter(fget(r, ["command", "mode"], "") for r in items)
        reasons = Counter(fget(r, ["command", "reason"], "") for r in items)
        trajs = Counter(fget(r, ["command", "source_trajectory"], "") for r in items)
        goal_modes = Counter(fget(r, ["goal_manager", "goal_reason"], "") for r in items)
        frontend_status = Counter(fget(r, ["frontend", "local_astar", "status"], "") for r in items)

        min_clear = min(float(fget(r, ["best", "min_clearance"], 999.0) or 999.0) for r in items)
        min_static = min(float(fget(r, ["best", "min_static_clearance"], 999.0) or 999.0) for r in items)
        min_lidar = min(float(fget(r, ["lidar_stats", "nearest_body_range"], 999.0) or 999.0) for r in items)
        min_safe = min(int(r.get("safe_count", 0)) for r in items)
        max_safe = max(int(r.get("safe_count", 0)) for r in items)
        collision_like = [
            r for r in items
            if float(fget(r, ["best", "min_clearance"], 999.0) or 999.0) < 0.0
            or float(fget(r, ["best", "min_static_clearance"], 999.0) or 999.0) < 0.0
            or float(fget(r, ["lidar_stats", "nearest_body_range"], 999.0) or 999.0) < 0.45
        ]

        print(f"drone x500_{drone_id}")
        print(f"  records: {len(items)}")
        print(f"  command modes: {modes.most_common(6)}")
        print(f"  top reasons: {reasons.most_common(8)}")
        print(f"  top trajectories: {trajs.most_common(8)}")
        print(f"  goal modes: {goal_modes.most_common(6)}")
        print(f"  astar status: {frontend_status.most_common(6)}")
        print(f"  min_clearance={fmt(min_clear)} min_static={fmt(min_static)} min_lidar_range={fmt(min_lidar)} safe_count_range=[{min_safe},{max_safe}]")
        print(f"  collision_like_frames: {len(collision_like)}")

        if collision_like:
            print("  last collision-like frames:")
            for r in collision_like[-8:]:
                cmd = r.get("command", {})
                vel = cmd.get("velocity", {})
                astar = fget(r, ["frontend", "local_astar"], {}) or {}
                goal = r.get("goal_manager", {}) or {}
                print(
                    "    t={t} mode={mode} traj={traj} v=({vx},{vy},{vz}) "
                    "safe={safe}/{cand} clear={clear} static={static} lidar={lidar} "
                    "goal_mode={goal_mode} corridor={corr} astar={astar_status} label={label} reason={reason}".format(
                        t=fmt(r.get("stamp", 0.0), 2),
                        mode=cmd.get("mode", ""),
                        traj=cmd.get("source_trajectory", ""),
                        vx=fmt(vel.get("x", 0.0), 2),
                        vy=fmt(vel.get("y", 0.0), 2),
                        vz=fmt(vel.get("z", 0.0), 2),
                        safe=r.get("safe_count", 0),
                        cand=r.get("candidate_count", 0),
                        clear=fmt(fget(r, ["best", "min_clearance"], 999.0)),
                        static=fmt(fget(r, ["best", "min_static_clearance"], 999.0)),
                        lidar=fmt(fget(r, ["lidar_stats", "nearest_body_range"], 999.0)),
                        goal_mode=goal.get("goal_reason", ""),
                        corr=fmt(goal.get("corridor_side", 0.0), 1),
                        astar_status=astar.get("status", ""),
                        label=astar.get("label", astar.get("replan_reason", "")),
                        reason=cmd.get("reason", ""),
                    )
                )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
