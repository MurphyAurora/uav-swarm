#!/usr/bin/env python3
"""Apply the LiDAR unknown-obstacle safety patch to multi_waypoint2.py.

This helper is intentionally idempotent. It patches the mission controller so that:
1. LOCAL_PLANNER_MODE=risk_astar can still accept safe_primitive_override when LiDAR-TTC modifies primitive velocity.
2. safe_primitive_override is treated as a hard safety velocity, not only a weak correction.
3. local risk A* reacts earlier to near obstacles.
4. final LiDAR safety triggers earlier and logs no_lidar/stale_lidar states.

It is used by scripts/run_sando_forest3.sh before launching the forest3 experiment.
"""
from __future__ import annotations

import pathlib
import sys


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        print(f"[PATCH] {label}: already applied")
        return text
    if old not in text:
        raise RuntimeError(f"Cannot find patch target: {label}")
    print(f"[PATCH] {label}: applied")
    return text.replace(old, new, 1)


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    path = repo / "src" / "xtd2_mission" / "xtd2_mission" / "multi_waypoint2.py"
    text = path.read_text(encoding="utf-8")
    original = text

    text = replace_once(
        text,
        "if target_corridor_clear() and (nearest is None or nearest[0] > 1.6):",
        "if target_corridor_clear() and (nearest is None or nearest[0] > 2.4):",
        "make risk_astar corridor-clear bypass more conservative",
    )

    text = replace_once(
        text,
        "if nearest is not None and nearest[0] < 1.05:",
        "if nearest is not None and nearest[0] < 1.65:",
        "trigger near-obstacle escape earlier",
    )

    text = replace_once(
        text,
        "lx = 1.0 * away_x\n            ly = 1.0 * away_y",
        "lx = 1.4 * away_x\n            ly = 1.4 * away_y",
        "increase near-escape subgoal distance",
    )

    text = replace_once(
        text,
        "warning_ttc = 3.2\n        emergency_ttc = 1.4\n        safety_radius = 0.95\n        # Only use radial hard-stop for points that are almost touching the UAV.\n        # Directional danger is handled by the corridor/TTC checks below; otherwise\n        # side/back points make the drone brake and wander even when the path ahead is free.\n        hard_stop_distance = 1.25\n        corridor_stop_distance = 2.60\n        corridor_half_width = 1.20\n        cone_cos = math.cos(math.radians(60.0))",
        "warning_ttc = 3.5\n        emergency_ttc = 1.6\n        safety_radius = 1.05\n        # Use conservative near-field limits for dense forest3 pillars.\n        # Corridor stop protects the current commanded direction before PX4 inertia\n        # carries the UAV into a pillar.\n        hard_stop_distance = 1.35\n        corridor_stop_distance = 3.20\n        corridor_half_width = 1.35\n        cone_cos = math.cos(math.radians(70.0))",
        "make final LiDAR safety more conservative",
    )

    text = replace_once(
        text,
        "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        avoid_ok = False\n                        avoid_label = 'disabled_by_risk_astar'",
        "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        svx, svy, svz, safe_label, safe_ok = self._safe_primitive_override(\n                            drone_id,\n                            max_age_sec=max(1.2, 4.0 * sleep_dt),\n                        )\n                        if safe_ok:\n                            avx, avy, avz = svx, svy, svz\n                            avoid_label = safe_label\n                            avoid_ok = True\n                        else:\n                            avoid_ok = False\n                            avoid_label = 'disabled_by_risk_astar'",
        "allow safe primitive override while risk_astar is active",
    )

    text = replace_once(
        text,
        "if avoid_ok:\n                        if avoid_label.startswith('primitive'):\n                            dvx, dvy, dvz = self._primitive_as_correction(\n                                (vx, vy, vz),\n                                (avx, avy, avz),\n                            )\n                            vx, vy, vz = self._limit_vector(\n                                vx + dvx,\n                                vy + dvy,\n                                vz + dvz,\n                                max_follower_speed,\n                            )\n                        else:\n                            dvx, dvy, dvz = avx, avy, avz\n                            vx, vy, vz = self._limit_vector(\n                                vx + avx,\n                                vy + avy,\n                                vz + avz,\n                                max_follower_speed,\n                            )",
        "if avoid_ok:\n                        if avoid_label in ('safe_primitive_override', 'safe_primitive_only'):\n                            # LiDAR-TTC has already changed the primitive velocity, so this is\n                            # a near-field safety command. Use it as a hard override instead\n                            # of a weak correction on top of target tracking.\n                            dvx = avx - vx\n                            dvy = avy - vy\n                            dvz = avz - vz\n                            vx, vy, vz = self._limit_vector(\n                                avx,\n                                avy,\n                                avz,\n                                min(max_follower_speed, 0.55),\n                            )\n                        elif avoid_label.startswith('primitive'):\n                            dvx, dvy, dvz = self._primitive_as_correction(\n                                (vx, vy, vz),\n                                (avx, avy, avz),\n                                max_delta=0.45,\n                            )\n                            vx, vy, vz = self._limit_vector(\n                                vx + dvx,\n                                vy + dvy,\n                                vz + dvz,\n                                max_follower_speed,\n                            )\n                        else:\n                            dvx, dvy, dvz = avx, avy, avz\n                            vx, vy, vz = self._limit_vector(\n                                vx + avx,\n                                vy + avy,\n                                vz + avz,\n                                max_follower_speed,\n                            )",
        "treat safe primitive as hard safety override",
    )

    text = replace_once(
        text,
        "if final_safety_mode.startswith('final_'):\n            now_t = time.time()\n            if now_t - self._last_failsafe_log_t >= 1.0:\n                self._last_failsafe_log_t = now_t\n                self.get_logger().info(\n                    f'final lidar safety: x500_{drone_id} mode={final_safety_mode}, '\n                    f'raw_v=({raw_vx:.2f},{raw_vy:.2f},{raw_vz:.2f}), '\n                    f'safe_v=({vx:.2f},{vy:.2f},{vz:.2f})'\n                )",
        "now_t = time.time()\n        if final_safety_mode.startswith('final_'):\n            if now_t - self._last_failsafe_log_t >= 1.0:\n                self._last_failsafe_log_t = now_t\n                self.get_logger().info(\n                    f'final lidar safety: x500_{drone_id} mode={final_safety_mode}, '\n                    f'raw_v=({raw_vx:.2f},{raw_vy:.2f},{raw_vz:.2f}), '\n                    f'safe_v=({vx:.2f},{vy:.2f},{vz:.2f})'\n                )\n        elif final_safety_mode in ('no_lidar', 'stale_lidar') and now_t - self._last_failsafe_log_t >= 2.0:\n            self._last_failsafe_log_t = now_t\n            self.get_logger().warn(\n                f'final lidar safety inactive: x500_{drone_id} mode={final_safety_mode}, '\n                f'raw_v=({raw_vx:.2f},{raw_vy:.2f},{raw_vz:.2f})'\n            )",
        "log final safety inactive states",
    )

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"[PATCH] updated {path.relative_to(repo)}")
    else:
        print("[PATCH] no changes needed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[PATCH][ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
