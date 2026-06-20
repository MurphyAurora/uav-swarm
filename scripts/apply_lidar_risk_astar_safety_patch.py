#!/usr/bin/env python3
"""Apply the LiDAR unknown-obstacle safety patch to multi_waypoint2.py.

This helper is intentionally idempotent. It patches the mission controller so that:
1. LOCAL_PLANNER_MODE=risk_astar can still accept safe_primitive_override when LiDAR-TTC modifies primitive velocity.
2. safe_primitive_override is treated as a hard safety velocity only when local risk A* is not already escaping.
3. local risk A* reacts earlier to near obstacles and owns near-field escape/blocked states.
4. final LiDAR safety triggers earlier, logs no_lidar/stale_lidar states, and clears stale local-planner latches.

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


def replace_first_of(text: str, olds: list[str], new: str, label: str) -> str:
    if new in text:
        print(f"[PATCH] {label}: already applied")
        return text
    for old in olds:
        if old in text:
            print(f"[PATCH] {label}: applied")
            return text.replace(old, new, 1)
    raise RuntimeError(f"Cannot find patch target: {label}")


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

    risk_astar_priority_block = "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        # When local risk A* already chose a near-field escape/blocked\n                        # subgoal, it is the most direct point-cloud safety decision.\n                        # Do not let safe primitive override pull the UAV back toward\n                        # the obstacle or cause side-switch oscillation.\n                        if local_planner_label in (\n                            'risk_astar_near_escape',\n                            'risk_astar_escape_back',\n                            'risk_astar_escape_back_left',\n                            'risk_astar_escape_back_right',\n                            'risk_astar_blocked_hold',\n                        ):\n                            avoid_ok = False\n                            avoid_label = 'local_risk_astar_priority'\n                        else:\n                            svx, svy, svz, safe_label, safe_ok = self._safe_primitive_override(\n                                drone_id,\n                                max_age_sec=max(1.2, 4.0 * sleep_dt),\n                            )\n                            if safe_ok:\n                                avx, avy, avz = svx, svy, svz\n                                avoid_label = safe_label\n                                avoid_ok = True\n                            else:\n                                avoid_ok = False\n                                avoid_label = 'disabled_by_risk_astar'"
    text = replace_first_of(
        text,
        [
            "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        avoid_ok = False\n                        avoid_label = 'disabled_by_risk_astar'",
            "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        svx, svy, svz, safe_label, safe_ok = self._safe_primitive_override(\n                            drone_id,\n                            max_age_sec=max(1.2, 4.0 * sleep_dt),\n                        )\n                        if safe_ok:\n                            avx, avy, avz = svx, svy, svz\n                            avoid_label = safe_label\n                            avoid_ok = True\n                        else:\n                            avoid_ok = False\n                            avoid_label = 'disabled_by_risk_astar'",
        ],
        risk_astar_priority_block,
        "prioritize local risk A* near-field escape over safe primitive",
    )

    text = replace_first_of(
        text,
        [
            "if local_planner_label.startswith('risk_astar'):\n                        vz = 0.0\n                        vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.55))",
        ],
        "if local_planner_label.startswith('risk_astar'):\n                        vz = 0.0\n                        if local_planner_label in (\n                            'risk_astar_near_escape',\n                            'risk_astar_escape_back',\n                            'risk_astar_escape_back_left',\n                            'risk_astar_escape_back_right',\n                            'risk_astar_blocked_hold',\n                        ):\n                            vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.35))\n                        else:\n                            vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.55))",
        "slow down local risk A* near-field escape states",
    )

    text = replace_first_of(
        text,
        [
            "if avoid_ok:\n                        if avoid_label.startswith('primitive'):\n                            dvx, dvy, dvz = self._primitive_as_correction(\n                                (vx, vy, vz),\n                                (avx, avy, avz),\n                            )\n                            vx, vy, vz = self._limit_vector(\n                                vx + dvx,\n                                vy + dvy,\n                                vz + dvz,\n                                max_follower_speed,\n                            )\n                        else:\n                            dvx, dvy, dvz = avx, avy, avz\n                            vx, vy, vz = self._limit_vector(\n                                vx + avx,\n                                vy + avy,\n                                vz + avz,\n                                max_follower_speed,\n                            )",
            "if avoid_ok:\n                        if avoid_label in ('safe_primitive_override', 'safe_primitive_only'):\n                            # LiDAR-TTC has already changed the primitive velocity, so this is\n                            # a near-field safety command. Use it as a hard override instead\n                            # of a weak correction on top of target tracking.\n                            dvx = avx - vx\n                            dvy = avy - vy\n                            dvz = avz - vz\n                            vx, vy, vz = self._limit_vector(\n                                avx,\n                                avy,\n                                avz,\n                                min(max_follower_speed, 0.55),\n                            )\n                        elif avoid_label.startswith('primitive'):\n                            dvx, dvy, dvz = self._primitive_as_correction(\n                                (vx, vy, vz),\n                                (avx, avy, avz),\n                                max_delta=0.45,\n                            )\n                            vx, vy, vz = self._limit_vector(\n                                vx + dvx,\n                                vy + dvy,\n                                vz + dvz,\n                                max_follower_speed,\n                            )\n                        else:\n                            dvx, dvy, dvz = avx, avy, avz\n                            vx, vy, vz = self._limit_vector(\n                                vx + avx,\n                                vy + avy,\n                                vz + avz,\n                                max_follower_speed,\n                            )",
        ],
        "if avoid_ok:\n                        if avoid_label in ('safe_primitive_override', 'safe_primitive_only'):\n                            # LiDAR-TTC has already changed the primitive velocity. Use it\n                            # as a hard safety command only when local risk A* is not already\n                            # in near-field escape/blocked priority mode.\n                            dvx = avx - vx\n                            dvy = avy - vy\n                            dvz = avz - vz\n                            vx, vy, vz = self._limit_vector(\n                                avx,\n                                avy,\n                                avz,\n                                min(max_follower_speed, 0.45),\n                            )\n                        elif avoid_label.startswith('primitive'):\n                            dvx, dvy, dvz = self._primitive_as_correction(\n                                (vx, vy, vz),\n                                (avx, avy, avz),\n                                max_delta=0.45,\n                            )\n                            vx, vy, vz = self._limit_vector(\n                                vx + dvx,\n                                vy + dvy,\n                                vz + dvz,\n                                max_follower_speed,\n                            )\n                        else:\n                            dvx, dvy, dvz = avx, avy, avz\n                            vx, vy, vz = self._limit_vector(\n                                vx + avx,\n                                vy + avy,\n                                vz + avz,\n                                max_follower_speed,\n                            )",
        "treat safe primitive as hard safety override outside local escape priority",
    )

    text = replace_first_of(
        text,
        [
            "self.final_safety_until[int(drone_id)] = now_t + 1.0\n            self.final_safety_velocity[int(drone_id)] = (nvx, nvy, 0.0)\n            return nvx, nvy, 0.0, 'final_corridor_stop'",
        ],
        "self.final_safety_until[int(drone_id)] = now_t + 1.0\n            self.final_safety_velocity[int(drone_id)] = (nvx, nvy, 0.0)\n            self._local_planner_subgoals.pop(int(drone_id), None)\n            self._local_planner_side_latch.pop(int(drone_id), None)\n            return nvx, nvy, 0.0, 'final_corridor_stop'",
        "clear local planner latch on final corridor stop",
    )

    text = replace_first_of(
        text,
        [
            "self.final_safety_until[int(drone_id)] = now_t + 1.2\n            self.final_safety_velocity[int(drone_id)] = (nvx, nvy, 0.0)\n            return nvx, nvy, 0.0, 'final_near_stop'",
        ],
        "self.final_safety_until[int(drone_id)] = now_t + 1.2\n            self.final_safety_velocity[int(drone_id)] = (nvx, nvy, 0.0)\n            self._local_planner_subgoals.pop(int(drone_id), None)\n            self._local_planner_side_latch.pop(int(drone_id), None)\n            return nvx, nvy, 0.0, 'final_near_stop'",
        "clear local planner latch on final near stop",
    )

    text = replace_first_of(
        text,
        [
            "if latch_sec > 0.0:\n            self.final_safety_until[int(drone_id)] = now_t + latch_sec\n            self.final_safety_velocity[int(drone_id)] = (nvx, nvy, 0.0)\n        return nvx, nvy, 0.0, mode",
        ],
        "if latch_sec > 0.0:\n            self.final_safety_until[int(drone_id)] = now_t + latch_sec\n            self.final_safety_velocity[int(drone_id)] = (nvx, nvy, 0.0)\n            self._local_planner_subgoals.pop(int(drone_id), None)\n            self._local_planner_side_latch.pop(int(drone_id), None)\n        return nvx, nvy, 0.0, mode",
        "clear local planner latch on final hard stop",
    )

    text = replace_first_of(
        text,
        [
            "if final_safety_mode.startswith('final_'):\n            now_t = time.time()\n            if now_t - self._last_failsafe_log_t >= 1.0:\n                self._last_failsafe_log_t = now_t\n                self.get_logger().info(\n                    f'final lidar safety: x500_{drone_id} mode={final_safety_mode}, '\n                    f'raw_v=({raw_vx:.2f},{raw_vy:.2f},{raw_vz:.2f}), '\n                    f'safe_v=({vx:.2f},{vy:.2f},{vz:.2f})'\n                )",
        ],
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
