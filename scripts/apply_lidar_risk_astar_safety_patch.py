#!/usr/bin/env python3
"""Apply the LiDAR unknown-obstacle safety patch to multi_waypoint2.py.

This helper is intentionally idempotent. It patches the mission controller so that:
1. LOCAL_PLANNER_MODE=risk_astar can still accept safe_primitive_override when LiDAR-TTC modifies primitive velocity.
2. Local risk A* near-field escape has priority over safe primitive.
3. Dense-forest LiDAR risk uses a narrower local crop and smaller inflation so narrow passages are not over-blocked.
4. Final LiDAR safety uses a two-stage stop/recovery latch: short hard stop, then non-zero side/back recovery.

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


def replace_between(text: str, start: str, end: str, new: str, label: str) -> str:
    if new in text:
        print(f"[PATCH] {label}: already applied")
        return text
    s = text.find(start)
    if s < 0:
        raise RuntimeError(f"Cannot find patch start: {label}")
    e = text.find(end, s)
    if e < 0:
        raise RuntimeError(f"Cannot find patch end: {label}")
    print(f"[PATCH] {label}: applied")
    return text[:s] + new + text[e:]


FINAL_LIDAR_SAFETY_FILTER_V4 = '''    def _final_lidar_safety_filter(self, drone_id, vx, vy, vz):
        now_t = time.time()
        key = int(drone_id)
        latched_until = self.final_safety_until.get(key, 0.0)
        cloud = self.latest_lidar_clouds.get(key)
        st = self.latest_states.get(key)
        if not hasattr(self, '_final_recovery_since'):
            self._final_recovery_since = {}
        if not hasattr(self, '_final_recovery_side'):
            self._final_recovery_side = {}

        if cloud is None or st is None:
            if now_t < latched_until:
                safe_v = self.final_safety_velocity.get(key, (0.0, 0.0, 0.0))
                return safe_v[0], safe_v[1], safe_v[2], 'final_latched_stop'
            return vx, vy, vz, 'no_lidar'
        if time.time() - float(cloud.get('stamp', 0.0)) > 2.0:
            if now_t < latched_until:
                safe_v = self.final_safety_velocity.get(key, (0.0, 0.0, 0.0))
                return safe_v[0], safe_v[1], safe_v[2], 'final_latched_stop'
            return vx, vy, vz, 'stale_lidar'

        heading = self._finite_float(st.get('heading', 0.0))
        bvx, bvy = self._world_to_body_xy(vx, vy, heading)
        speed_xy = math.hypot(bvx, bvy)
        if speed_xy >= 0.08:
            dir_x = bvx / speed_xy
            dir_y = bvy / speed_xy
        else:
            # Use a stable default direction while latched/stopped so recovery can still be non-zero.
            dir_x, dir_y = 1.0, 0.0

        warning_ttc = 3.5
        emergency_ttc = 1.6
        safety_radius = 1.05
        # Keep final safety near-field only. A too-wide corridor can merge two pillars
        # and hide a valid gap, causing permanent stop in forest3.
        hard_stop_distance = 1.35
        corridor_stop_distance = 2.40
        corridor_half_width = 0.95
        cone_cos = math.cos(math.radians(70.0))
        worst = None
        nearest = None
        nearest_point = None
        corridor_hit = None

        for idx, point in enumerate(point_cloud2.read_points(cloud['msg'], field_names=('x', 'y', 'z'), skip_nans=True)):
            if idx >= 2500:
                break
            px = self._finite_float(point[0])
            py = self._finite_float(point[1])
            pz = self._finite_float(point[2])
            horizontal = math.hypot(px, py)
            if horizontal < 0.65 or horizontal > 4.50:
                continue
            if pz < -1.8 or pz > 2.2:
                continue
            if nearest is None or horizontal < nearest:
                nearest = horizontal
                nearest_point = (px, py, pz)

            forward = px * dir_x + py * dir_y
            lateral = abs(-dir_y * px + dir_x * py)
            if speed_xy >= 0.08 and 0.2 < forward < corridor_stop_distance and lateral < corridor_half_width:
                if corridor_hit is None or forward < corridor_hit[0]:
                    corridor_hit = (forward, lateral, pz)

            ux = px / max(horizontal, 1e-6)
            uy = py / max(horizontal, 1e-6)
            alignment = ux * dir_x + uy * dir_y
            if alignment < cone_cos:
                continue
            closing = bvx * ux + bvy * uy
            if closing <= 0.08:
                continue
            ttc = max(0.0, horizontal - safety_radius) / max(closing, 1e-6)
            hard = horizontal <= hard_stop_distance
            if ttc > warning_ttc and not hard:
                continue
            if worst is None or ttc < worst['ttc']:
                worst = {'ttc': ttc, 'ux': ux, 'uy': uy, 'hard': hard}

        def choose_recovery_body():
            side = self._final_recovery_side.get(key)
            if side not in (-1.0, 1.0):
                if nearest_point is not None:
                    # If obstacle is on left(+y), escape right(-y), and vice versa.
                    side = -1.0 if nearest_point[1] >= 0.0 else 1.0
                else:
                    side = 1.0
                self._final_recovery_side[key] = side
            # Slight side motion plus slight backward component. This is intentionally
            # non-zero so the UAV cannot remain in final_latched_stop forever.
            rbvx = -0.12 * dir_x + 0.32 * side * (-dir_y)
            rbvy = -0.12 * dir_y + 0.32 * side * dir_x
            rbvx, rbvy, _ = self._limit_vector(rbvx, rbvy, 0.0, 0.36)
            return rbvx, rbvy

        if now_t < latched_until:
            safe_v = self.final_safety_velocity.get(key, (0.0, 0.0, 0.0))
            # If a zero latch has lasted long enough, promote it to a non-zero recovery
            # escape. This is the anti-deadlock part of the state machine.
            if math.hypot(float(safe_v[0]), float(safe_v[1])) < 0.04:
                start_t = self._final_recovery_since.setdefault(key, now_t)
                if now_t - start_t >= 0.45:
                    rbvx, rbvy = choose_recovery_body()
                    nvx, nvy = self._body_to_world_xy(rbvx, rbvy, heading)
                    self.final_safety_until[key] = now_t + 0.55
                    self.final_safety_velocity[key] = (nvx, nvy, 0.0)
                    return nvx, nvy, 0.0, 'final_recovery_escape'
            return safe_v[0], safe_v[1], safe_v[2], 'final_latched_stop'

        if nearest is not None and nearest > 1.85:
            self._final_recovery_since.pop(key, None)
            self._final_recovery_side.pop(key, None)

        if speed_xy < 0.08:
            return vx, vy, vz, 'pass'

        if corridor_hit is not None:
            start_t = self._final_recovery_since.setdefault(key, now_t)
            if now_t - start_t < 0.45:
                nbvx, nbvy = 0.0, 0.0
                mode = 'final_corridor_stop'
                latch_sec = 0.45
            else:
                nbvx, nbvy = choose_recovery_body()
                mode = 'final_recovery_escape'
                latch_sec = 0.55
            nvx, nvy = self._body_to_world_xy(nbvx, nbvy, heading)
            self.final_safety_until[key] = now_t + latch_sec
            self.final_safety_velocity[key] = (nvx, nvy, 0.0)
            self._local_planner_subgoals.pop(key, None)
            self._local_planner_side_latch.pop(key, None)
            return nvx, nvy, 0.0, mode

        if nearest is not None and nearest <= hard_stop_distance and nearest_point is not None:
            px, py, _ = nearest_point
            away_x = -px / max(nearest, 1e-6)
            away_y = -py / max(nearest, 1e-6)
            # Slower but non-zero near-field recovery. It should move out of contact,
            # not bounce hard into a neighboring obstacle.
            nbvx = 0.22 * away_x
            nbvy = 0.22 * away_y
            nvx, nvy = self._body_to_world_xy(nbvx, nbvy, heading)
            self._final_recovery_since.setdefault(key, now_t)
            self.final_safety_until[key] = now_t + 0.75
            self.final_safety_velocity[key] = (nvx, nvy, 0.0)
            self._local_planner_subgoals.pop(key, None)
            self._local_planner_side_latch.pop(key, None)
            return nvx, nvy, 0.0, 'final_near_stop'

        if worst is None:
            self._final_recovery_since.pop(key, None)
            self._final_recovery_side.pop(key, None)
            return vx, vy, vz, 'pass'

        ux = worst['ux']
        uy = worst['uy']
        toward = max(0.0, bvx * ux + bvy * uy)
        side_x = bvx - toward * ux
        side_y = bvy - toward * uy
        if worst['hard'] or worst['ttc'] <= emergency_ttc:
            nbvx = 0.08 * side_x - 0.35 * ux
            nbvy = 0.08 * side_y - 0.35 * uy
            mode = 'final_hard_stop'
            latch_sec = 0.75
        else:
            scale = (warning_ttc - worst['ttc']) / max(warning_ttc - emergency_ttc, 1e-6)
            remove = toward * min(1.0, max(0.0, scale * 1.25))
            nbvx = bvx - remove * ux
            nbvy = bvy - remove * uy
            mode = 'final_project'
            latch_sec = 0.0
        nvx, nvy = self._body_to_world_xy(nbvx, nbvy, heading)
        if latch_sec > 0.0:
            self.final_safety_until[key] = now_t + latch_sec
            self.final_safety_velocity[key] = (nvx, nvy, 0.0)
            self._local_planner_subgoals.pop(key, None)
            self._local_planner_side_latch.pop(key, None)
        return nvx, nvy, 0.0, mode

'''


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    path = repo / "src" / "xtd2_mission" / "xtd2_mission" / "multi_waypoint2.py"
    text = path.read_text(encoding="utf-8")
    original = text

    # Local risk A*: narrower crop and smaller inflation. The previous 6m x 8m crop
    # with 3-cell inflation could merge neighboring pillars and ignore a valid gap.
    text = replace_first_of(
        text,
        [
            "horizon = 6.0\n        half_width = 4.0\n        res = 0.5",
            "horizon = 4.5\n        half_width = 3.0\n        res = 0.5",
        ],
        "horizon = 4.5\n        half_width = 3.0\n        res = 0.5",
        "narrow local risk astar lidar crop",
    )
    text = replace_first_of(
        text,
        ["inflation_cells = 3", "inflation_cells = 2"],
        "inflation_cells = 2",
        "reduce local risk astar obstacle inflation",
    )
    text = replace_first_of(
        text,
        [
            "if target_corridor_clear() and (nearest is None or nearest[0] > 1.6):",
            "if target_corridor_clear() and (nearest is None or nearest[0] > 2.4):",
            "if target_corridor_clear() and (nearest is None or nearest[0] > 2.0):",
        ],
        "if target_corridor_clear() and (nearest is None or nearest[0] > 2.0):",
        "set balanced risk_astar corridor-clear bypass threshold",
    )
    text = replace_first_of(
        text,
        [
            "if cloud is None or time.time() - float(cloud.get('stamp', 0.0)) > 0.8:\n            return ref_x, ref_y, ref_z, 'no_lidar'",
            "if cloud is None or time.time() - float(cloud.get('stamp', 0.0)) > 2.0:\n            return ref_x, ref_y, ref_z, 'no_lidar'",
        ],
        "if cloud is None or time.time() - float(cloud.get('stamp', 0.0)) > 2.0:\n            return ref_x, ref_y, ref_z, 'no_lidar'",
        "relax local risk astar lidar stale threshold",
    )

    # Near-field local recovery: do not move forward, but allow side-only escape if back is blocked.
    text = replace_first_of(
        text,
        [
            "lx = 1.0 * away_x\n            ly = 1.0 * away_y",
            "lx = 1.4 * away_x\n            ly = 1.4 * away_y",
            "# Near-field recovery must not move further forward into a dense pillar field.\n            # Keep only backward / side-back options and choose the clearest available one.\n            recovery_candidates = [\n                ('risk_astar_escape_back', -1.2, 0.0),\n                ('risk_astar_escape_back_left', -0.9, 0.9),\n                ('risk_astar_escape_back_right', -0.9, -0.9),\n            ]\n            best_recovery = None\n            for cand_label, cand_x, cand_y in recovery_candidates:\n                if not segment_clear_to(cand_x, cand_y):\n                    continue\n                cand_score = math.hypot(cand_x, cand_y)\n                if cand_y > 0.0:\n                    cand_score += 0.25 * sector_min['left']\n                elif cand_y < 0.0:\n                    cand_score += 0.25 * sector_min['right']\n                else:\n                    cand_score += 0.25 * sector_min['back']\n                if best_recovery is None or cand_score > best_recovery[0]:\n                    best_recovery = (cand_score, cand_label, cand_x, cand_y)\n            if best_recovery is None:\n                lx, ly = -0.8, 0.0\n                recovery_label = 'risk_astar_escape_back_forced'\n            else:\n                _, recovery_label, lx, ly = best_recovery",
        ],
        "# Near-field recovery must not move further forward into a dense pillar field.\n            # Allow backward, side-back, and side-only options so the UAV can escape\n            # a narrow gap instead of remaining stuck when back is blocked.\n            recovery_candidates = [\n                ('risk_astar_escape_back', -1.0, 0.0),\n                ('risk_astar_escape_back_left', -0.7, 0.75),\n                ('risk_astar_escape_back_right', -0.7, -0.75),\n                ('risk_astar_escape_side_left', 0.0, 0.85),\n                ('risk_astar_escape_side_right', 0.0, -0.85),\n            ]\n            best_recovery = None\n            for cand_label, cand_x, cand_y in recovery_candidates:\n                if not segment_clear_to(cand_x, cand_y):\n                    continue\n                cand_score = math.hypot(cand_x, cand_y)\n                if cand_y > 0.0:\n                    cand_score += 0.30 * sector_min['left']\n                elif cand_y < 0.0:\n                    cand_score += 0.30 * sector_min['right']\n                else:\n                    cand_score += 0.30 * sector_min['back']\n                if best_recovery is None or cand_score > best_recovery[0]:\n                    best_recovery = (cand_score, cand_label, cand_x, cand_y)\n            if best_recovery is None:\n                side = 'left' if sector_min['left'] >= sector_min['right'] else 'right'\n                lx, ly = 0.0, (0.65 if side == 'left' else -0.65)\n                recovery_label = f'risk_astar_escape_side_{side}_forced'\n            else:\n                _, recovery_label, lx, ly = best_recovery",
        "make near-field risk_astar recovery non-forward but not deadlocked",
    )
    text = replace_first_of(
        text,
        ["return store_subgoal(lx, ly, 'risk_astar_near_escape', now_t)", "return store_subgoal(lx, ly, recovery_label, now_t)"],
        "return store_subgoal(lx, ly, recovery_label, now_t)",
        "label near-field recovery subgoal",
    )

    # Replace the final safety method as a whole. This is more reliable than stacking
    # more one-line edits after several experimental versions.
    text = replace_between(
        text,
        "    def _final_lidar_safety_filter(self, drone_id, vx, vy, vz):\n",
        "    def _estimate_virtual_ref_from_states",
        FINAL_LIDAR_SAFETY_FILTER_V4,
        "replace final lidar safety with stop-then-nonzero-recovery state machine",
    )

    risk_astar_priority_block = "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        # When local risk A* already chose a near-field escape/blocked\n                        # subgoal, it is the most direct point-cloud safety decision.\n                        if local_planner_label in (\n                            'risk_astar_near_escape',\n                            'risk_astar_escape_back',\n                            'risk_astar_escape_back_left',\n                            'risk_astar_escape_back_right',\n                            'risk_astar_escape_back_forced',\n                            'risk_astar_escape_side_left',\n                            'risk_astar_escape_side_right',\n                            'risk_astar_escape_side_left_forced',\n                            'risk_astar_escape_side_right_forced',\n                            'risk_astar_blocked_hold',\n                        ):\n                            avoid_ok = False\n                            avoid_label = 'local_risk_astar_priority'\n                        else:\n                            svx, svy, svz, safe_label, safe_ok = self._safe_primitive_override(\n                                drone_id,\n                                max_age_sec=max(1.2, 4.0 * sleep_dt),\n                            )\n                            if safe_ok:\n                                avx, avy, avz = svx, svy, svz\n                                avoid_label = safe_label\n                                avoid_ok = True\n                            else:\n                                avoid_ok = False\n                                avoid_label = 'disabled_by_risk_astar'"
    text = replace_first_of(
        text,
        [
            "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        avoid_ok = False\n                        avoid_label = 'disabled_by_risk_astar'",
            "if self.local_planner_mode in ('risk_astar', 'astar'):\n                        # When local risk A* already chose a near-field escape/blocked\n                        # subgoal, it is the most direct point-cloud safety decision.\n                        # Do not let safe primitive override pull the UAV back toward\n                        # the obstacle or cause side-switch oscillation.\n                        if local_planner_label in (\n                            'risk_astar_near_escape',\n                            'risk_astar_escape_back',\n                            'risk_astar_escape_back_left',\n                            'risk_astar_escape_back_right',\n                            'risk_astar_escape_back_forced',\n                            'risk_astar_blocked_hold',\n                        ):\n                            avoid_ok = False\n                            avoid_label = 'local_risk_astar_priority'\n                        else:\n                            svx, svy, svz, safe_label, safe_ok = self._safe_primitive_override(\n                                drone_id,\n                                max_age_sec=max(1.2, 4.0 * sleep_dt),\n                            )\n                            if safe_ok:\n                                avx, avy, avz = svx, svy, svz\n                                avoid_label = safe_label\n                                avoid_ok = True\n                            else:\n                                avoid_ok = False\n                                avoid_label = 'disabled_by_risk_astar'",
        ],
        risk_astar_priority_block,
        "prioritize expanded local risk A* recovery labels over safe primitive",
    )

    text = replace_first_of(
        text,
        [
            "if local_planner_label.startswith('risk_astar'):\n                        vz = 0.0\n                        vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.55))",
            "if local_planner_label.startswith('risk_astar'):\n                        vz = 0.0\n                        if local_planner_label in (\n                            'risk_astar_near_escape',\n                            'risk_astar_escape_back',\n                            'risk_astar_escape_back_left',\n                            'risk_astar_escape_back_right',\n                            'risk_astar_escape_back_forced',\n                            'risk_astar_blocked_hold',\n                        ):\n                            vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.30))\n                        else:\n                            vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.55))",
        ],
        "if local_planner_label.startswith('risk_astar'):\n                        vz = 0.0\n                        if local_planner_label in (\n                            'risk_astar_near_escape',\n                            'risk_astar_escape_back',\n                            'risk_astar_escape_back_left',\n                            'risk_astar_escape_back_right',\n                            'risk_astar_escape_back_forced',\n                            'risk_astar_escape_side_left',\n                            'risk_astar_escape_side_right',\n                            'risk_astar_escape_side_left_forced',\n                            'risk_astar_escape_side_right_forced',\n                            'risk_astar_blocked_hold',\n                        ):\n                            vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.32))\n                        else:\n                            vx, vy, _ = self._limit_vector(vx, vy, 0.0, min(max_follower_speed, 0.55))",
        "slow expanded local risk A* near-field recovery states",
    )

    text = replace_first_of(
        text,
        [
            "if final_safety_mode.startswith('final_'):\n            now_t = time.time()\n            if now_t - self._last_failsafe_log_t >= 1.0:\n                self._last_failsafe_log_t = now_t\n                self.get_logger().info(\n                    f'final lidar safety: x500_{drone_id} mode={final_safety_mode}, '\n                    f'raw_v=({raw_vx:.2f},{raw_vy:.2f},{raw_vz:.2f}), '\n                    f'safe_v=({vx:.2f},{vy:.2f},{vz:.2f})'\n                )",
            "now_t = time.time()\n        if final_safety_mode.startswith('final_'):\n            if now_t - self._last_failsafe_log_t >= 1.0:\n                self._last_failsafe_log_t = now_t\n                self.get_logger().info(\n                    f'final lidar safety: x500_{drone_id} mode={final_safety_mode}, '\n                    f'raw_v=({raw_vx:.2f},{raw_vy:.2f},{raw_vz:.2f}), '\n                    f'safe_v=({vx:.2f},{vy:.2f},{vz:.2f})'\n                )\n        elif final_safety_mode in ('no_lidar', 'stale_lidar') and now_t - self._last_failsafe_log_t >= 2.0:\n            self._last_failsafe_log_t = now_t\n            self.get_logger().warn(\n                f'final lidar safety inactive: x500_{drone_id} mode={final_safety_mode}, '\n                f'raw_v=({raw_vx:.2f},{raw_vy:.2f},{raw_vz:.2f})'\n            )",
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
