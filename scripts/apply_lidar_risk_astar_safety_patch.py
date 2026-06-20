#!/usr/bin/env python3
"""Apply the forest3 LiDAR avoidance patch to multi_waypoint2.py.

V5 changes the experiment from many competing velocity modifiers into one
unified candidate-velocity selector:
- generate candidate body-frame velocities;
- rollout each candidate for a short horizon;
- hard-filter candidates with insufficient LiDAR clearance;
- score the remaining candidates by target alignment, clearance, and smoothness;
- publish only one selected velocity, with final LiDAR safety kept as a last guard.

This is intentionally an experiment patch, not a permanent architecture rewrite.
It is used by scripts/run_sando_forest3.sh before launching the forest3 run.
"""
from __future__ import annotations

import pathlib
import sys


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


UNIFIED_AND_FINAL_METHODS_V5 = '''    def _unified_lidar_velocity_selector(
        self,
        drone_id,
        st,
        base_vx,
        base_vy,
        base_vz,
        ref_x,
        ref_y,
        ref_z,
        max_speed,
        local_planner_label='off',
    ):
        """Select one final local velocity from LiDAR clearance rollouts.

        This replaces stacked risk_astar / primitive / lidar_ttc arbitration in
        direct-pose forest3 experiments. All safety decisions are made before
        publishing a single velocity command.
        """
        key = int(drone_id)
        cloud = self.latest_lidar_clouds.get(key)
        heading = self._finite_float(st.get('heading', 0.0))
        now_t = time.time()
        if cloud is None or now_t - float(cloud.get('stamp', 0.0)) > 2.0:
            vx, vy, vz = self._limit_vector(base_vx, base_vy, base_vz, min(float(max_speed), 0.25))
            return vx, vy, vz, 'unified_no_lidar_slow'

        # Target direction in the UAV body frame.
        tx = float(ref_x) - float(st['x'])
        ty = float(ref_y) - float(st['y'])
        tz = float(ref_z) - float(st['z'])
        tbx, tby = self._world_to_body_xy(tx, ty, heading)
        target_norm = math.hypot(tbx, tby)
        if target_norm < 1e-3:
            target_dir = (1.0, 0.0)
        else:
            target_dir = (tbx / target_norm, tby / target_norm)

        bbx, bby = self._world_to_body_xy(base_vx, base_vy, heading)
        max_xy_speed = min(float(max_speed), 0.32)
        cruise = min(0.30, max_xy_speed)
        side = min(0.26, max_xy_speed)
        back = min(0.22, max_xy_speed)

        def rot(vec, deg):
            c = math.cos(math.radians(deg))
            s = math.sin(math.radians(deg))
            return (vec[0] * c - vec[1] * s, vec[0] * s + vec[1] * c)

        # Candidate velocities in body frame. The first three follow the target
        # direction, the rest are explicit side/back recovery options.
        td = target_dir
        candidates = []
        for label, scale, vec in [
            ('target_slow', cruise, td),
            ('target_left', cruise * 0.82, rot(td, 45.0)),
            ('target_right', cruise * 0.82, rot(td, -45.0)),
            ('side_left', side, (0.0, 1.0)),
            ('side_right', side, (0.0, -1.0)),
            ('back_left', back, (-0.70, 0.70)),
            ('back_right', back, (-0.70, -0.70)),
            ('back', back, (-1.0, 0.0)),
            ('stop', 0.0, (0.0, 0.0)),
        ]:
            norm = math.hypot(vec[0], vec[1])
            if norm < 1e-6:
                cbx, cby = 0.0, 0.0
            else:
                cbx = scale * vec[0] / norm
                cby = scale * vec[1] / norm
            candidates.append((label, cbx, cby))

        points = []
        nearest = None
        for idx, point in enumerate(point_cloud2.read_points(cloud['msg'], field_names=('x', 'y', 'z'), skip_nans=True)):
            if idx >= 3000:
                break
            px = self._finite_float(point[0])
            py = self._finite_float(point[1])
            pz = self._finite_float(point[2])
            horizontal = math.hypot(px, py)
            # Keep the local control horizon compact; far points can make narrow
            # passages look blocked, but near points still protect collision.
            if horizontal < 0.55 or horizontal > 4.20:
                continue
            if pz < -1.6 or pz > 2.1:
                continue
            points.append((px, py))
            if nearest is None or horizontal < nearest:
                nearest = horizontal

        if not points:
            vx, vy = self._body_to_world_xy(cruise * td[0], cruise * td[1], heading)
            vz = max(-0.15, min(0.15, float(base_vz)))
            return vx, vy, vz, 'unified_no_points_target_slow'

        rollout_horizon = 1.8
        rollout_dt = 0.30
        effective_radius = 0.85
        preferred_radius = 1.20
        emergency_radius = 0.62
        samples = [rollout_dt * i for i in range(1, int(rollout_horizon / rollout_dt) + 1)]
        evaluated = []
        safe = []

        for label, cbx, cby in candidates:
            min_dist = 99.0
            for t in samples:
                px_uav = cbx * t
                py_uav = cby * t
                for ox, oy in points:
                    d = math.hypot(ox - px_uav, oy - py_uav)
                    if d < min_dist:
                        min_dist = d
            speed = math.hypot(cbx, cby)
            if speed < 1e-6:
                align = -0.10
            else:
                align = (cbx * td[0] + cby * td[1]) / max(speed, 1e-6)
            smooth = math.hypot(cbx - bbx, cby - bby)
            back_penalty = max(0.0, -cbx)
            side_penalty = 0.10 * abs(cby)
            score = (
                1.20 * align
                + 0.95 * min(min_dist, 2.0)
                - 0.75 * smooth
                - 0.30 * back_penalty
                - side_penalty
            )
            item = (score, label, cbx, cby, min_dist)
            evaluated.append(item)
            if min_dist >= effective_radius:
                safe.append(item)

        if safe:
            chosen = max(safe, key=lambda x: x[0])
            mode = 'unified_safe_' + chosen[1]
        else:
            # No candidate satisfies the conservative safety radius. Do not pick
            # the target direction. Choose the best escape among back/side options
            # by clearance, and only stop if even that is extremely close.
            escape = [x for x in evaluated if x[1] in ('side_left', 'side_right', 'back_left', 'back_right', 'back', 'stop')]
            chosen = max(escape, key=lambda x: (x[4], x[0]))
            if chosen[4] < emergency_radius:
                chosen = next(x for x in evaluated if x[1] == 'stop')
                mode = 'unified_emergency_stop'
            else:
                mode = 'unified_recovery_' + chosen[1]

        _, label, cbx, cby, min_dist = chosen
        vx, vy = self._body_to_world_xy(cbx, cby, heading)
        # Keep altitude almost fixed for this experiment. This is not a climb-over
        # solution; vertical motion is only ordinary altitude regulation.
        vz = max(-0.12, min(0.12, float(base_vz)))
        vx, vy, vz = self._limit_vector(vx, vy, vz, max_xy_speed)

        if now_t - self._last_failsafe_log_t >= 1.0:
            self._last_failsafe_log_t = now_t
            self.get_logger().info(
                f'unified selector: x500_{drone_id} mode={mode}, '
                f'local_planner={local_planner_label}, nearest={nearest if nearest is not None else 999.0:.2f}, '
                f'chosen={label}, clearance={min_dist:.2f}, safe={len(safe)}/{len(evaluated)}, '
                f'cmd_v=({vx:.2f},{vy:.2f},{vz:.2f})'
            )
        return vx, vy, vz, mode

    def _final_lidar_safety_filter(self, drone_id, vx, vy, vz):
        """Last-resort guard after unified selection.

        The unified selector should already avoid obstacles. This filter is kept
        conservative but compact, and it should not become another planner.
        """
        key = int(drone_id)
        now_t = time.time()
        cloud = self.latest_lidar_clouds.get(key)
        st = self.latest_states.get(key)
        if cloud is None or st is None:
            return vx, vy, vz, 'no_lidar'
        if now_t - float(cloud.get('stamp', 0.0)) > 2.0:
            return vx, vy, vz, 'stale_lidar'

        heading = self._finite_float(st.get('heading', 0.0))
        bvx, bvy = self._world_to_body_xy(vx, vy, heading)
        speed_xy = math.hypot(bvx, bvy)
        if speed_xy < 0.06:
            return vx, vy, vz, 'pass'
        dir_x = bvx / speed_xy
        dir_y = bvy / speed_xy

        nearest_forward = None
        nearest_any = None
        for idx, point in enumerate(point_cloud2.read_points(cloud['msg'], field_names=('x', 'y', 'z'), skip_nans=True)):
            if idx >= 2500:
                break
            px = self._finite_float(point[0])
            py = self._finite_float(point[1])
            pz = self._finite_float(point[2])
            horizontal = math.hypot(px, py)
            if horizontal < 0.55 or horizontal > 3.0 or pz < -1.6 or pz > 2.1:
                continue
            if nearest_any is None or horizontal < nearest_any:
                nearest_any = horizontal
            forward = px * dir_x + py * dir_y
            lateral = abs(-dir_y * px + dir_x * py)
            if 0.10 < forward < 1.60 and lateral < 0.70:
                if nearest_forward is None or forward < nearest_forward:
                    nearest_forward = forward

        if nearest_any is not None and nearest_any < 0.55:
            return 0.0, 0.0, 0.0, 'final_contact_stop'
        if nearest_forward is not None:
            # Final guard only removes the dangerous forward component; it does
            # not create a new side/back plan because unified selector does that.
            side_x = -dir_y
            side_y = dir_x
            side_mag = bvx * side_x + bvy * side_y
            nbvx = side_mag * side_x
            nbvy = side_mag * side_y
            nvx, nvy = self._body_to_world_xy(nbvx, nbvy, heading)
            nvx, nvy, _ = self._limit_vector(nvx, nvy, 0.0, 0.22)
            return nvx, nvy, 0.0, 'final_forward_removed'
        return vx, vy, vz, 'pass'

'''


UNIFIED_DIRECT_BLOCK = '''                local_planner_label = 'off'
                if direct_pose_mode:
                    # Keep the old local risk A* reference only as a diagnostic / weak
                    # target-ref modifier. Final velocity is chosen by the unified
                    # LiDAR selector below, not by primitive/TTC/final-safety stacking.
                    ref_x, ref_y, ref_z, local_planner_label = self._local_risk_astar_ref(
                        drone_id,
                        st,
                        ref_x,
                        ref_y,
                        ref_z,
                    )

                ex = ref_x - float(st['x'])
                ey = ref_y - float(st['y'])
                ez = ref_z - float(st['z'])
                vx = float(formation_ff[0]) + formation_kp * ex
                vy = float(formation_ff[1]) + formation_kp * ey
                vz = float(formation_ff[2]) + formation_kp * ez
                vx, vy, vz = self._limit_vector(vx, vy, vz, max_follower_speed)
                if direct_pose_mode:
                    vx, vy, vz, unified_label = self._unified_lidar_velocity_selector(
                        drone_id,
                        st,
                        vx,
                        vy,
                        vz,
                        ref_x,
                        ref_y,
                        ref_z,
                        max_follower_speed,
                        local_planner_label,
                    )
                    z_upper = ref_z - 0.25
                    z_lower = ref_z + 0.45
                    current_z = float(st['z'])
                    if current_z <= z_upper and vz < 0.0:
                        vz = 0.0
                    elif current_z >= z_lower and vz > 0.0:
                        vz = 0.0
                    self._publish_vel(drone_id, vx, vy, vz)
'''


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[1]
    path = repo / "src" / "xtd2_mission" / "xtd2_mission" / "multi_waypoint2.py"
    text = path.read_text(encoding="utf-8")
    original = text

    # Keep the local risk A* crop compact so it does not swallow narrow gaps.
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

    # Insert unified selector and replace final safety with a last-resort guard.
    text = replace_between(
        text,
        "    def _final_lidar_safety_filter(self, drone_id, vx, vy, vz):\n",
        "    def _estimate_virtual_ref_from_states",
        UNIFIED_AND_FINAL_METHODS_V5,
        "install unified lidar selector and compact final guard",
    )

    # Replace the direct-pose velocity arbitration block. This disables primitive /
    # lidar_ttc / safe_primitive_override as independent final velocity sources in
    # risk_astar direct-pose forest3 mode.
    text = replace_between(
        text,
        "                local_planner_label = 'off'\n                if direct_pose_mode:\n",
        "                else:\n                    # Position setpoints use the same proven control path as takeoff hold;",
        UNIFIED_DIRECT_BLOCK,
        "route direct-pose velocity through unified selector only",
    )

    # The collision monitor is an abstract geometric monitor, not Gazebo contact.
    # Make the runtime log remind us not to treat it as physical contact truth.
    text = replace_first_of(
        text,
        [
            "self.get_logger().info(f\"{stage_name}: 持续 {duration:.1f}s, 发布频率 {hz:.1f}Hz\")",
        ],
        "self.get_logger().info(f\"{stage_name}: 持续 {duration:.1f}s, 发布频率 {hz:.1f}Hz\")\n        self.get_logger().info('collision monitor is algorithmic clearance only; Gazebo contact may differ from monitor events')",
        "add collision-monitor-vs-gazebo-contact log note",
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
