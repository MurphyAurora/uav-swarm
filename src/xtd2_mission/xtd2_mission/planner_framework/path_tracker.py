#!/usr/bin/env python3
"""Convert a local A* path into a short executable trajectory."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from .common import CandidateTrajectory, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .local_map_builder import BodyPoint


class PathTracker:
    """Pure-pursuit style path tracker used after frontend search."""

    def __init__(self, config: PlannerConfig):
        self.config = config
        self._last_body_velocity: BodyPoint = (0.0, 0.0)
        self._last_heading: Optional[float] = None

    def track(
        self,
        name: str,
        state: PlannerState,
        body_path: Sequence[BodyPoint],
        local_goal: Vec3,
        nearest_obstacle: float,
        path_clearance: float,
        metadata: Dict,
    ) -> CandidateTrajectory:
        body_path = self._prepare_body_path(body_path)
        if len(body_path) < 2:
            return CandidateTrajectory(name=name, velocity=Vec3(), points=[TrajectoryPoint(0.0, state.position)], metadata=metadata)

        frame_reset = self._maybe_reset_for_heading_change(state.heading)
        look_x, look_y = self._lookahead(body_path)
        if look_x < 0.30:
            look_x = 0.30
        look_x, look_y, lateral_limited = self._limit_lateral_lookahead(look_x, look_y, path_clearance)
        speed = self._speed_limit(nearest_obstacle, path_clearance)
        norm = max(math.hypot(look_x, look_y), 1.0e-6)
        bx = speed * look_x / norm
        by = speed * look_y / norm
        bx, by = self._smooth_body_velocity(bx, by)
        bx, by, progress_guard = self._guard_goal_progress(bx, by, state, local_goal, path_clearance)
        if bx < 0.02:
            bx = 0.0
            by = 0.0
        vx, vy = self._body_to_world(bx, by, state.heading)
        dz = local_goal.z - state.position.z
        vz = max(-0.05, min(0.05, 0.25 * dz)) if abs(dz) >= 0.25 else 0.0
        velocity = Vec3(vx, vy, vz).limit_norm(self.config.max_speed)
        points = self._rollout_points(state, body_path, speed, velocity.z)
        meta = dict(metadata)
        meta.update(
            {
                "tracker": "pure_pursuit_path_tracker",
                "lookahead_body": {"x": float(look_x), "y": float(look_y)},
                "tracker_speed": float(speed),
                "path_clearance": float(path_clearance),
                "nearest_obstacle": float(nearest_obstacle),
                "body_velocity": {"x": float(bx), "y": float(by)},
                "planning_frame_heading": float(state.heading),
                "frame_reset": bool(frame_reset),
                "progress_guard": progress_guard,
                "lateral_lookahead_limited": bool(lateral_limited),
            }
        )
        return CandidateTrajectory(name=name, velocity=velocity, points=points, metadata=meta)

    def reset(self) -> None:
        self._last_body_velocity = (0.0, 0.0)
        self._last_heading = None

    def _maybe_reset_for_heading_change(self, heading: float) -> bool:
        if self._last_heading is None:
            self._last_heading = heading
            return True
        delta = abs(math.atan2(math.sin(heading - self._last_heading), math.cos(heading - self._last_heading)))
        self._last_heading = heading
        if delta > 0.18:
            # Body-frame velocity memory belongs to the previous planning frame.
            # Reusing it after a goal-aligned frame rotation creates the small
            # loops seen in RViz, because old lateral velocity is smoothed into a
            # new coordinate system.
            self._last_body_velocity = (0.0, 0.0)
            return True
        return False

    def _speed_limit(self, nearest: float, clearance: float) -> float:
        hard = self._hard_required_clearance()
        safe = self._safe_required_clearance()
        speed = min(self.config.cruise_speed, self.config.max_speed, 0.42)
        if clearance < hard + 0.10:
            speed = min(speed, 0.10)
        elif clearance < safe:
            speed = min(speed, 0.18)
        elif nearest < 1.05:
            speed = min(speed, 0.14)
        elif nearest < 1.35:
            speed = min(speed, 0.24)
        else:
            speed = min(speed, 0.34)
        return max(0.0, speed)

    def _lookahead(self, body_path: Sequence[BodyPoint]) -> BodyPoint:
        last = body_path[0]
        acc = 0.0
        distance = max(0.90, min(self.config.astar_lookahead_dist, 1.35))
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            if acc + seg >= distance:
                r = (distance - acc) / max(seg, 1.0e-6)
                return last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1])
            acc += seg
            last = point
        return body_path[-1]

    @staticmethod
    def _prepare_body_path(body_path: Sequence[BodyPoint]) -> list[BodyPoint]:
        """Anchor path tracking at the current vehicle position.

        Committed A* paths are stored in world frame, so after the vehicle moves
        their first waypoint is often behind the vehicle.  Tracking that stale
        point is a common source of backwards commands and small loops.
        """
        cleaned: list[BodyPoint] = [(0.0, 0.0)]
        for x, y in body_path:
            if math.hypot(x, y) < 0.12:
                continue
            if x < -0.25:
                continue
            if cleaned and math.hypot(x - cleaned[-1][0], y - cleaned[-1][1]) < 0.18:
                continue
            cleaned.append((float(x), float(y)))
        if len(cleaned) >= 2:
            return cleaned
        return list(body_path)

    def _limit_lateral_lookahead(self, look_x: float, look_y: float, path_clearance: float) -> tuple[float, float, bool]:
        # Allow lateral motion for bypassing, but keep pure pursuit from chasing
        # a far side waypoint so aggressively that forward progress disappears.
        ratio = 1.65 if path_clearance < self._safe_required_clearance() else 1.25
        max_abs_y = max(0.45, ratio * max(look_x, 0.35))
        if abs(look_y) <= max_abs_y:
            return look_x, look_y, False
        return look_x, math.copysign(max_abs_y, look_y), True

    def _smooth_body_velocity(self, bx: float, by: float) -> BodyPoint:
        old_x, old_y = self._last_body_velocity
        alpha = max(0.55, min(1.0, self.config.output_alpha))
        sx = alpha * bx + (1.0 - alpha) * old_x
        sy = alpha * by + (1.0 - alpha) * old_y
        mag = math.hypot(sx, sy)
        if mag > self.config.max_speed:
            sx *= self.config.max_speed / mag
            sy *= self.config.max_speed / mag
        self._last_body_velocity = (sx, sy)
        return sx, sy

    def _guard_goal_progress(
        self,
        bx: float,
        by: float,
        state: PlannerState,
        local_goal: Vec3,
        path_clearance: float,
    ) -> Tuple[float, float, Dict[str, float | bool]]:
        gx, gy = self._world_to_body(local_goal.x - state.position.x, local_goal.y - state.position.y, state.heading)
        gnorm = math.hypot(gx, gy)
        speed = math.hypot(bx, by)
        if gnorm < 0.25 or speed < 1.0e-6:
            return bx, by, {"active": False, "progress": 0.0}
        ux = gx / gnorm
        uy = gy / gnorm
        progress = bx * ux + by * uy
        min_progress = min(0.10, max(0.04, 0.35 * speed))
        if progress >= min_progress or path_clearance < self._hard_required_clearance():
            return bx, by, {"active": False, "progress": float(progress)}

        lx = bx - progress * ux
        ly = by - progress * uy
        lateral = math.hypot(lx, ly)
        max_lateral = max(0.04, 0.75 * max(speed, min_progress))
        if lateral > max_lateral:
            lx *= max_lateral / lateral
            ly *= max_lateral / lateral
        nbx = min_progress * ux + lx
        nby = min_progress * uy + ly
        mag = math.hypot(nbx, nby)
        limit = max(speed, min_progress)
        if mag > limit:
            nbx *= limit / mag
            nby *= limit / mag
        self._last_body_velocity = (nbx, nby)
        return nbx, nby, {
            "active": True,
            "progress": float(progress),
            "min_progress": float(min_progress),
            "goal_body_x": float(gx),
            "goal_body_y": float(gy),
        }

    def _rollout_points(self, state: PlannerState, body_path: Sequence[BodyPoint], speed: float, vz: float):
        points = [TrajectoryPoint(t=0.0, position=state.position)]
        steps = max(1, int(round(self.config.horizon / max(self.config.dt, 1.0e-6))))
        total_len = self._body_path_length(body_path)
        for idx in range(1, steps + 1):
            t = idx * self.config.dt
            bx, by = self._point_at_distance(body_path, min(total_len, speed * t))
            wx, wy = self._body_to_world(bx, by, state.heading)
            points.append(TrajectoryPoint(t=t, position=Vec3(state.position.x + wx, state.position.y + wy, state.position.z + vz * t)))
        return points

    @staticmethod
    def _body_path_length(body_path: Sequence[BodyPoint]) -> float:
        total = 0.0
        last = body_path[0]
        for point in body_path[1:]:
            total += math.hypot(point[0] - last[0], point[1] - last[1])
            last = point
        return total

    @staticmethod
    def _point_at_distance(body_path: Sequence[BodyPoint], distance: float) -> BodyPoint:
        if not body_path:
            return 0.0, 0.0
        if distance <= 0.0:
            return body_path[0]
        remaining = distance
        last = body_path[0]
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            if seg <= 1.0e-6:
                last = point
                continue
            if remaining <= seg:
                r = remaining / seg
                return last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1])
            remaining -= seg
            last = point
        return body_path[-1]

    @staticmethod
    def _body_to_world(bx: float, by: float, heading: float) -> Tuple[float, float]:
        c = math.cos(heading)
        s = math.sin(heading)
        return c * bx - s * by, s * bx + c * by

    @staticmethod
    def _world_to_body(wx: float, wy: float, heading: float) -> Tuple[float, float]:
        c = math.cos(heading)
        s = math.sin(heading)
        return c * wx + s * wy, -s * wx + c * wy

    def _hard_required_clearance(self) -> float:
        return max(0.72, self.config.drone_radius + 0.18 + 0.18)

    def _safe_required_clearance(self) -> float:
        return max(1.02, self._hard_required_clearance() + 0.28)
