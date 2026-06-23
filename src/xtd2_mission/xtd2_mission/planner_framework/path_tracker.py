#!/usr/bin/env python3
"""Convert a local A* path into a short executable trajectory."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from .common import CandidateTrajectory, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .local_map_builder import BodyPoint


class PathTracker:
    """Pure-pursuit style path tracker used after frontend search.

    The frontend owns path search.  This class owns path-to-trajectory conversion:
    it limits speed from clearance/nearest obstacle, prevents reverse commands,
    and generates horizon points along the path for the SafetyChecker.
    """

    def __init__(self, config: PlannerConfig):
        self.config = config
        self._last_body_velocity: BodyPoint = (0.0, 0.0)

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
        if len(body_path) < 2:
            return CandidateTrajectory(name=name, velocity=Vec3(), points=[TrajectoryPoint(0.0, state.position)], metadata=metadata)

        look_x, look_y = self._lookahead(body_path)
        # Do not allow path tracking to command reverse motion during normal
        # obstacle bypass.  Braking/replanning handles impossible paths.
        if look_x < 0.30:
            look_x = 0.30
        speed = self._speed_limit(nearest_obstacle, path_clearance)
        norm = max(math.hypot(look_x, look_y), 1.0e-6)
        bx = speed * look_x / norm
        by = speed * look_y / norm
        bx, by = self._smooth_body_velocity(bx, by)
        # A final local check: smoothing must not turn the command backwards.
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
            }
        )
        return CandidateTrajectory(name=name, velocity=velocity, points=points, metadata=meta)

    def reset(self) -> None:
        self._last_body_velocity = (0.0, 0.0)

    def _speed_limit(self, nearest: float, clearance: float) -> float:
        speed = min(self.config.cruise_speed, self.config.max_speed, 0.38)
        if nearest < 2.2 or clearance < self._safe_required_clearance():
            speed = min(speed, 0.16)
        if nearest < 1.55 or clearance < self._hard_required_clearance() + 0.15:
            speed = min(speed, 0.10)
        return max(0.0, speed)

    def _lookahead(self, body_path: Sequence[BodyPoint]) -> BodyPoint:
        last = body_path[0]
        acc = 0.0
        distance = max(0.75, min(self.config.astar_lookahead_dist, 1.10))
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            if acc + seg >= distance:
                r = (distance - acc) / max(seg, 1.0e-6)
                return last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1])
            acc += seg
            last = point
        return body_path[-1]

    def _smooth_body_velocity(self, bx: float, by: float) -> BodyPoint:
        old_x, old_y = self._last_body_velocity
        alpha = max(0.25, min(1.0, self.config.output_alpha))
        sx = alpha * bx + (1.0 - alpha) * old_x
        sy = alpha * by + (1.0 - alpha) * old_y
        mag = math.hypot(sx, sy)
        if mag > self.config.max_speed:
            sx *= self.config.max_speed / mag
            sy *= self.config.max_speed / mag
        self._last_body_velocity = (sx, sy)
        return sx, sy

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

    def _hard_required_clearance(self) -> float:
        return max(1.22, self.config.drone_radius + self.config.obstacle_margin + self.config.hard_clearance + 0.18)

    def _safe_required_clearance(self) -> float:
        return max(1.55, self.config.drone_radius + self.config.obstacle_margin + max(0.60, self.config.emergency_clearance) + 0.18)
