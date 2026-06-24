#!/usr/bin/env python3
"""Front-end candidate trajectory generator."""

import math
from typing import Iterable, List, Optional, Tuple

from .common import CandidateTrajectory, PerceptionData, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .local_astar_passable import LocalAStarPlanner


class FrontendPlanner:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.local_astar = LocalAStarPlanner(self.config)
        self.diagnostics = {}

    def generate(self, state: PlannerState, local_goal: Vec3, perception: Optional[PerceptionData] = None) -> List[CandidateTrajectory]:
        candidates: List[CandidateTrajectory] = []
        mode = str(self.config.frontend_mode or "local_astar").lower()
        if perception is not None and mode in ("local_astar", "astar", "hybrid_astar", "hybrid"):
            astar_candidate = self.local_astar.plan_candidate(state, local_goal, perception.obstacles)
            self.diagnostics = {"mode": mode, "local_astar": dict(self.local_astar.diagnostics)}
            if astar_candidate is not None:
                candidates.append(astar_candidate)
            escape_candidates = self._hard_zone_escape_candidates(state, perception)
            if escape_candidates:
                candidates.extend(escape_candidates)
                self.diagnostics["escape_candidates"] = len(escape_candidates)
        else:
            self.diagnostics = {"mode": mode}
            candidates.append(self._rollout("track_goal", self._desired_velocity_towards_goal(state, local_goal), state.position))
        return candidates

    def generate_from_velocity_set(self, state: PlannerState, velocity_set: Iterable[Tuple[str, Vec3]]) -> List[CandidateTrajectory]:
        return [self._rollout(name, velocity.limit_norm(self.config.max_speed), state.position) for name, velocity in velocity_set]

    def _desired_velocity_towards_goal(self, state: PlannerState, local_goal: Vec3) -> Vec3:
        to_goal = local_goal - state.position
        distance = to_goal.norm()
        if distance < 1.0e-6:
            return Vec3()
        speed = min(self.config.cruise_speed, distance / max(self.config.horizon, 1.0e-6))
        return to_goal.normalized() * speed

    def _rollout(self, name: str, velocity: Vec3, start: Vec3) -> CandidateTrajectory:
        velocity = velocity.limit_norm(self.config.max_speed)
        points = [TrajectoryPoint(t=0.0, position=start)]
        steps = max(1, int(round(self.config.horizon / max(self.config.dt, 1.0e-6))))
        for idx in range(1, steps + 1):
            t = idx * self.config.dt
            points.append(TrajectoryPoint(t=t, position=start + velocity * t))
        return CandidateTrajectory(name=name, velocity=velocity, points=points)

    def _hard_zone_escape_candidates(self, state: PlannerState, perception: PerceptionData) -> List[CandidateTrajectory]:
        nearest = None
        nearest_clearance = 999.0
        for obs in perception.obstacles:
            rel = state.position - obs.position_at(0.0)
            dist_xy = math.hypot(rel.x, rel.y)
            if dist_xy <= 1.0e-6:
                continue
            clearance = dist_xy - self._execution_radius(obs)
            if clearance < nearest_clearance:
                nearest_clearance = clearance
                nearest = (rel.x / dist_xy, rel.y / dist_xy)

        if nearest is None:
            return []
        trigger = max(min(self.config.static_hard_clearance, 0.35), 0.28) + 0.08
        if nearest_clearance > trigger:
            return []

        ax, ay = nearest
        px, py = -ay, ax
        speed = min(0.18, max(0.12, self.config.max_speed * 0.25))
        directions = [
            ("local_escape:away", (ax, ay)),
            ("local_escape:away_left", self._norm2(ax + 0.45 * px, ay + 0.45 * py)),
            ("local_escape:away_right", self._norm2(ax - 0.45 * px, ay - 0.45 * py)),
        ]
        return [self._rollout(name, Vec3(dx * speed, dy * speed, 0.0), state.position) for name, (dx, dy) in directions]

    def _execution_radius(self, obs) -> float:
        if obs.source == "lidar_near_field":
            return self.config.drone_radius + float(obs.radius) + 0.20
        if obs.source == "static":
            return self.config.drone_radius + float(obs.radius) + min(self.config.obstacle_margin, 0.10)
        return self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin

    @staticmethod
    def _norm2(x: float, y: float) -> Tuple[float, float]:
        mag = math.hypot(x, y)
        if mag <= 1.0e-6:
            return 1.0, 0.0
        return x / mag, y / mag
