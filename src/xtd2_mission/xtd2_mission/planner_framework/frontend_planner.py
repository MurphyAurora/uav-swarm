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
            escape_candidates = self._hard_zone_escape_candidates(state, local_goal, perception)
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

    def _hard_zone_escape_candidates(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData) -> List[CandidateTrajectory]:
        nearest_obs = None
        nearest_clearance = 999.0
        nearest_rel = None
        for obs in perception.obstacles:
            obs_pos = obs.position_at(0.0)
            rel_from_obs = state.position - obs_pos
            dist_xy = math.hypot(rel_from_obs.x, rel_from_obs.y)
            if dist_xy <= 1.0e-6:
                continue
            clearance = dist_xy - self._execution_radius(obs)
            if clearance < nearest_clearance:
                nearest_clearance = clearance
                nearest_obs = obs
                nearest_rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)

        if nearest_obs is None or nearest_rel is None:
            return []

        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        goal_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        lateral_left = Vec3(-goal_dir.y, goal_dir.x, 0.0)
        lateral_right = Vec3(goal_dir.y, -goal_dir.x, 0.0)

        obs_forward = nearest_rel.x * goal_dir.x + nearest_rel.y * goal_dir.y
        obs_lateral = nearest_rel.x * lateral_left.x + nearest_rel.y * lateral_left.y
        front_obstacle = 0.2 <= obs_forward <= 4.5 and abs(obs_lateral) <= 1.8
        trigger = max(self.config.static_emergency_clearance, 0.85)
        if nearest_clearance > trigger and not front_obstacle:
            return []

        # Pick the side with more immediate clearance.  For a centered obstacle,
        # try both sides, but make both candidates mostly lateral.  Do not allow a
        # forward component here; the old escape candidates still had vx>0 and
        # simply drove into the pillar while being labeled as escaping.
        preferred = lateral_left if obs_lateral <= 0.0 else lateral_right
        secondary = lateral_right if obs_lateral <= 0.0 else lateral_left
        back = goal_dir * -1.0
        speed = min(0.28, max(0.20, self.config.max_speed * 0.38))
        back_speed = min(0.18, max(0.10, self.config.max_speed * 0.22))

        directions = [
            ("local_escape:lateral_preferred", preferred, speed),
            ("local_escape:lateral_secondary", secondary, speed * 0.85),
            ("local_escape:back", back, back_speed),
            ("local_escape:back_preferred", self._norm_vec(back * 0.45 + preferred), speed * 0.75),
        ]
        return [
            self._rollout(name, direction * cmd_speed, state.position)
            for name, direction, cmd_speed in directions
        ]

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

    @staticmethod
    def _norm_vec(v: Vec3) -> Vec3:
        return Vec3(v.x, v.y, 0.0).normalized(Vec3(0.0, 1.0, 0.0))
