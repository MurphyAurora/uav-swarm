#!/usr/bin/env python3
"""Front-end candidate trajectory generator."""

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
