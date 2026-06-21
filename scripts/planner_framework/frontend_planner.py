#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Front-end candidate trajectory generator.

This is the light-weight substitute for a full kinodynamic front-end.  It
produces short constant-velocity primitives that the backend and safety checker
can evaluate.  Existing A*/primitive code can later be moved behind this same
interface.
"""

from typing import Iterable, List, Optional, Tuple

from .common import CandidateTrajectory, PlannerConfig, PlannerState, TrajectoryPoint, Vec3


class FrontendPlanner:
    """Generate short-horizon local motion candidates."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def generate(self, state: PlannerState, local_goal: Vec3) -> List[CandidateTrajectory]:
        desired = self._desired_velocity_towards_goal(state, local_goal)
        primitives = [
            ("track_goal", desired),
            ("slow_goal", desired * 0.55),
            ("wait", Vec3()),
            ("back", self._body_relative(state, desired, forward_scale=-0.55, lateral_scale=0.0, vertical=0.0)),
            ("strafe_left", self._body_relative(state, desired, forward_scale=0.25, lateral_scale=1.0, vertical=0.0)),
            ("strafe_right", self._body_relative(state, desired, forward_scale=0.25, lateral_scale=-1.0, vertical=0.0)),
            ("climb", Vec3(0.0, 0.0, -self.config.vertical_speed)),
            ("descend", Vec3(0.0, 0.0, self.config.vertical_speed)),
            ("back_climb", self._body_relative(state, desired, forward_scale=-0.45, lateral_scale=0.0, vertical=-self.config.vertical_speed)),
            ("left_climb", self._body_relative(state, desired, forward_scale=0.1, lateral_scale=0.8, vertical=-self.config.vertical_speed)),
            ("right_climb", self._body_relative(state, desired, forward_scale=0.1, lateral_scale=-0.8, vertical=-self.config.vertical_speed)),
        ]
        return [self._rollout(name, velocity.limit_norm(self.config.max_speed), state.position) for name, velocity in primitives]

    def generate_from_velocity_set(
        self,
        state: PlannerState,
        velocity_set: Iterable[Tuple[str, Vec3]],
    ) -> List[CandidateTrajectory]:
        """Adapter for moving existing primitive-library outputs into the framework."""
        return [
            self._rollout(name, velocity.limit_norm(self.config.max_speed), state.position)
            for name, velocity in velocity_set
        ]

    def _desired_velocity_towards_goal(self, state: PlannerState, local_goal: Vec3) -> Vec3:
        to_goal = local_goal - state.position
        distance = to_goal.norm()
        if distance < 1.0e-6:
            return Vec3()
        speed = min(self.config.cruise_speed, distance / max(self.config.horizon, 1.0e-6))
        return to_goal.normalized() * speed

    def _body_relative(
        self,
        state: PlannerState,
        desired: Vec3,
        forward_scale: float,
        lateral_scale: float,
        vertical: float,
    ) -> Vec3:
        forward = Vec3(desired.x, desired.y, 0.0).normalized(Vec3(1.0, 0.0, 0.0))
        lateral = Vec3(-forward.y, forward.x, 0.0)
        vx = forward.x * self.config.cruise_speed * forward_scale + lateral.x * self.config.lateral_speed * lateral_scale
        vy = forward.y * self.config.cruise_speed * forward_scale + lateral.y * self.config.lateral_speed * lateral_scale
        return Vec3(vx, vy, vertical)

    def _rollout(self, name: str, velocity: Vec3, start: Vec3) -> CandidateTrajectory:
        points: List[TrajectoryPoint] = []
        steps = max(1, int(round(self.config.horizon / max(self.config.dt, 1.0e-6))))
        for idx in range(1, steps + 1):
            t = idx * self.config.dt
            points.append(TrajectoryPoint(t=t, position=start + velocity * t))
        return CandidateTrajectory(name=name, velocity=velocity, points=points)
