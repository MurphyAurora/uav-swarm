#!/usr/bin/env python3
"""Fallback commands used when no candidate is safely executable."""

import time
import math
from typing import Optional, Sequence

from .common import CandidateEvaluation, PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3


class FallbackManager:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._latched_escape_until = 0.0
        self._latched_escape_velocity = Vec3()
        self._latched_escape_reason = ""

    def should_fallback(self, best: Optional[CandidateEvaluation], evaluations: Sequence[CandidateEvaluation], perception: PerceptionData) -> bool:
        if best is None:
            return True
        if perception.near_field_danger and best.safety.min_static_clearance < self.config.emergency_clearance:
            return True
        if not best.safety.feasible:
            return True
        return not any(item.safety.safe for item in evaluations)

    def command(self, state: PlannerState, perception: PerceptionData, evaluations: Sequence[CandidateEvaluation]) -> PlannerCommand:
        near_escape = self._nearest_obstacle_escape(state, perception)
        if near_escape is not None:
            return near_escape
        escape_eval = self._best_escape(evaluations)
        if escape_eval is not None:
            clearance = min(escape_eval.safety.min_clearance, escape_eval.safety.min_static_clearance)
            soft_escape_ok = clearance >= -0.10 and escape_eval.trajectory.velocity.norm() > 0.05
            no_feasible_motion = not any(item.safety.feasible for item in evaluations)
            if escape_eval.safety.feasible or soft_escape_ok or no_feasible_motion:
                return PlannerCommand(escape_eval.trajectory.velocity, "fallback_escape", escape_eval.trajectory.name, escape_eval.safety.reason)
        if perception.near_field_danger:
            return PlannerCommand(Vec3(0.0, 0.0, -self.config.vertical_speed), "fallback_climb", "near_field_climb", "near_field_lidar_danger")
        return PlannerCommand(Vec3(), "fallback_hover", "hover", "no_safe_candidate")

    def _nearest_obstacle_escape(self, state: PlannerState, perception: PerceptionData) -> Optional[PlannerCommand]:
        if not perception.near_field_danger:
            self._latched_escape_until = 0.0
            return None
        now = time.time()
        if now < self._latched_escape_until and self._latched_escape_velocity.norm() > 0.05:
            return PlannerCommand(
                self._latched_escape_velocity,
                "fallback_escape",
                "latched_lidar_escape",
                self._latched_escape_reason,
            )

        close_count = 0
        pressure = Vec3()
        nearest_clearance = 999.0
        c = math.cos(state.heading)
        s = math.sin(state.heading)
        for obs in perception.obstacles:
            if obs.source != "lidar_near_field":
                continue
            clearance = state.position.distance_to(obs.position) - float(obs.radius) - self.config.drone_radius
            rel_x = obs.position.x - state.position.x
            rel_y = obs.position.y - state.position.y
            body_x = c * rel_x + s * rel_y
            body_y = -s * rel_x + c * rel_y
            hard_contact = clearance <= self.config.hard_clearance + 0.10
            front_threat = body_x >= -0.15 and abs(body_y) <= 1.10 and clearance <= self.config.emergency_clearance
            if not hard_contact and not front_threat:
                continue
            away = Vec3(
                state.position.x - obs.position.x,
                state.position.y - obs.position.y,
                0.0,
            )
            distance = max(away.norm(), 1.0e-3)
            weight = 1.0 / max(0.12, clearance + 0.35)
            pressure = pressure + away * (weight / distance)
            close_count += 1
            if clearance < nearest_clearance:
                nearest_clearance = clearance
        if close_count <= 0:
            return None
        fallback_dir = self._latched_escape_velocity.normalized(Vec3(-1.0, 0.0, 0.0))
        away = pressure.normalized(fallback_dir)
        if self._latched_escape_velocity.norm() > 0.05:
            prev = self._latched_escape_velocity.normalized()
            if away.x * prev.x + away.y * prev.y < 0.25:
                away = prev
        speed = min(self.config.max_speed, max(self.config.lateral_speed, 0.45 * self.config.max_speed))
        velocity = away * speed
        reason = f"near_field_lidar_clearance={nearest_clearance:.2f}, close_points={close_count}"
        self._latched_escape_velocity = velocity
        self._latched_escape_reason = reason
        self._latched_escape_until = now + 1.0
        return PlannerCommand(
            velocity,
            "fallback_escape",
            "latched_lidar_escape",
            reason,
        )

    def _best_escape(self, evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
        escape_names = {
            "wait",
            "back",
            "strafe_left",
            "strafe_right",
            "wide_left",
            "wide_right",
            "hard_left",
            "hard_right",
            "climb",
            "descend",
            "back_climb",
            "left_climb",
            "right_climb",
        }
        escape = [
            item for item in evaluations
            if item.trajectory.name in escape_names or item.trajectory.name.startswith("local_astar:")
        ]
        if not escape:
            return None
        moving_escape = [item for item in escape if item.trajectory.velocity.norm() > 0.05]
        if moving_escape:
            escape = moving_escape
        astar_escape = [item for item in escape if item.trajectory.name.startswith("local_astar:")]
        if astar_escape:
            return max(astar_escape, key=lambda item: (item.safety.feasible, item.safety.min_clearance, -item.score))
        return max(escape, key=lambda item: (item.safety.min_clearance, -item.score))
