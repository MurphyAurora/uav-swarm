#!/usr/bin/env python3
"""Fallback commands used when no candidate is safely executable."""

from typing import Optional, Sequence

from .common import CandidateEvaluation, PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3


class FallbackManager:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

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
            return None
        nearest = None
        nearest_clearance = 999.0
        for obs in perception.obstacles:
            if obs.source != "lidar_near_field":
                continue
            clearance = state.position.distance_to(obs.position) - float(obs.radius) - self.config.drone_radius
            if clearance < nearest_clearance:
                nearest = obs
                nearest_clearance = clearance
        if nearest is None or nearest_clearance > self.config.static_emergency_clearance:
            return None
        away = Vec3(
            state.position.x - nearest.position.x,
            state.position.y - nearest.position.y,
            0.0,
        ).normalized(Vec3(-1.0, 0.0, 0.0))
        speed = min(self.config.max_speed, max(self.config.lateral_speed, 0.45 * self.config.max_speed))
        return PlannerCommand(
            away * speed,
            "fallback_escape",
            "nearest_lidar_escape",
            f"near_field_lidar_clearance={nearest_clearance:.2f}",
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
        escape = [item for item in evaluations if item.trajectory.name in escape_names]
        if not escape:
            return None
        moving_escape = [item for item in escape if item.trajectory.velocity.norm() > 0.05]
        if moving_escape:
            escape = moving_escape
        return max(escape, key=lambda item: (item.safety.min_clearance, -item.score))
