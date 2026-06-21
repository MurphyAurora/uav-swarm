#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fallback commands used when no candidate trajectory is safely executable."""

from typing import Optional, Sequence

from .common import CandidateEvaluation, PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3


class FallbackManager:
    """Hard fallback layer separated from normal scoring.

    This prevents the planner from continuing to chase the goal when all local
    candidates violate hard or emergency safety margins.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def should_fallback(
        self,
        best: Optional[CandidateEvaluation],
        evaluations: Sequence[CandidateEvaluation],
        perception: PerceptionData,
    ) -> bool:
        if best is None:
            return True
        if perception.near_field_danger and best.safety.min_static_clearance < self.config.emergency_clearance:
            return True
        if not best.safety.feasible:
            return True
        # If no trajectory reaches the safe emergency margin, use fallback instead
        # of selecting a merely less-bad forward action.
        return not any(item.safety.safe for item in evaluations)

    def command(self, state: PlannerState, perception: PerceptionData, evaluations: Sequence[CandidateEvaluation]) -> PlannerCommand:
        escape_eval = self._best_escape(evaluations)
        if escape_eval is not None and escape_eval.safety.feasible:
            return PlannerCommand(
                velocity=escape_eval.trajectory.velocity,
                mode="fallback_escape",
                source_trajectory=escape_eval.trajectory.name,
                reason=escape_eval.safety.reason,
            )

        if perception.near_field_danger:
            return PlannerCommand(
                velocity=Vec3(0.0, 0.0, -self.config.vertical_speed),
                mode="fallback_climb",
                source_trajectory="near_field_climb",
                reason="near_field_lidar_danger",
            )

        return PlannerCommand(
            velocity=Vec3(),
            mode="fallback_hover",
            source_trajectory="hover",
            reason="no_safe_candidate",
        )

    def _best_escape(self, evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
        escape_names = {
            "wait",
            "back",
            "strafe_left",
            "strafe_right",
            "climb",
            "descend",
            "back_climb",
            "left_climb",
            "right_climb",
        }
        escape = [item for item in evaluations if item.trajectory.name in escape_names]
        if not escape:
            return None
        return max(escape, key=lambda item: (item.safety.min_clearance, -item.score))
