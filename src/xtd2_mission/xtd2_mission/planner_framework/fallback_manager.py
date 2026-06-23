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
        if not best.safety.feasible:
            return True
        return not any(item.safety.safe for item in evaluations)

    def command(self, state: PlannerState, perception: PerceptionData, evaluations: Sequence[CandidateEvaluation]) -> PlannerCommand:
        """Conservative fallback.

        During single-UAV static-obstacle bring-up, fallback must not create a new
        motion direction.  Previous fallback escape and the old shield both could
        command away/reverse velocity from an arbitrary nearest LiDAR point.  That
        made the vehicle hit the opposite wall.  Fallback now only stops; path
        generation and cautious motion are handled by planner_core/local_astar.
        """
        best = self._best_escape(evaluations)
        if best is not None:
            return PlannerCommand(Vec3(), "fallback_hover", "hover", f"fallback_stop:{best.safety.reason}")
        if perception.near_field_danger:
            return PlannerCommand(Vec3(), "fallback_hover", "hover", "near_field_lidar_stop")
        return PlannerCommand(Vec3(), "fallback_hover", "hover", "no_safe_candidate")

    @staticmethod
    def _best_escape(evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
        if not evaluations:
            return None
        return max(
            evaluations,
            key=lambda item: (
                item.safety.safe,
                item.safety.feasible,
                item.safety.min_clearance,
                -item.score,
            ),
        )
