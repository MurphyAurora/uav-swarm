#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend scoring and trajectory selection for the EGO-like planner."""

from typing import List, Optional, Sequence

from .common import CandidateEvaluation, CandidateTrajectory, PlannerConfig, PlannerState, SafetyReport, Vec3


class BackendSelector:
    """Score feasible trajectories after the hard safety checker runs."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def evaluate_all(
        self,
        candidates: Sequence[CandidateTrajectory],
        safety_reports: Sequence[SafetyReport],
        state: PlannerState,
        local_goal: Vec3,
    ) -> List[CandidateEvaluation]:
        evaluations = []
        for candidate, safety in zip(candidates, safety_reports):
            evaluations.append(self.evaluate(candidate, safety, state, local_goal))
        return evaluations

    def evaluate(
        self,
        candidate: CandidateTrajectory,
        safety: SafetyReport,
        state: PlannerState,
        local_goal: Vec3,
    ) -> CandidateEvaluation:
        goal_cost = candidate.final_position().distance_to(local_goal)
        clearance_cost = self._clearance_cost(safety.min_clearance)
        ttc_cost = self._ttc_cost(safety.min_ttc)
        smooth_cost = (candidate.velocity - state.velocity).norm()

        score = (
            self.config.goal_weight * goal_cost
            + self.config.clearance_weight * clearance_cost
            + self.config.ttc_weight * ttc_cost
            + self.config.smooth_weight * smooth_cost
        )
        if not safety.feasible:
            score += 10000.0
        elif not safety.safe:
            score += 1000.0

        return CandidateEvaluation(
            trajectory=candidate,
            safety=safety,
            score=float(score),
            costs={
                "goal": float(goal_cost),
                "clearance": float(clearance_cost),
                "ttc": float(ttc_cost),
                "smooth": float(smooth_cost),
            },
        )

    def select_best(self, evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
        safe = [item for item in evaluations if item.safety.safe]
        if safe:
            return min(safe, key=lambda item: item.score)
        feasible = [item for item in evaluations if item.safety.feasible]
        if feasible:
            # During an emergency margin violation, prefer the largest clearance,
            # then the lower score.  Fallback still has the final veto.
            return min(feasible, key=lambda item: (-item.safety.min_clearance, item.score))
        if evaluations:
            return min(evaluations, key=lambda item: (-item.safety.min_clearance, item.score))
        return None

    def _clearance_cost(self, min_clearance: float) -> float:
        if min_clearance <= 0.0:
            return 1.0
        if min_clearance >= self.config.risk_radius:
            return 0.0
        return (self.config.risk_radius - min_clearance) / max(self.config.risk_radius, 1.0e-6)

    def _ttc_cost(self, min_ttc: float) -> float:
        if min_ttc >= self.config.horizon:
            return 0.0
        return (self.config.horizon - max(0.0, min_ttc)) / max(self.config.horizon, 1.0e-6)
