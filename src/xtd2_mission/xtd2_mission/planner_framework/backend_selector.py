#!/usr/bin/env python3
"""Backend scoring and selection."""

from typing import List, Optional, Sequence

from .common import CandidateEvaluation, CandidateTrajectory, PlannerConfig, PlannerState, SafetyReport, Vec3


class BackendSelector:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def evaluate_all(self, candidates: Sequence[CandidateTrajectory], safety_reports: Sequence[SafetyReport], state: PlannerState, local_goal: Vec3) -> List[CandidateEvaluation]:
        return [self.evaluate(candidate, safety, state, local_goal) for candidate, safety in zip(candidates, safety_reports)]

    def evaluate(self, candidate: CandidateTrajectory, safety: SafetyReport, state: PlannerState, local_goal: Vec3) -> CandidateEvaluation:
        goal_cost = candidate.final_position().distance_to(local_goal)
        clearance_cost = self._clearance_cost(safety.min_clearance)
        ttc_cost = self._ttc_cost(safety.min_ttc)
        smooth_cost = (candidate.velocity - state.velocity).norm()
        progress, lateral = self._progress_terms(candidate.velocity, state, local_goal)
        progress_cost = max(0.0, 1.0 - progress)
        reverse_cost = max(0.0, -progress)
        lateral_cost = abs(lateral)
        score = (
            self.config.goal_weight * goal_cost
            + self.config.clearance_weight * clearance_cost
            + self.config.ttc_weight * ttc_cost
            + self.config.smooth_weight * smooth_cost
            + self.config.progress_weight * progress_cost
            + self.config.reverse_penalty * reverse_cost
            + self.config.lateral_penalty * lateral_cost
        )
        if not safety.feasible:
            score += 10000.0
        elif not safety.safe:
            score += 1000.0
        return CandidateEvaluation(
            candidate,
            safety,
            float(score),
            {
                "goal": goal_cost,
                "clearance": clearance_cost,
                "ttc": ttc_cost,
                "smooth": smooth_cost,
                "progress": progress_cost,
                "reverse": reverse_cost,
                "lateral": lateral_cost,
            },
        )

    def select_best(self, evaluations: Sequence[CandidateEvaluation], state: Optional[PlannerState] = None, local_goal: Optional[Vec3] = None) -> Optional[CandidateEvaluation]:
        prefer_motion = False
        if state is not None and local_goal is not None:
            prefer_motion = state.position.distance_to(local_goal) > self.config.local_goal_reached_radius
        safe = [item for item in evaluations if item.safety.safe]
        if safe:
            moving_safe = [item for item in safe if item.trajectory.velocity.norm() > 0.05]
            if prefer_motion and moving_safe:
                local_astar_safe = self._local_astar_items(moving_safe)
                if local_astar_safe:
                    return min(local_astar_safe, key=lambda item: item.score)
                goal_tracking = [
                    item for item in moving_safe
                    if item.trajectory.name in ("track_goal", "slow_goal")
                ]
                if goal_tracking:
                    return min(goal_tracking, key=lambda item: (item.costs.get("goal", 999.0), item.score))
                progress_safe = [
                    item for item in moving_safe
                    if item.costs.get("reverse", 0.0) <= 1.0e-6
                ]
                if progress_safe:
                    return min(progress_safe, key=lambda item: item.score)
                return min(moving_safe, key=lambda item: (item.costs.get("reverse", 999.0), item.score))
            return min(safe, key=lambda item: item.score)
        feasible = [item for item in evaluations if item.safety.feasible]
        if feasible:
            moving_feasible = [item for item in feasible if item.trajectory.velocity.norm() > 0.05]
            if prefer_motion and moving_feasible:
                local_astar_feasible = self._local_astar_items(moving_feasible)
                if local_astar_feasible:
                    return min(local_astar_feasible, key=lambda item: (-item.safety.min_clearance, item.score))
                progress_feasible = [
                    item for item in moving_feasible
                    if item.costs.get("reverse", 0.0) <= 1.0e-6
                ]
                if progress_feasible:
                    return min(
                        progress_feasible,
                        key=lambda item: (
                            item.costs.get("progress", 999.0),
                            item.costs.get("lateral", 999.0),
                            -item.safety.min_clearance,
                            item.score,
                        ),
                    )
                return min(moving_feasible, key=lambda item: (item.costs.get("reverse", 999.0), -item.safety.min_clearance, item.score))
            return min(feasible, key=lambda item: (-item.safety.min_clearance, item.score))
        return min(evaluations, key=lambda item: (-item.safety.min_clearance, item.score)) if evaluations else None

    @staticmethod
    def _local_astar_items(evaluations: Sequence[CandidateEvaluation]) -> List[CandidateEvaluation]:
        return [item for item in evaluations if item.trajectory.name.startswith("local_astar:")]

    def _progress_terms(self, velocity: Vec3, state: PlannerState, local_goal: Vec3) -> tuple:
        to_goal = local_goal - state.position
        direction = Vec3(to_goal.x, to_goal.y, 0.0).normalized(Vec3(1.0, 0.0, 0.0))
        speed_xy = max((velocity.x * velocity.x + velocity.y * velocity.y) ** 0.5, 1.0e-6)
        progress = (velocity.x * direction.x + velocity.y * direction.y) / speed_xy
        lateral = (-direction.y * velocity.x + direction.x * velocity.y) / max(self.config.max_speed, 1.0e-6)
        return float(progress), float(lateral)

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
