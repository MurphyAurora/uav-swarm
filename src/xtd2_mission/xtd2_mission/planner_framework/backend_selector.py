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
        if self._is_sampling_mpc_candidate(candidate):
            return self._evaluate_sampling_mpc(candidate, safety, state, local_goal)

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
        if reverse_cost > 0.05:
            score += 5000.0
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

    def _evaluate_sampling_mpc(self, candidate: CandidateTrajectory, safety: SafetyReport, state: PlannerState, local_goal: Vec3) -> CandidateEvaluation:
        metadata = candidate.metadata or {}
        sample_cost = float(metadata.get("sample_cost", 999999.0) or 999999.0)
        progress, lateral = self._progress_terms(candidate.velocity, state, local_goal)
        reverse_cost = max(0.0, -progress)
        moving = self._is_moving_candidate_obj(candidate)
        stop_like = self._is_stop_candidate_obj(candidate)

        # Sampling MPC already evaluates full-horizon objective terms.  The backend
        # should only add hard-safety and selection-shape penalties, not overwrite
        # the MPC objective with the legacy one-step candidate score.
        score = sample_cost
        if not safety.feasible:
            score += 10000.0
        elif not safety.safe:
            # Keep feasible moving trajectories alive in dynamic-avoid/recovery.
            # A large legacy +1000 penalty makes safe stop dominate every time.
            score += 250.0 if moving and not stop_like else 800.0
        if stop_like:
            # Stop is a fallback, not a normal MPC solution when the goal is far.
            goal_dist = state.position.distance_to(local_goal)
            if goal_dist > self.config.local_goal_reached_radius:
                score += 750.0
        if reverse_cost > 0.05:
            score += 1200.0

        costs = dict(metadata.get("sample_breakdown", {}) or {})
        costs.update(
            {
                "sample_cost": sample_cost,
                "backend_score": float(score),
                "progress": max(0.0, 1.0 - progress),
                "reverse": reverse_cost,
                "lateral": abs(lateral),
                "moving": 0.0 if moving else 1.0,
                "stop_like": 1.0 if stop_like else 0.0,
            }
        )
        return CandidateEvaluation(candidate, safety, float(score), costs)

    def select_best(self, evaluations: Sequence[CandidateEvaluation], state: Optional[PlannerState] = None, local_goal: Optional[Vec3] = None) -> Optional[CandidateEvaluation]:
        if any(self._is_sampling_mpc_evaluation(item) for item in evaluations):
            selected = self._select_sampling_mpc_best(evaluations, state, local_goal)
            if selected is not None:
                return selected

        prefer_motion = False
        if state is not None and local_goal is not None:
            prefer_motion = state.position.distance_to(local_goal) > self.config.local_goal_reached_radius
        normal = [item for item in evaluations if item.costs.get("reverse", 0.0) <= 0.05]
        pool = normal if normal else list(evaluations)
        safe = [item for item in pool if item.safety.safe]
        if not safe and pool is not evaluations:
            safe = [item for item in evaluations if item.safety.safe]
        if safe:
            moving_safe = [item for item in safe if item.trajectory.velocity.norm() > 0.05]
            if prefer_motion and moving_safe:
                local_astar_safe = self._local_astar_items(moving_safe)
                if local_astar_safe:
                    return min(
                        local_astar_safe,
                        key=lambda item: (
                            item.costs.get("goal", 999.0),
                            item.costs.get("progress", 999.0),
                            item.costs.get("lateral", 999.0),
                            item.score,
                        ),
                    )
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
                    return min(
                        progress_safe,
                        key=lambda item: (
                            item.costs.get("goal", 999.0),
                            item.costs.get("progress", 999.0),
                            item.costs.get("lateral", 999.0),
                            item.score,
                        ),
                    )
                return min(moving_safe, key=lambda item: (item.costs.get("reverse", 999.0), item.costs.get("goal", 999.0), item.score))
            return min(safe, key=lambda item: item.score)
        feasible = [item for item in pool if item.safety.feasible]
        if not feasible and pool is not evaluations:
            feasible = [item for item in evaluations if item.safety.feasible]
        if feasible:
            moving_feasible = [item for item in feasible if item.trajectory.velocity.norm() > 0.05]
            if prefer_motion and moving_feasible:
                local_astar_feasible = self._local_astar_items(moving_feasible)
                if local_astar_feasible:
                    return min(
                        local_astar_feasible,
                        key=lambda item: (
                            item.costs.get("goal", 999.0),
                            item.costs.get("progress", 999.0),
                            item.costs.get("lateral", 999.0),
                            -item.safety.min_clearance,
                            item.score,
                        ),
                    )
                progress_feasible = [
                    item for item in moving_feasible
                    if item.costs.get("reverse", 0.0) <= 1.0e-6
                ]
                if progress_feasible:
                    return min(
                        progress_feasible,
                        key=lambda item: (
                            item.costs.get("goal", 999.0),
                            item.costs.get("progress", 999.0),
                            item.costs.get("lateral", 999.0),
                            -item.safety.min_clearance,
                            item.score,
                        ),
                    )
                return min(moving_feasible, key=lambda item: (item.costs.get("reverse", 999.0), item.costs.get("goal", 999.0), -item.safety.min_clearance, item.score))
            return min(feasible, key=lambda item: (-item.safety.min_clearance, item.score))
        return min(pool, key=lambda item: (-item.safety.min_clearance, item.score)) if pool else None

    def _select_sampling_mpc_best(self, evaluations: Sequence[CandidateEvaluation], state: Optional[PlannerState], local_goal: Optional[Vec3]) -> Optional[CandidateEvaluation]:
        mpc_items = [item for item in evaluations if self._is_sampling_mpc_evaluation(item)]
        if not mpc_items:
            return None
        prefer_motion = True
        if state is not None and local_goal is not None:
            prefer_motion = state.position.distance_to(local_goal) > self.config.local_goal_reached_radius

        moving = [item for item in mpc_items if self._is_moving_evaluation(item) and not self._is_stop_evaluation(item)]
        safe_moving = [item for item in moving if item.safety.safe]
        feasible_moving = [item for item in moving if item.safety.feasible]
        stop_like = [item for item in mpc_items if self._is_stop_evaluation(item)]
        safe_stop = [item for item in stop_like if item.safety.safe]
        feasible_stop = [item for item in stop_like if item.safety.feasible]

        if prefer_motion:
            if safe_moving:
                return min(safe_moving, key=lambda item: item.score)
            if feasible_moving:
                return min(feasible_moving, key=lambda item: item.score)
            if safe_stop:
                return min(safe_stop, key=lambda item: item.score)
            if feasible_stop:
                return min(feasible_stop, key=lambda item: item.score)
        safe = [item for item in mpc_items if item.safety.safe]
        if safe:
            return min(safe, key=lambda item: item.score)
        feasible = [item for item in mpc_items if item.safety.feasible]
        if feasible:
            return min(feasible, key=lambda item: item.score)
        return min(mpc_items, key=lambda item: item.score)

    @staticmethod
    def _local_astar_items(evaluations: Sequence[CandidateEvaluation]) -> List[CandidateEvaluation]:
        return [item for item in evaluations if item.trajectory.name.startswith("local_astar:")]

    @staticmethod
    def _is_sampling_mpc_candidate(candidate: CandidateTrajectory) -> bool:
        return bool((candidate.metadata or {}).get("planner") == "sampling_mpc")

    @staticmethod
    def _is_sampling_mpc_evaluation(item: CandidateEvaluation) -> bool:
        return bool((item.trajectory.metadata or {}).get("planner") == "sampling_mpc")

    @staticmethod
    def _is_stop_candidate_obj(candidate: CandidateTrajectory) -> bool:
        metadata = candidate.metadata or {}
        kind = str(metadata.get("sample_kind", "")).lower()
        name = str(candidate.name).lower()
        return bool(metadata.get("is_stop", False) or kind in {"stop", "brake"} or name.endswith(":stop") or name.endswith(":brake"))

    @classmethod
    def _is_stop_evaluation(cls, item: CandidateEvaluation) -> bool:
        return cls._is_stop_candidate_obj(item.trajectory)

    @staticmethod
    def _is_moving_candidate_obj(candidate: CandidateTrajectory) -> bool:
        return candidate.velocity.norm() > 0.05

    @classmethod
    def _is_moving_evaluation(cls, item: CandidateEvaluation) -> bool:
        return cls._is_moving_candidate_obj(item.trajectory)

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
