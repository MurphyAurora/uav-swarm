#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level EGO-like planner orchestration.

The core wires the framework modules together while keeping each concern
separate: local goal, front-end generation, hard safety, backend selection,
fallback, and command smoothing.
"""

import time
from typing import Dict, Optional

from .backend_selector import BackendSelector
from .command_output import CommandOutput
from .common import PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3
from .fallback_manager import FallbackManager
from .frontend_planner import FrontendPlanner
from .goal_manager import LocalGoalManager
from .safety_checker import SafetyChecker


class EgoLikePlannerCore:
    """Composable local replanning core."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.goal_manager = LocalGoalManager(self.config)
        self.frontend = FrontendPlanner(self.config)
        self.safety_checker = SafetyChecker(self.config)
        self.backend = BackendSelector(self.config)
        self.fallback = FallbackManager(self.config)
        self.output = CommandOutput(self.config)
        self._last_plan_time: Optional[float] = None

    def plan(
        self,
        state: PlannerState,
        final_goal: Vec3,
        perception: Optional[PerceptionData] = None,
    ) -> Dict:
        perception = perception or PerceptionData()
        now = time.time()
        dt = None if self._last_plan_time is None else max(0.0, now - self._last_plan_time)
        self._last_plan_time = now

        local_goal = self.goal_manager.compute_local_goal(state, final_goal, perception)
        candidates = self.frontend.generate(state, local_goal)
        safety_reports = [self.safety_checker.evaluate(candidate, state, perception) for candidate in candidates]
        evaluations = self.backend.evaluate_all(candidates, safety_reports, state, local_goal)
        best = self.backend.select_best(evaluations)

        if self.fallback.should_fallback(best, evaluations, perception):
            command = self.fallback.command(state, perception, evaluations)
        else:
            command = PlannerCommand(
                velocity=best.trajectory.velocity,
                mode="planner",
                source_trajectory=best.trajectory.name,
                reason=best.safety.reason,
            )

        command = self.output.process(command, dt=dt)
        return {
            "stamp": now,
            "drone_id": state.drone_id,
            "local_goal": local_goal.to_dict(),
            "final_goal": final_goal.to_dict(),
            "command": command.to_dict(),
            "best": best.to_dict() if best else None,
            "candidate_count": len(candidates),
            "safe_count": sum(1 for item in evaluations if item.safety.safe),
            "feasible_count": sum(1 for item in evaluations if item.safety.feasible),
            "evaluations": [item.to_dict() for item in sorted(evaluations, key=lambda item: item.score)[:5]],
        }

    def reset_output_filter(self) -> None:
        self.output.reset()
