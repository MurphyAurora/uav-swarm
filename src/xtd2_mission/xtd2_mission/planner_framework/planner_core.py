#!/usr/bin/env python3
"""Top-level EGO-like planner orchestration."""

import time
from typing import Dict, Optional, Sequence

from .backend_selector import BackendSelector
from .command_output import CommandOutput
from .common import PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3
from .fallback_manager import FallbackManager
from .frontend_planner import FrontendPlanner
from .goal_manager import LocalGoalManager
from .safety_checker import SafetyChecker


class EgoLikePlannerCore:
    def __init__(self, config: Optional[PlannerConfig] = None, waypoints: Optional[Sequence[Vec3]] = None):
        self.config = config or PlannerConfig()
        self.goal_manager = LocalGoalManager(self.config, waypoints=waypoints)
        self.frontend = FrontendPlanner(self.config)
        self.safety_checker = SafetyChecker(self.config)
        self.backend = BackendSelector(self.config)
        self.fallback = FallbackManager(self.config)
        self.output = CommandOutput(self.config)
        self._last_plan_time: Optional[float] = None

    def set_waypoints(self, waypoints: Sequence[Vec3], reset: bool = True) -> None:
        self.goal_manager.set_waypoints(waypoints, reset=reset)

    def plan(self, state: PlannerState, final_goal: Vec3, perception: Optional[PerceptionData] = None) -> Dict:
        perception = perception or PerceptionData()
        now = time.time()
        dt = None if self._last_plan_time is None else max(0.0, now - self._last_plan_time)
        self._last_plan_time = now

        active_goal = self.goal_manager.active_waypoint(final_goal)
        local_goal = self.goal_manager.compute_local_goal(state, final_goal, perception)
        candidates = self.frontend.generate(state, local_goal, perception)
        safety_reports = [self.safety_checker.evaluate(candidate, state, perception) for candidate in candidates]
        evaluations = self.backend.evaluate_all(candidates, safety_reports, state, local_goal)
        best = self.backend.select_best(evaluations, state, local_goal)

        if self.fallback.should_fallback(best, evaluations, perception):
            command = self.fallback.command(state, perception, evaluations)
        else:
            command = PlannerCommand(best.trajectory.velocity, "planner", best.trajectory.name, best.safety.reason)
        command = self.output.process(command, dt=dt)

        return {
            "stamp": now,
            "drone_id": state.drone_id,
            "active_waypoint_index": self.goal_manager.active_waypoint_index(),
            "active_goal": active_goal.to_dict(),
            "local_goal": local_goal.to_dict(),
            "final_goal": final_goal.to_dict(),
            "mission_reached": self.goal_manager.mission_reached(state, final_goal),
            "command": command.to_dict(),
            "best": best.to_dict() if best else None,
            "candidate_count": len(candidates),
            "safe_count": sum(1 for item in evaluations if item.safety.safe),
            "feasible_count": sum(1 for item in evaluations if item.safety.feasible),
            "frontend": dict(self.frontend.diagnostics),
            "evaluations": [item.to_dict() for item in sorted(evaluations, key=lambda item: item.score)[:5]],
        }

    def reset_output_filter(self) -> None:
        self.output.reset()
