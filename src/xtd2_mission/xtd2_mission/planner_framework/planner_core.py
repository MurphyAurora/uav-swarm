#!/usr/bin/env python3
"""Top-level EGO-like planner orchestration."""

import time
from typing import Dict, Optional, Sequence, Tuple

from .backend_selector import BackendSelector
from .command_output import CommandOutput
from .common import Obstacle, PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3
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
        self._last_shield_info: Dict[str, object] = {"active": False}

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

        if best is None or not best.safety.feasible:
            command = self.fallback.command(state, perception, evaluations)
        elif best.safety.safe:
            command = PlannerCommand(best.trajectory.velocity, "planner", best.trajectory.name, best.safety.reason)
        else:
            command = PlannerCommand(
                best.trajectory.velocity.limit_norm(min(0.16, self.config.max_speed)),
                "planner_cautious",
                best.trajectory.name,
                best.safety.reason,
            )

        command = self._lidar_velocity_shield(command, state, perception)
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
            "goal_manager": self.goal_manager.diagnostics(),
            "shield": dict(self._last_shield_info),
            "evaluations": [item.to_dict() for item in sorted(evaluations, key=lambda item: item.score)[:5]],
        }

    def _lidar_velocity_shield(self, command: PlannerCommand, state: PlannerState, perception: PerceptionData) -> PlannerCommand:
        """Project velocity away from close LiDAR obstacles before publishing.

        The local A* path can look clear in the grid while the executed velocity
        still points into a freshly observed near-field point because of latency,
        point-cloud sparsity, or body/world-frame mismatch.  Mature planners use
        a final safety filter/CBF-like layer for this reason.  This shield does
        not plan; it only removes the velocity component that closes distance to
        the nearest LiDAR obstacle and adds a small tangential/away component.
        """
        self._last_shield_info = {"active": False}
        if command.velocity.norm() <= 1.0e-6:
            return command
        nearest = self._nearest_lidar_obstacle(state, perception)
        if nearest is None:
            return command
        obs, dist, clearance, rel = nearest
        protect_dist = max(1.35, self.config.drone_radius + self.config.obstacle_margin + 0.80)
        hard_dist = max(1.05, self.config.drone_radius + self.config.obstacle_margin + 0.55)
        if dist >= protect_dist:
            return command

        away = Vec3(state.position.x - obs.position.x, state.position.y - obs.position.y, 0.0).normalized(Vec3())
        if away.norm() <= 1.0e-6:
            return command
        v = Vec3(command.velocity.x, command.velocity.y, 0.0)
        closing = -(v.x * away.x + v.y * away.y)
        new_v = Vec3(v.x, v.y, 0.0)
        action = "limit"
        if closing > 0.0:
            # Remove the component moving into the obstacle.
            new_v = new_v + away * closing
            action = "project"
        tangent = Vec3(-away.y, away.x, 0.0)
        goal_dir = Vec3(command.velocity.x, command.velocity.y, 0.0).normalized(tangent)
        if tangent.x * goal_dir.x + tangent.y * goal_dir.y < 0.0:
            tangent = tangent * -1.0
        if dist < hard_dist:
            # Near contact: stop forward closing and bias outward.  Keep the
            # speed tiny to avoid oscillation while still preventing a slide-in.
            new_v = away * 0.12 + tangent * 0.05
            action = "hard_away"
        elif new_v.norm() < 0.05:
            new_v = tangent * 0.10 + away * 0.04
            action = "tangent"
        new_v = new_v.limit_norm(min(command.velocity.norm(), 0.18))
        out = Vec3(new_v.x, new_v.y, command.velocity.z)
        self._last_shield_info = {
            "active": True,
            "action": action,
            "dist": float(dist),
            "clearance": float(clearance),
            "closing": float(closing),
            "source": obs.source,
            "old_velocity": command.velocity.to_dict(),
            "new_velocity": out.to_dict(),
        }
        return PlannerCommand(out, command.mode, command.source_trajectory, f"{command.reason}|shield:{action}")

    def _nearest_lidar_obstacle(self, state: PlannerState, perception: PerceptionData) -> Optional[Tuple[Obstacle, float, float, Vec3]]:
        nearest = None
        best_dist = 999.0
        for obs in perception.obstacles:
            if obs.source != "lidar_near_field":
                continue
            rel = obs.position - state.position
            dist = (rel.x * rel.x + rel.y * rel.y + rel.z * rel.z) ** 0.5
            if dist < best_dist:
                clearance = dist - (self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin)
                nearest = (obs, dist, clearance, rel)
                best_dist = dist
        return nearest

    def reset_output_filter(self) -> None:
        self.output.reset()
