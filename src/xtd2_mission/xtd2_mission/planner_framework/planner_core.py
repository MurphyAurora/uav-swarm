#!/usr/bin/env python3
"""Top-level EGO-like planner orchestration."""

import math
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
            min_clearance = float(best.safety.min_clearance)
            min_ttc = float(best.safety.min_ttc)
            if min_clearance < 0.38 or min_ttc < 1.2:
                command = PlannerCommand(Vec3(), "fallback_hover", "hover", f"soft_margin_too_close:{best.safety.reason}")
            else:
                command = PlannerCommand(
                    best.trajectory.velocity.limit_norm(min(0.12, self.config.max_speed)),
                    "planner_cautious",
                    best.trajectory.name,
                    best.safety.reason,
                )

        command = self._lidar_velocity_shield(command, state, perception)
        command = self.output.process(command, dt=dt)

        return {
            "stamp": now,
            "drone_id": state.drone_id,
            "state": {
                "position": state.position.to_dict(),
                "velocity": state.velocity.to_dict(),
                "heading": float(state.heading),
                "heading_deg": float(math.degrees(state.heading)),
                "stamp": float(state.stamp),
            },
            "coordinate_debug": self._coordinate_debug(state, final_goal, local_goal, command.velocity),
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

    def _coordinate_debug(self, state: PlannerState, final_goal: Vec3, local_goal: Vec3, command_velocity: Vec3) -> Dict[str, object]:
        to_final = final_goal - state.position
        to_local = local_goal - state.position
        heading = state.heading
        c = math.cos(heading)
        s = math.sin(heading)
        body_cmd_x = c * command_velocity.x + s * command_velocity.y
        body_cmd_y = -s * command_velocity.x + c * command_velocity.y
        body_goal_x = c * to_local.x + s * to_local.y
        body_goal_y = -s * to_local.x + c * to_local.y
        xy_speed = math.hypot(command_velocity.x, command_velocity.y)
        goal_xy = math.hypot(to_local.x, to_local.y)
        progress_dot = 0.0
        if xy_speed > 1.0e-6 and goal_xy > 1.0e-6:
            progress_dot = (command_velocity.x * to_local.x + command_velocity.y * to_local.y) / (xy_speed * goal_xy)
        enu_to_ned = {"x": float(command_velocity.y), "y": float(command_velocity.x), "z": float(-command_velocity.z)}
        return {
            "to_final_world": to_final.to_dict(),
            "to_local_world": to_local.to_dict(),
            "to_local_body": {"x": float(body_goal_x), "y": float(body_goal_y)},
            "command_world": command_velocity.to_dict(),
            "command_body_from_heading": {"x": float(body_cmd_x), "y": float(body_cmd_y)},
            "command_enu_to_ned_hypothesis": enu_to_ned,
            "progress_dot_to_local": float(progress_dot),
        }

    def _lidar_velocity_shield(self, command: PlannerCommand, state: PlannerState, perception: PerceptionData) -> PlannerCommand:
        self._last_shield_info = {"active": False}
        if command.velocity.norm() <= 1.0e-6:
            return command
        lidar_obs = [obs for obs in perception.obstacles if obs.source == "lidar_near_field"]
        if not lidar_obs:
            return command

        original = Vec3(command.velocity.x, command.velocity.y, 0.0)
        new_v = Vec3(original.x, original.y, 0.0)
        protect_dist = max(1.55, self.config.drone_radius + self.config.obstacle_margin + 0.95)
        hard_stop_dist = max(1.18, self.config.drone_radius + self.config.obstacle_margin + 0.65)
        min_dist = 999.0
        min_clearance = 999.0
        max_closing = 0.0
        projected = 0

        for obs in lidar_obs:
            rel = obs.position - state.position
            rel_xy = Vec3(rel.x, rel.y, 0.0)
            dist = rel_xy.norm()
            if dist <= 1.0e-6:
                continue
            clearance = dist - (self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin)
            if dist < min_dist:
                min_dist = dist
                min_clearance = clearance
            if dist > protect_dist:
                continue
            unit = rel_xy * (1.0 / dist)
            closing = new_v.x * unit.x + new_v.y * unit.y
            max_closing = max(max_closing, closing)
            if dist <= hard_stop_dist and closing > 0.01:
                self._last_shield_info = {
                    "active": True,
                    "action": "brake",
                    "dist": float(dist),
                    "clearance": float(clearance),
                    "closing": float(closing),
                    "old_velocity": command.velocity.to_dict(),
                    "new_velocity": Vec3().to_dict(),
                }
                return PlannerCommand(Vec3(), "fallback_hover", "hover", f"{command.reason}|shield:brake")
            if closing > 0.0:
                new_v = new_v - unit * closing
                projected += 1

        if projected <= 0:
            return command
        if new_v.norm() < 0.035:
            self._last_shield_info = {
                "active": True,
                "action": "project_stop",
                "dist": float(min_dist),
                "clearance": float(min_clearance),
                "closing": float(max_closing),
                "old_velocity": command.velocity.to_dict(),
                "new_velocity": Vec3().to_dict(),
            }
            return PlannerCommand(Vec3(0.0, 0.0, command.velocity.z), "fallback_hover", "hover", f"{command.reason}|shield:project_stop")

        if new_v.x * original.x + new_v.y * original.y < 0.0:
            self._last_shield_info = {
                "active": True,
                "action": "reverse_blocked",
                "dist": float(min_dist),
                "clearance": float(min_clearance),
                "closing": float(max_closing),
                "old_velocity": command.velocity.to_dict(),
                "new_velocity": Vec3().to_dict(),
            }
            return PlannerCommand(Vec3(0.0, 0.0, command.velocity.z), "fallback_hover", "hover", f"{command.reason}|shield:reverse_blocked")

        new_v = new_v.limit_norm(min(command.velocity.norm(), 0.14))
        out = Vec3(new_v.x, new_v.y, command.velocity.z)
        self._last_shield_info = {
            "active": True,
            "action": "project",
            "dist": float(min_dist),
            "clearance": float(min_clearance),
            "closing": float(max_closing),
            "projected_points": int(projected),
            "old_velocity": command.velocity.to_dict(),
            "new_velocity": out.to_dict(),
        }
        return PlannerCommand(out, command.mode, command.source_trajectory, f"{command.reason}|shield:project")

    def reset_output_filter(self) -> None:
        self.output.reset()
