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
    NORMAL_TRACK = "NORMAL_TRACK"
    DYNAMIC_AVOID = "DYNAMIC_AVOID"
    ESCAPE = "ESCAPE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RECOVERY = "RECOVERY"

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
        self._planning_heading: Optional[float] = None
        self._planning_heading_desired: Optional[float] = None
        self.mode = self.NORMAL_TRACK
        self.mode_until = 0.0
        self.safe_streak = 0
        self.unsafe_streak = 0
        self.last_mode_reason = "init"
        self._last_recovery_choice = {
            "type": "none",
            "name": "",
            "reason": "",
            "min_clearance": 999.0,
            "min_static_clearance": 999.0,
        }

    @staticmethod
    def _planner_now(state: PlannerState) -> float:
        try:
            return float(state.stamp)
        except (TypeError, ValueError):
            return time.time()

    def set_waypoints(self, waypoints: Sequence[Vec3], reset: bool = True) -> None:
        self.goal_manager.set_waypoints(waypoints, reset=reset)
        if reset:
            self._planning_heading = None
            self._planning_heading_desired = None
            self.output.reset()

    def plan(self, state: PlannerState, final_goal: Vec3, perception: Optional[PerceptionData] = None) -> Dict:
        perception = perception or PerceptionData()
        now = self._planner_now(state)
        dt = None if self._last_plan_time is None else max(0.0, now - self._last_plan_time)
        self._last_plan_time = now

        active_goal = self.goal_manager.active_waypoint(final_goal)
        local_goal = self.goal_manager.compute_local_goal(state, final_goal, perception)
        planning_state, planning_heading = self._goal_aligned_state(state, local_goal, dt)

        candidates = self.frontend.generate(planning_state, local_goal, perception)
        safety_reports = [self.safety_checker.evaluate(candidate, state, perception) for candidate in candidates]
        evaluations = self.backend.evaluate_all(candidates, safety_reports, state, local_goal)
        best = self.backend.select_best(evaluations, state, local_goal)

        risk = self._risk_summary(evaluations, best, state, perception)
        mode = self._update_mode(now, risk)

        if mode == self.NORMAL_TRACK:
            selected = self._select_normal_candidate(evaluations, best)
            if selected is None:
                command = self.fallback.command(state, perception, evaluations)
            else:
                command = PlannerCommand(selected.trajectory.velocity, "planner", selected.trajectory.name, selected.safety.reason)
        elif mode == self.DYNAMIC_AVOID:
            selected = self._select_dynamic_avoid_candidate(evaluations)
            if selected is None:
                command = self.fallback.command(state, perception, self._non_stop_evaluations(evaluations))
            else:
                command = PlannerCommand(
                    selected.trajectory.velocity.limit_norm(min(0.22, self.config.max_speed)),
                    "planner",
                    selected.trajectory.name,
                    f"{selected.safety.reason}|fsm:dynamic_avoid",
                )
        elif mode == self.ESCAPE:
            command = self.fallback.command(state, perception, evaluations)
        elif mode == self.EMERGENCY_STOP:
            command = PlannerCommand(Vec3(), "fallback_hover", "hover", "emergency_stop")
        elif mode == self.RECOVERY:
            selected = self._select_recovery_candidate(evaluations, float(risk.get("min_clearance", 999.0)))
            if selected is None:
                command = self.fallback.command(state, perception, self._non_stop_evaluations(evaluations))
            else:
                fallback_command = None
                if self._last_recovery_choice.get("type") == "recoverable_rejoin":
                    fallback_command = self.fallback.command(state, perception, evaluations)
                if fallback_command is not None and fallback_command.mode == "fallback_escape":
                    command = fallback_command
                else:
                    command = PlannerCommand(
                        selected.trajectory.velocity.limit_norm(min(0.15, self.config.max_speed)),
                        "planner",
                        selected.trajectory.name,
                        f"{selected.safety.reason}|fsm:recovery",
                    )
        else:
            command = self.fallback.command(state, perception, evaluations)

        if mode == self.ESCAPE and command.source_trajectory in {"track_goal", "slow_goal", "target_slow", "target_left", "target_right"}:
            command = PlannerCommand(Vec3(), "fallback_hover", "hover", "escape_blocked_goal_directed")

        if mode == self.DYNAMIC_AVOID and command.velocity.norm() > 0.20:
            command = PlannerCommand(
                command.velocity.limit_norm(min(0.22, self.config.max_speed)),
                command.mode,
                command.source_trajectory,
                command.reason,
            )
        elif mode == self.RECOVERY and command.velocity.norm() > 0.12:
            command = PlannerCommand(
                command.velocity.limit_norm(min(0.15, self.config.max_speed)),
                command.mode,
                command.source_trajectory,
                command.reason,
            )

        command = self._lidar_velocity_shield(command, state, perception)
        command = self._goal_progress_guard(command, state, local_goal)
        command = self.output.process(command, dt=dt)

        return {
            "stamp": now,
            "drone_id": state.drone_id,
            "state": {
                "position": state.position.to_dict(),
                "velocity": state.velocity.to_dict(),
                "heading": float(state.heading),
                "heading_deg": float(math.degrees(state.heading)),
                "planning_heading": float(planning_heading),
                "planning_heading_deg": float(math.degrees(planning_heading)),
                "planning_heading_desired": float(self._planning_heading_desired or planning_heading),
                "planning_heading_desired_deg": float(math.degrees(self._planning_heading_desired or planning_heading)),
                "stamp": float(state.stamp),
            },
            "coordinate_debug": self._coordinate_debug(state, planning_state, final_goal, local_goal, command.velocity),
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
            "planner_fsm": {
                "mode": self.mode,
                "mode_until": self.mode_until,
                "safe_streak": self.safe_streak,
                "unsafe_streak": self.unsafe_streak,
                "reason": self.last_mode_reason,
                "risk": risk,
                "recovery_choice": dict(self._last_recovery_choice),
            },
            "evaluations": [item.to_dict() for item in sorted(evaluations, key=lambda item: item.score)[:5]],
        }


    def _risk_summary(self, evaluations, best, state: PlannerState, perception: PerceptionData) -> Dict[str, object]:
        safe_count = sum(1 for item in evaluations if item.safety.safe)
        feasible_count = sum(1 for item in evaluations if item.safety.feasible)
        candidate_min_clearance = min((item.safety.min_clearance for item in evaluations), default=999.0)
        min_clearance = self._current_min_clearance(state, perception)
        recoverable_count = sum(1 for item in evaluations if self._is_recoverable_rejoin(item, min_clearance))
        min_ttc = min((item.safety.min_ttc for item in evaluations), default=999.0)
        best_name = best.trajectory.name if best is not None else None
        return {
            "safe_count": int(safe_count),
            "feasible_count": int(feasible_count),
            "recoverable_count": int(recoverable_count),
            "min_clearance": float(min_clearance),
            "candidate_min_clearance": float(candidate_min_clearance),
            "min_ttc": float(min_ttc),
            "near_field_danger": bool(perception.near_field_danger),
            "best_name": best_name,
            "best_safe": bool(best.safety.safe) if best is not None else False,
            "best_feasible": bool(best.safety.feasible) if best is not None else False,
            "best_goal_directed": bool(best is not None and self._is_goal_directed_name(best.trajectory.name)),
        }


    def _current_min_clearance(self, state: PlannerState, perception: PerceptionData) -> float:
        min_clearance = 999.0
        for obs in perception.obstacles:
            rel = obs.position_at(0.0) - state.position
            clearance = rel.norm() - self._execution_radius(obs)
            min_clearance = min(min_clearance, clearance)
        return float(min_clearance)

    def _execution_radius(self, obs) -> float:
        if obs.source == "lidar_near_field":
            return self.config.drone_radius + float(obs.radius) + 0.20
        if obs.source == "static":
            return self.config.drone_radius + float(obs.radius) + min(self.config.obstacle_margin, 0.10)
        return self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin

    def _update_mode(self, now: float, risk: Dict[str, object]) -> str:
        safe_count = int(risk.get("safe_count", 0) or 0)
        recoverable_count = int(risk.get("recoverable_count", 0) or 0)
        min_clearance = float(risk.get("min_clearance", 999.0) or 999.0)
        min_ttc = float(risk.get("min_ttc", 999.0) or 999.0)
        near_field_danger = bool(risk.get("near_field_danger", False))

        if safe_count > 0 or recoverable_count > 0:
            self.safe_streak += 1
            self.unsafe_streak = 0
        else:
            self.unsafe_streak += 1
            self.safe_streak = 0

        emergency_ttc = float(getattr(self.config, "emergency_ttc", 0.80))
        emergency_hold_sec = float(getattr(self.config, "emergency_hold_sec", 0.80))
        escape_hold_sec = float(getattr(self.config, "escape_hold_sec", 0.60))
        recovery_hold_sec = float(getattr(self.config, "recovery_hold_sec", 0.80))
        recovery_safe_frames = int(getattr(self.config, "recovery_safe_frames", 3))
        recovery_clearance = float(getattr(self.config, "recovery_clearance", max(0.75, self.config.emergency_clearance)))
        dynamic_clearance = float(getattr(self.config, "dynamic_avoid_clearance", max(0.75, self.config.emergency_clearance)))
        dynamic_ttc = float(getattr(self.config, "dynamic_avoid_ttc", 1.20))

        if near_field_danger or min_clearance < 0.0 or min_ttc < emergency_ttc:
            self._set_mode(self.EMERGENCY_STOP, now + emergency_hold_sec, "emergency_risk")
            return self.mode

        if self.mode == self.EMERGENCY_STOP:
            if now < self.mode_until:
                self.last_mode_reason = "emergency_hold"
                return self.mode
            if safe_count > 0 or recoverable_count > 0:
                self._set_mode(self.RECOVERY, now + recovery_hold_sec, "emergency_recovery")
            else:
                self._set_mode(self.ESCAPE, now + escape_hold_sec, "emergency_escape")
            return self.mode

        if self.mode in (self.NORMAL_TRACK, self.DYNAMIC_AVOID):
            if safe_count == 0 and recoverable_count == 0:
                self._set_mode(self.ESCAPE, now + escape_hold_sec, "no_safe_candidates")
            elif safe_count == 0:
                self._set_mode(self.RECOVERY, now + recovery_hold_sec, "recoverable_rejoin")
            elif min_clearance < dynamic_clearance or min_ttc < dynamic_ttc:
                self._set_mode(self.DYNAMIC_AVOID, now, "elevated_risk")
            else:
                self._set_mode(self.NORMAL_TRACK, now, "risk_low")
            return self.mode

        if self.mode == self.ESCAPE:
            if now < self.mode_until:
                self.last_mode_reason = "escape_hold"
                return self.mode
            if safe_count > 0 or recoverable_count > 0:
                self._set_mode(self.RECOVERY, now + recovery_hold_sec, "escape_recovery")
            else:
                self._set_mode(self.ESCAPE, now + escape_hold_sec, "escape_no_safe_candidates")
            return self.mode

        if self.mode == self.RECOVERY:
            if safe_count == 0 and recoverable_count == 0:
                self._set_mode(self.ESCAPE, now + escape_hold_sec, "recovery_lost_safe_candidates")
                return self.mode
            if now < self.mode_until:
                self.last_mode_reason = "recovery_hold"
                return self.mode
            if safe_count > 0 and self.safe_streak >= recovery_safe_frames and min_clearance > recovery_clearance:
                self._set_mode(self.NORMAL_TRACK, now, "recovery_complete")
            else:
                self.last_mode_reason = "recovery_wait_clearance"
            return self.mode

        self._set_mode(self.NORMAL_TRACK, now, "unknown_mode_reset")
        return self.mode

    def _set_mode(self, mode: str, mode_until: float, reason: str) -> None:
        self.mode = mode
        self.mode_until = float(mode_until)
        self.last_mode_reason = reason

    def _select_normal_candidate(self, evaluations, best):
        safe = [item for item in evaluations if item.safety.safe]
        if best is not None and best.safety.safe:
            return best
        return min(safe, key=lambda item: item.score) if safe else None

    def _select_dynamic_avoid_candidate(self, evaluations):
        safe = [item for item in evaluations if item.safety.safe]
        if not safe:
            return None
        moving = [item for item in safe if item.trajectory.velocity.norm() > 0.035]
        moving_non_stop = [item for item in moving if not self._is_stop_like_candidate(item)]
        if not moving_non_stop:
            return None
        return min(
            moving_non_stop,
            key=lambda item: (
                0 if self._is_dynamic_avoid_preferred(item) else 1,
                item.costs.get("reverse", 999.0),
                item.costs.get("progress", 999.0),
                -float(item.safety.min_clearance),
                -float(item.safety.min_ttc),
                float(item.trajectory.metadata.get("sample_cost", item.score)),
                item.costs.get("goal", 999.0),
                item.score,
            ),
        )

    def _is_dynamic_avoid_preferred(self, item) -> bool:
        if self._is_avoid_preferred_name(item.trajectory.name):
            return True
        if item.trajectory.metadata.get("planner") != "sampling_mpc":
            return False
        kind = str(item.trajectory.metadata.get("sample_kind", "")).lower()
        return any(token in kind for token in ("side", "wide", "bias"))

    @staticmethod
    def _is_stop_like_candidate(item) -> bool:
        kind = str(item.trajectory.metadata.get("sample_kind", "")).lower()
        name = str(item.trajectory.name).lower()
        return kind in {"stop", "brake"} or name.endswith(":stop") or name.endswith(":brake")

    def _non_stop_evaluations(self, evaluations):
        filtered = [item for item in evaluations if not self._is_stop_like_candidate(item)]
        return filtered or list(evaluations)

    def _select_recovery_candidate(self, evaluations, current_clearance: float = 999.0):
        safe = [item for item in evaluations if item.safety.safe]
        safe_bypass = [
            item for item in safe
            if self._is_recovery_preferred_name(item.trajectory.name)
            or self._is_committed_astar(item)
            or self._is_dynamic_avoid_preferred(item)
        ]
        if safe_bypass:
            selected = min(safe_bypass, key=self._recovery_rank_key)
            self._record_recovery_choice(selected, "safe_bypass")
            return selected

        recoverable = [
            item for item in evaluations
            if self._is_recoverable_rejoin(item, current_clearance)
        ]
        if recoverable:
            selected = min(recoverable, key=self._recovery_rank_key)
            self._record_recovery_choice(selected, "recoverable_rejoin")
            return selected

        safe_other = [item for item in safe if item not in safe_bypass]
        if safe_other:
            moving_other = [item for item in safe_other if item.trajectory.velocity.norm() > 0.035]
            moving_non_stop = [item for item in moving_other if not self._is_stop_like_candidate(item)]
            if moving_non_stop:
                selected = min(moving_non_stop, key=self._recovery_rank_key)
                self._record_recovery_choice(selected, "safe_other")
                return selected

        self._record_recovery_choice(None, "none")
        return None

    def _record_recovery_choice(self, item, choice_type: str) -> None:
        if item is None:
            self._last_recovery_choice = {
                "type": choice_type,
                "name": "",
                "reason": "",
                "min_clearance": 999.0,
                "min_static_clearance": 999.0,
            }
            return
        self._last_recovery_choice = {
            "type": choice_type,
            "name": str(item.trajectory.name),
            "reason": str(item.safety.reason),
            "min_clearance": float(item.safety.min_clearance),
            "min_static_clearance": float(item.safety.min_static_clearance),
        }

    @staticmethod
    def _is_recovery_preferred_name(name: str) -> bool:
        lower = str(name).lower()
        return bool(
            lower.startswith("local_astar:")
            or lower.startswith("corridor:")
            or lower.startswith("local_escape:")
            or "lateral" in lower
            or "bypass" in lower
            or "rejoin" in lower
        )

    @staticmethod
    def _is_committed_astar(item) -> bool:
        metadata = item.trajectory.metadata or {}
        replan_reason = str(metadata.get("replan_reason", "")).lower()
        label = str(metadata.get("label", "")).lower()
        joined = f"{replan_reason} {label}"
        return bool("committed" in joined or "bypass" in joined or "rejoin" in joined)

    def _recovery_rank_key(self, item):
        return (
            0 if self._is_committed_astar(item) or self._is_dynamic_avoid_preferred(item) else 1,
            1 if self._is_stop_like_candidate(item) else 0,
            0 if item.trajectory.velocity.norm() > 0.035 else 1,
            item.costs.get("reverse", 999.0),
            item.costs.get("progress", 999.0),
            -float(item.safety.min_clearance),
            -float(item.safety.min_ttc),
            float(item.trajectory.metadata.get("sample_cost", item.score)),
            item.costs.get("goal", 999.0),
            item.costs.get("lateral", 999.0),
            item.score,
        )

    def _is_recoverable_rejoin(self, item, current_clearance: float) -> bool:
        allowed_reasons = {"rejoining_corridor", "emergency_margin_violation"}
        default_clearance = max(float(getattr(self.config, "static_hard_clearance", 0.25)), self.config.hard_clearance) + 0.01
        rejoin_clearance = float(getattr(self.config, "recoverable_rejoin_clearance", default_clearance))
        static_floor = max(0.28, float(getattr(self.config, "static_hard_clearance", 0.28)))
        name = str(item.trajectory.name)
        if current_clearance < rejoin_clearance:
            return False
        if self._is_goal_directed_name(name):
            return False

        return bool(
            item.safety.feasible
            and not item.safety.safe
            and item.safety.reason in allowed_reasons
            and name.startswith("local_astar:")
            and item.safety.min_static_clearance >= static_floor
            and item.safety.min_clearance >= -0.02
            and item.trajectory.velocity.norm() <= 0.40
        )

    @staticmethod
    def _is_avoid_preferred_name(name: str) -> bool:
        return name.startswith("local_astar:") or "lateral" in name

    @staticmethod
    def _is_goal_directed_name(name: str) -> bool:
        return name in {"track_goal", "slow_goal", "target_slow", "target_left", "target_right"}

    def _goal_aligned_state(self, state: PlannerState, local_goal: Vec3, dt: Optional[float]) -> tuple[PlannerState, float]:
        to_goal = local_goal - state.position
        if math.hypot(to_goal.x, to_goal.y) > 0.20:
            desired = math.atan2(to_goal.y, to_goal.x)
        else:
            desired = self._planning_heading if self._planning_heading is not None else state.heading
        self._planning_heading_desired = desired
        if self._planning_heading is None:
            self._planning_heading = desired
        else:
            # A mature local planner does not rotate its planning frame by tens of
            # degrees every frame.  Rate-limit the frame so subgoals and tracker
            # velocity stay consistent while still following the mission direction.
            step = max(0.08, min(0.22, 0.85 * (dt if dt is not None else 0.1)))
            delta = math.atan2(math.sin(desired - self._planning_heading), math.cos(desired - self._planning_heading))
            delta = max(-step, min(step, delta))
            self._planning_heading = math.atan2(
                math.sin(self._planning_heading + delta),
                math.cos(self._planning_heading + delta),
            )
        heading = self._planning_heading
        return PlannerState(
            drone_id=state.drone_id,
            position=state.position,
            velocity=state.velocity,
            stamp=state.stamp,
            heading=heading,
        ), heading

    def _coordinate_debug(self, state: PlannerState, planning_state: PlannerState, final_goal: Vec3, local_goal: Vec3, command_velocity: Vec3) -> Dict[str, object]:
        to_final = final_goal - state.position
        to_local = local_goal - state.position
        heading = state.heading
        pc = math.cos(planning_state.heading)
        ps = math.sin(planning_state.heading)
        c = math.cos(heading)
        s = math.sin(heading)
        body_cmd_x = c * command_velocity.x + s * command_velocity.y
        body_cmd_y = -s * command_velocity.x + c * command_velocity.y
        plan_cmd_x = pc * command_velocity.x + ps * command_velocity.y
        plan_cmd_y = -ps * command_velocity.x + pc * command_velocity.y
        plan_goal_x = pc * to_local.x + ps * to_local.y
        plan_goal_y = -ps * to_local.x + pc * to_local.y
        body_goal_x = c * to_local.x + s * to_local.y
        body_goal_y = -s * to_local.x + c * to_local.y
        xy_speed = math.hypot(command_velocity.x, command_velocity.y)
        goal_xy = math.hypot(to_local.x, to_local.y)
        progress_dot = 0.0
        if xy_speed > 1.0e-6 and goal_xy > 1.0e-6:
            progress_dot = (command_velocity.x * to_local.x + command_velocity.y * to_local.y) / (xy_speed * goal_xy)
        return {
            "to_final_world": to_final.to_dict(),
            "to_local_world": to_local.to_dict(),
            "to_local_body_vehicle_heading": {"x": float(body_goal_x), "y": float(body_goal_y)},
            "to_local_body_planning_heading": {"x": float(plan_goal_x), "y": float(plan_goal_y)},
            "command_world": command_velocity.to_dict(),
            "command_body_vehicle_heading": {"x": float(body_cmd_x), "y": float(body_cmd_y)},
            "command_body_planning_heading": {"x": float(plan_cmd_x), "y": float(plan_cmd_y)},
            "command_enu_to_ned_hypothesis": {"x": float(command_velocity.y), "y": float(command_velocity.x), "z": float(-command_velocity.z)},
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
        protect_dist = 1.25
        hard_stop_dist = 0.88
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
            clearance = dist - (self.config.drone_radius + float(obs.radius) + 0.20)
            if dist < min_dist:
                min_dist = dist
                min_clearance = clearance
            if dist > protect_dist:
                continue
            unit = rel_xy * (1.0 / dist)
            closing = new_v.x * unit.x + new_v.y * unit.y
            max_closing = max(max_closing, closing)
            if dist <= hard_stop_dist and closing > 0.02:
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
        if new_v.norm() < 0.02:
            if min_dist <= 0.95:
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
            return command

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

        # Keep the projected vector, but do not cap it to a crawl.  PathTracker
        # and planner_cautious already bound the speed; the shield only removes
        # the dangerous component.
        new_v = new_v.limit_norm(min(command.velocity.norm(), 0.18))
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

    def _goal_progress_guard(self, command: PlannerCommand, state: PlannerState, local_goal: Vec3) -> PlannerCommand:
        if command.velocity.norm() <= 1.0e-6:
            return command
        if command.source_trajectory.startswith("local_escape:") or command.mode == "fallback_escape":
            return command
        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        dist = to_goal.norm()
        if dist < 0.25:
            return command
        unit = to_goal * (1.0 / dist)
        vxy = Vec3(command.velocity.x, command.velocity.y, 0.0)
        progress = vxy.x * unit.x + vxy.y * unit.y
        if progress >= -0.005:
            return command
        corrected = vxy - unit * progress
        if corrected.norm() < 0.02:
            corrected = Vec3()
        out = Vec3(corrected.x, corrected.y, command.velocity.z).limit_norm(command.velocity.norm())
        return PlannerCommand(
            out,
            command.mode,
            command.source_trajectory,
            f"{command.reason}|goal_progress_guard",
        )

    def reset_output_filter(self) -> None:
        self.output.reset()


