#!/usr/bin/env python3
"""Front-end candidate trajectory generator."""

import math
from typing import Iterable, List, Optional, Tuple

from .common import CandidateTrajectory, PerceptionData, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .local_astar_passable import LocalAStarPlanner


class FrontendPlanner:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.local_astar = LocalAStarPlanner(self.config)
        self.diagnostics = {}
        self._escape_side = 0.0
        self._avoid_active = False
        self._avoid_side = 0.0
        self._avoid_until = 0.0
        self._last_risk = 0.0

    def generate(self, state: PlannerState, local_goal: Vec3, perception: Optional[PerceptionData] = None) -> List[CandidateTrajectory]:
        candidates: List[CandidateTrajectory] = []
        mode = str(self.config.frontend_mode or "local_astar").lower()
        if perception is not None and mode in ("local_astar", "astar", "hybrid_astar", "hybrid"):
            risk = self._update_risk_memory(state, local_goal, perception)
            if self._avoid_active and abs(self._avoid_side) > 0.5:
                # Feed the same-side memory into LocalAStarPlanner/LocalSubgoalSelector.
                # This is the path-memory buffer: once the vehicle chooses a side,
                # subgoal sampling keeps using that side until the obstacle is
                # actually cleared instead of oscillating back toward the centerline.
                try:
                    self.local_astar._side_latch = self._avoid_side
                    self.local_astar._side_latch_until = max(getattr(self.local_astar, "_side_latch_until", 0.0), state.stamp + 1.2)
                except Exception:
                    pass

            astar_candidate = self.local_astar.plan_candidate(state, local_goal, perception.obstacles)
            astar_direct = bool(astar_candidate is not None and astar_candidate.name == "local_astar:direct")
            if astar_direct:
                # Direct tracking means LocalAStar has found a clear/rejoining
                # corridor.  At this point escape should not compete with the
                # main planner; otherwise rejoin_shell oscillates around the shell
                # instead of returning to the goal line.
                self._avoid_active = False
                self._avoid_side = 0.0
                self._escape_side = 0.0
                self._avoid_until = 0.0
            self.diagnostics = {
                "mode": mode,
                "local_astar": dict(self.local_astar.diagnostics),
                "risk_field": dict(risk),
                "avoid_active": bool(self._avoid_active),
                "avoid_side": float(self._avoid_side),
                "avoid_until": float(self._avoid_until),
                "astar_direct": bool(astar_direct),
            }
            if astar_candidate is not None:
                astar_candidate.metadata["avoid_active"] = bool(self._avoid_active)
                astar_candidate.metadata["avoid_side"] = float(self._avoid_side)
                astar_candidate.metadata["risk"] = float(self._last_risk)
                candidates.append(astar_candidate)

            # Escape is a safety fallback, not the default way to pass a visible
            # obstacle. It is not allowed to override direct/rejoin tracking.
            escape_candidates: List[CandidateTrajectory] = []
            if not astar_direct:
                escape_candidates = self._hard_zone_escape_candidates(
                    state,
                    local_goal,
                    perception,
                    allow_early_escape=astar_candidate is None or self._last_risk > 0.75,
                )
            if escape_candidates:
                candidates.extend(escape_candidates)
                self.diagnostics["escape_candidates"] = len(escape_candidates)
        else:
            self.diagnostics = {"mode": mode}
            self._escape_side = 0.0
            self._avoid_active = False
            self._avoid_side = 0.0
            candidates.append(self._rollout("track_goal", self._desired_velocity_towards_goal(state, local_goal), state.position))
        return candidates

    def generate_from_velocity_set(self, state: PlannerState, velocity_set: Iterable[Tuple[str, Vec3]]) -> List[CandidateTrajectory]:
        return [self._rollout(name, velocity.limit_norm(self.config.max_speed), state.position) for name, velocity in velocity_set]

    def _desired_velocity_towards_goal(self, state: PlannerState, local_goal: Vec3) -> Vec3:
        to_goal = local_goal - state.position
        distance = to_goal.norm()
        if distance < 1.0e-6:
            return Vec3()
        speed = min(self.config.cruise_speed, distance / max(self.config.horizon, 1.0e-6))
        return to_goal.normalized() * speed

    def _rollout(self, name: str, velocity: Vec3, start: Vec3) -> CandidateTrajectory:
        velocity = velocity.limit_norm(self.config.max_speed)
        points = [TrajectoryPoint(t=0.0, position=start)]
        steps = max(1, int(round(self.config.horizon / max(self.config.dt, 1.0e-6))))
        for idx in range(1, steps + 1):
            t = idx * self.config.dt
            points.append(TrajectoryPoint(t=t, position=start + velocity * t))
        return CandidateTrajectory(name=name, velocity=velocity, points=points)

    def _update_risk_memory(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData) -> dict:
        nearest = self._nearest_context(state, local_goal, perception)
        if nearest is None:
            self._last_risk = 0.0
            if state.stamp > self._avoid_until:
                self._avoid_active = False
                self._avoid_side = 0.0
                self._escape_side = 0.0
            return {"risk": 0.0, "nearest": None}

        obs_forward, obs_lateral, clearance, dist = nearest
        front_obstacle = 0.2 <= obs_forward <= 4.5 and abs(obs_lateral) <= 1.8
        clearance_risk = max(0.0, min(1.0, (1.45 - clearance) / 1.10))
        alignment_risk = max(0.0, min(1.0, 1.0 - abs(obs_lateral) / 1.8)) if front_obstacle else 0.0
        risk = max(clearance_risk, alignment_risk * 0.85)
        self._last_risk = float(risk)

        enter = front_obstacle and risk >= 0.35
        release = (not front_obstacle) or clearance >= 1.45 or obs_forward < -0.25 or (obs_forward > 3.2 and abs(obs_lateral) >= 1.65)
        if enter:
            if not self._avoid_active or abs(self._avoid_side) < 0.5:
                if abs(obs_lateral) < 0.10:
                    self._avoid_side = 1.0
                else:
                    self._avoid_side = 1.0 if obs_lateral <= 0.0 else -1.0
            self._avoid_active = True
            self._escape_side = self._avoid_side
            self._avoid_until = max(self._avoid_until, state.stamp + 1.6)
        elif release and state.stamp > self._avoid_until:
            self._avoid_active = False
            self._avoid_side = 0.0
            self._escape_side = 0.0

        return {
            "risk": float(risk),
            "front_obstacle": bool(front_obstacle),
            "obs_forward": float(obs_forward),
            "obs_lateral": float(obs_lateral),
            "clearance": float(clearance),
            "dist": float(dist),
            "release": bool(release),
        }

    def _nearest_context(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData):
        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        goal_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        lateral_left = Vec3(-goal_dir.y, goal_dir.x, 0.0)
        best = None
        best_dist = 999.0
        for obs in perception.obstacles:
            obs_pos = obs.position_at(0.0)
            rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)
            dist = rel.norm()
            if dist <= 1.0e-6 or dist >= best_dist:
                continue
            clearance = dist - self._execution_radius(obs)
            obs_forward = rel.x * goal_dir.x + rel.y * goal_dir.y
            obs_lateral = rel.x * lateral_left.x + rel.y * lateral_left.y
            best = (obs_forward, obs_lateral, clearance, dist)
            best_dist = dist
        return best

    def _hard_zone_escape_candidates(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData,
                                     allow_early_escape: bool = False) -> List[CandidateTrajectory]:
        nearest_obs = None
        nearest_clearance = 999.0
        nearest_rel = None
        for obs in perception.obstacles:
            obs_pos = obs.position_at(0.0)
            rel_from_obs = state.position - obs_pos
            dist_xy = math.hypot(rel_from_obs.x, rel_from_obs.y)
            if dist_xy <= 1.0e-6:
                continue
            clearance = dist_xy - self._execution_radius(obs)
            if clearance < nearest_clearance:
                nearest_clearance = clearance
                nearest_obs = obs
                nearest_rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)

        if nearest_obs is None or nearest_rel is None:
            self._escape_side = 0.0
            return []

        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        goal_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        lateral_left = Vec3(-goal_dir.y, goal_dir.x, 0.0)
        lateral_right = Vec3(goal_dir.y, -goal_dir.x, 0.0)

        obs_forward = nearest_rel.x * goal_dir.x + nearest_rel.y * goal_dir.y
        obs_lateral = nearest_rel.x * lateral_left.x + nearest_rel.y * lateral_left.y
        front_obstacle = 0.2 <= obs_forward <= 3.8 and abs(obs_lateral) <= 1.5
        close_trigger = max(self.config.static_emergency_clearance, 0.65)
        if nearest_clearance > close_trigger and not (allow_early_escape and front_obstacle):
            if nearest_clearance > close_trigger + 0.20 and not self._avoid_active:
                self._escape_side = 0.0
            return []

        if abs(self._escape_side) < 0.5:
            if abs(self._avoid_side) > 0.5:
                self._escape_side = self._avoid_side
            elif abs(obs_lateral) < 0.10:
                self._escape_side = 1.0
            else:
                self._escape_side = 1.0 if obs_lateral <= 0.0 else -1.0
        preferred = lateral_left if self._escape_side > 0.0 else lateral_right
        back = goal_dir * -1.0
        speed = min(0.24, max(0.18, self.config.max_speed * 0.32))
        back_speed = min(0.14, max(0.08, self.config.max_speed * 0.18))

        directions = [
            ("local_escape:lateral_latched", preferred, speed),
            ("local_escape:back_latched", self._norm_vec(back * 0.35 + preferred), speed * 0.65),
            ("local_escape:back", back, back_speed),
        ]
        return [
            self._rollout(name, direction * cmd_speed, state.position)
            for name, direction, cmd_speed in directions
        ]

    def _execution_radius(self, obs) -> float:
        if obs.source == "lidar_near_field":
            return self.config.drone_radius + float(obs.radius) + 0.20
        if obs.source == "static":
            return self.config.drone_radius + float(obs.radius) + min(self.config.obstacle_margin, 0.10)
        return self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin

    @staticmethod
    def _norm2(x: float, y: float) -> Tuple[float, float]:
        mag = math.hypot(x, y)
        if mag <= 1.0e-6:
            return 1.0, 0.0
        return x / mag, y / mag

    @staticmethod
    def _norm_vec(v: Vec3) -> Vec3:
        return Vec3(v.x, v.y, 0.0).normalized(Vec3(0.0, 1.0, 0.0))
