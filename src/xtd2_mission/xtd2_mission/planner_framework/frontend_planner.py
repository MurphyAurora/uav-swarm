#!/usr/bin/env python3
"""Front-end candidate trajectory generator."""

import math
from typing import Iterable, List, Optional, Tuple

from .common import CandidateTrajectory, PerceptionData, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .corridor_selector import CorridorSelector
from .local_astar_passable import LocalAStarPlanner


class FrontendPlanner:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.local_astar = LocalAStarPlanner(self.config)
        self.corridor_selector = CorridorSelector(self.config)
        self.diagnostics = {}
        self._escape_side = 0.0
        self._avoid_active = False
        self._avoid_side = 0.0
        self._avoid_until = 0.0
        self._last_risk = 0.0
        self._corridor_lock_until = 0.0
        self._corridor_lock_active = False
        # Temporal consistency for constrained passages.  This is intentionally a
        # soft cooldown: it can suppress weak rejoin attempts, but it must not keep
        # generating back/escape candidates once clearance is already acceptable.
        self._escape_hold_until = 0.0
        self._last_escape_stamp = -999.0

    def generate(self, state: PlannerState, local_goal: Vec3, perception: Optional[PerceptionData] = None) -> List[CandidateTrajectory]:
        candidates: List[CandidateTrajectory] = []
        mode = str(self.config.frontend_mode or "local_astar").lower()
        if perception is not None and mode in ("local_astar", "astar", "hybrid_astar", "hybrid"):
            risk = self._update_risk_memory(state, local_goal, perception)
            corridor_decision = self.corridor_selector.update(state, local_goal, perception, risk)
            if corridor_decision.active and abs(corridor_decision.side) > 0.5:
                # Corridor-level preference is stronger than instantaneous nearest
                # obstacle side.  It is the lightweight topology/mode decision layer.
                self._avoid_side = corridor_decision.side
                if abs(self._escape_side) < 0.5:
                    self._escape_side = corridor_decision.side
                if bool(risk.get("front_obstacle", False)) or float(risk.get("risk", 0.0) or 0.0) > 0.25:
                    self._avoid_active = True
                    self._avoid_until = max(self._avoid_until, state.stamp + 1.2)

            escape_hold = state.stamp <= self._escape_hold_until
            if (self._avoid_active or escape_hold or corridor_decision.active) and abs(self._avoid_side or self._escape_side or corridor_decision.side) > 0.5 and not self._corridor_lock_active:
                try:
                    side = self._avoid_side if abs(self._avoid_side) > 0.5 else (self._escape_side if abs(self._escape_side) > 0.5 else corridor_decision.side)
                    self.local_astar._side_latch = side
                    self.local_astar._side_latch_until = max(
                        getattr(self.local_astar, "_side_latch_until", 0.0),
                        state.stamp + (1.8 if corridor_decision.active else 1.2),
                    )
                except Exception:
                    pass

            astar_candidate = self.local_astar.plan_candidate(state, local_goal, perception.obstacles)
            astar_direct = bool(astar_candidate is not None and astar_candidate.name == "local_astar:direct")
            candidate_reason = ""
            candidate_margin = 999.0
            if astar_candidate is not None:
                candidate_reason = str(astar_candidate.metadata.get("safety_reason", astar_candidate.metadata.get("reason", "")) or "")
                candidate_margin = float(astar_candidate.metadata.get("min_path_clearance", 999.0) or 999.0)

            risk_value = float(risk.get("risk", 1.0) or 1.0)
            # If a candidate is already well clear, stop holding the previous escape.
            # This is the key guard that prevents single_pillar from freezing after
            # it has safely gone around the obstacle.
            if escape_hold and candidate_margin >= 0.88 and risk_value < 0.70:
                self._escape_hold_until = 0.0
                escape_hold = False

            if astar_direct:
                self._update_corridor_lock(state, astar_candidate, risk)
            else:
                self._release_corridor_lock_if_expired(state, risk)

            corridor_locked = self._corridor_lock_active and state.stamp <= self._corridor_lock_until
            low_risk_direct = bool(astar_direct and risk.get("risk", 1.0) < 0.25 and not risk.get("front_obstacle", False))
            allow_rejoin_during_hold = bool(
                not escape_hold
                or low_risk_direct
                or (candidate_margin >= 0.88 and risk_value < 0.70)
            )
            if low_risk_direct and not corridor_decision.active:
                self._avoid_active = False
                self._avoid_side = 0.0
                self._escape_side = 0.0
                self._avoid_until = 0.0
                self._escape_hold_until = 0.0

            self.diagnostics = {
                "mode": mode,
                "local_astar": dict(self.local_astar.diagnostics),
                "risk_field": dict(risk),
                "corridor_selector": corridor_decision.to_dict(),
                "avoid_active": bool(self._avoid_active),
                "avoid_side": float(self._avoid_side),
                "avoid_until": float(self._avoid_until),
                "astar_direct": bool(astar_direct),
                "low_risk_direct": bool(low_risk_direct),
                "corridor_locked": bool(corridor_locked),
                "corridor_lock_until": float(self._corridor_lock_until),
                "escape_hold": bool(escape_hold),
                "escape_hold_until": float(self._escape_hold_until),
                "allow_rejoin_during_hold": bool(allow_rejoin_during_hold),
                "candidate_margin": float(candidate_margin),
                "candidate_reason": candidate_reason,
            }
            if astar_candidate is not None and allow_rejoin_during_hold:
                astar_candidate.metadata["avoid_active"] = bool(self._avoid_active)
                astar_candidate.metadata["avoid_side"] = float(self._avoid_side)
                astar_candidate.metadata["risk"] = float(self._last_risk)
                astar_candidate.metadata["corridor_locked"] = bool(corridor_locked)
                astar_candidate.metadata["corridor_selector"] = corridor_decision.to_dict()
                astar_candidate.metadata["escape_hold"] = bool(escape_hold)
                candidates.append(astar_candidate)
            elif astar_candidate is not None:
                self.diagnostics["suppressed_astar"] = astar_candidate.name

            escape_candidates: List[CandidateTrajectory] = []
            if not corridor_locked and not low_risk_direct:
                escape_candidates = self._hard_zone_escape_candidates(
                    state,
                    local_goal,
                    perception,
                    # Do not use escape_hold itself as early-escape permission.
                    # Otherwise a previously valid escape can keep regenerating back
                    # motions forever even after clearance has improved.
                    allow_early_escape=astar_candidate is None or self._last_risk > 0.70,
                )
            if escape_candidates:
                self._last_escape_stamp = state.stamp
                # Shorter hold than before; enough to avoid one-frame flip-flop but
                # not enough to dominate easy maps.
                self._escape_hold_until = max(self._escape_hold_until, state.stamp + 0.45)
                for cand in escape_candidates:
                    cand.metadata["escape_hold_until"] = float(self._escape_hold_until)
                    cand.metadata["corridor_selector"] = corridor_decision.to_dict()
                candidates.extend(escape_candidates)
                self.diagnostics["escape_candidates"] = len(escape_candidates)
                self.diagnostics["escape_hold_until"] = float(self._escape_hold_until)
            elif astar_candidate is not None and not candidates:
                astar_candidate.metadata["forced_after_empty_escape"] = True
                candidates.append(astar_candidate)
        else:
            self.diagnostics = {"mode": mode}
            self._escape_side = 0.0
            self._avoid_active = False
            self._avoid_side = 0.0
            self._corridor_lock_active = False
            self._corridor_lock_until = 0.0
            self._escape_hold_until = 0.0
            self.corridor_selector.reset()
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

    def _update_corridor_lock(self, state: PlannerState, astar_candidate: CandidateTrajectory, risk: dict) -> None:
        if astar_candidate is None:
            return
        path_clearance = float(astar_candidate.metadata.get("min_path_clearance", 999.0) or 999.0)
        risk_value = float(risk.get("risk", 1.0) or 1.0)
        front_obstacle = bool(risk.get("front_obstacle", False))
        enter_lock = path_clearance >= 0.55 and (risk_value <= 0.75 or not front_obstacle) and state.stamp > self._escape_hold_until
        if enter_lock:
            self._corridor_lock_active = True
            self._corridor_lock_until = max(self._corridor_lock_until, state.stamp + 2.2)
            self._escape_side = 0.0

    def _release_corridor_lock_if_expired(self, state: PlannerState, risk: dict) -> None:
        if not self._corridor_lock_active:
            return
        if state.stamp > self._corridor_lock_until:
            self._corridor_lock_active = False
            return
        if float(risk.get("risk", 1.0) or 1.0) > 0.92:
            self._corridor_lock_active = False

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
        if enter and not self._corridor_lock_active:
            if not self._avoid_active or abs(self._avoid_side) < 0.5:
                if abs(obs_lateral) < 0.10:
                    self._avoid_side = 1.0
                else:
                    self._avoid_side = 1.0 if obs_lateral <= 0.0 else -1.0
            self._avoid_active = True
            self._escape_side = self._avoid_side
            self._avoid_until = max(self._avoid_until, state.stamp + 1.6)
        elif release and state.stamp > self._avoid_until and state.stamp > self._escape_hold_until:
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
            if nearest_clearance > close_trigger + 0.20 and not self._avoid_active and state.stamp > self._escape_hold_until:
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
