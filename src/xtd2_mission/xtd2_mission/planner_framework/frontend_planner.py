#!/usr/bin/env python3
"""Front-end candidate trajectory generator.

This frontend is path-centered rather than behavior-centered:

1. Local A* / PathTracker owns the nominal path-following behavior.
2. CorridorSelector is only a multi-obstacle search bias.
3. The frontend generates a small family of variants around the nominal path.
4. Escape candidates are emergency fallback only.
"""

import math
from typing import Iterable, List, Optional, Tuple

from .common import CandidateTrajectory, PerceptionData, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .corridor_selector import CorridorSelector
from .local_astar_passable import LocalAStarPlanner
from .sampling_mpc_planner import SamplingMPCPlanner


class FrontendPlanner:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.local_astar = LocalAStarPlanner(self.config)
        self.sampling_mpc = SamplingMPCPlanner(self.config)
        self.corridor_selector = CorridorSelector(self.config)
        self.diagnostics = {}
        self._escape_side = 0.0
        self._avoid_active = False
        self._avoid_side = 0.0
        self._avoid_until = 0.0
        self._last_risk = 0.0
        self._corridor_lock_until = 0.0
        self._corridor_lock_active = False
        self._escape_hold_until = 0.0
        self._last_escape_stamp = -999.0

    def generate(self, state: PlannerState, local_goal: Vec3, perception: Optional[PerceptionData] = None) -> List[CandidateTrajectory]:
        candidates: List[CandidateTrajectory] = []
        mode = str(self.config.frontend_mode or "local_astar").lower()
        if mode in ("sampling_mpc", "mpc", "sampling"):
            candidates = self.sampling_mpc.generate(state, local_goal, perception)
            self.diagnostics = dict(self.sampling_mpc.diagnostics)
            self._escape_side = 0.0
            self._avoid_active = False
            self._avoid_side = 0.0
            self._corridor_lock_active = False
            self._corridor_lock_until = 0.0
            self._escape_hold_until = 0.0
            self.corridor_selector.reset()
        elif perception is not None and mode in ("local_astar", "astar", "hybrid_astar", "hybrid"):
            risk = self._update_risk_memory(state, local_goal, perception)

            # Corridor selection is only a hint for local A* in multi-obstacle scenes.
            # It must not directly create a control action or disturb single-pillar tests.
            obstacle_count = sum(1 for obs in perception.obstacles if obs.source not in ("boundary", "lidar_near_field"))
            enable_corridor = bool(obstacle_count >= 2)
            if enable_corridor:
                corridor_decision = self.corridor_selector.update(state, local_goal, perception, risk)
            else:
                self.corridor_selector.reset()
                corridor_decision = self.corridor_selector._last_decision
                self._corridor_lock_active = False
                self._corridor_lock_until = 0.0

            if corridor_decision.active and abs(corridor_decision.side) > 0.5 and not self._corridor_lock_active:
                try:
                    self.local_astar._side_latch = corridor_decision.side
                    self.local_astar._side_latch_until = max(
                        getattr(self.local_astar, "_side_latch_until", 0.0),
                        state.stamp + 1.5,
                    )
                except Exception:
                    pass

            astar_candidate = self.local_astar.plan_candidate(state, local_goal, perception.obstacles)
            astar_direct = bool(astar_candidate is not None and astar_candidate.name == "local_astar:direct")
            candidate_reason = ""
            candidate_margin = 999.0
            candidate_progress = 0.0
            path_variants: List[CandidateTrajectory] = []

            if astar_candidate is not None:
                candidate_reason = str(astar_candidate.metadata.get("safety_reason", astar_candidate.metadata.get("reason", "")) or "")
                candidate_margin = float(astar_candidate.metadata.get("min_path_clearance", 999.0) or 999.0)
                to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
                goal_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
                candidate_progress = astar_candidate.velocity.x * goal_dir.x + astar_candidate.velocity.y * goal_dir.y
                self._tag_path_candidate(astar_candidate, "primary", enable_corridor, corridor_decision, candidate_progress)
                candidates.append(astar_candidate)

                # Important: a mature path-centric frontend does not output only one
                # nominal path candidate.  It outputs a small candidate family around
                # the same path direction so SafetyChecker/BackendSelector can still
                # choose a safe command when the centerline is temporarily too close.
                path_variants = self._path_centered_variants(state, local_goal, astar_candidate, candidate_progress)
                for cand in path_variants:
                    self._tag_path_candidate(cand, cand.metadata.get("path_variant", "variant"), enable_corridor, corridor_decision, candidate_progress)
                candidates.extend(path_variants)

            risk_value = float(risk.get("risk", 0.0) or 0.0)
            current_clearance = float(risk.get("clearance", 999.0) or 999.0)
            path_available = astar_candidate is not None
            path_forward_ok = bool(path_available and candidate_progress > 0.0 and candidate_margin >= 0.40)
            critical_current = bool(current_clearance < 0.20)
            path_missing_or_unusable = bool(not path_available or (candidate_margin < 0.25 and candidate_progress <= 0.0))

            # Escape remains a fallback.  It is allowed when no path exists, when the
            # path is clearly unusable, or when current clearance is already critical.
            # It is not a normal alternative to the path-centered variants.
            escape_candidates: List[CandidateTrajectory] = []
            allow_escape = bool(path_missing_or_unusable or critical_current)
            if allow_escape:
                escape_candidates = self._hard_zone_escape_candidates(
                    state,
                    local_goal,
                    perception,
                    allow_early_escape=not path_forward_ok and risk_value > 0.90,
                )
            if escape_candidates:
                self._last_escape_stamp = state.stamp
                self._escape_hold_until = max(self._escape_hold_until, state.stamp + 0.25)
                for cand in escape_candidates:
                    cand.metadata["emergency_only"] = True
                    cand.metadata["path_primary_available"] = bool(path_available)
                    cand.metadata["corridor_selector"] = corridor_decision.to_dict()
                candidates.extend(escape_candidates)

            if path_forward_ok and not critical_current:
                self._escape_hold_until = 0.0
                self._avoid_active = False
                self._avoid_side = 0.0
                self._escape_side = 0.0
                self._avoid_until = 0.0

            self.diagnostics = {
                "mode": mode,
                "local_astar": dict(self.local_astar.diagnostics),
                "risk_field": dict(risk),
                "path_centered": True,
                "path_available": bool(path_available),
                "path_forward_ok": bool(path_forward_ok),
                "path_variants": len(path_variants),
                "candidate_progress": float(candidate_progress),
                "candidate_margin": float(candidate_margin),
                "candidate_reason": candidate_reason,
                "critical_current": bool(critical_current),
                "allow_escape": bool(allow_escape),
                "escape_candidates": len(escape_candidates),
                "corridor_enabled": bool(enable_corridor),
                "corridor_selector": corridor_decision.to_dict(),
                "astar_direct": bool(astar_direct),
            }
        else:
            self.diagnostics = {"mode": mode, "path_centered": False}
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

    def _tag_path_candidate(self, candidate, variant: str, enable_corridor: bool, corridor_decision, progress: float) -> None:
        candidate.metadata["path_primary"] = variant == "primary"
        candidate.metadata["path_centered"] = True
        candidate.metadata["path_variant"] = str(variant)
        candidate.metadata["corridor_enabled"] = bool(enable_corridor)
        candidate.metadata["corridor_selector"] = corridor_decision.to_dict()
        candidate.metadata["risk"] = float(self._last_risk)
        candidate.metadata["candidate_progress"] = float(progress)

    def _path_centered_variants(self, state: PlannerState, local_goal: Vec3, primary: CandidateTrajectory,
                                primary_progress: float) -> List[CandidateTrajectory]:
        variants: List[CandidateTrajectory] = []
        base = Vec3(primary.velocity.x, primary.velocity.y, 0.0)
        speed = base.norm()
        if speed <= 1.0e-6:
            return variants

        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        goal_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        base_dir = base.normalized(goal_dir)
        left = Vec3(-base_dir.y, base_dir.x, 0.0)
        max_speed = max(self.config.max_speed, 1.0e-6)

        # Slow path-follow: useful when the nominal path is directionally correct but
        # the current clearance is tight.
        slow = self._rollout("local_astar:path_slow", base_dir * min(speed * 0.55, max_speed * 0.45), state.position)
        slow.metadata["path_variant"] = "slow"
        variants.append(slow)

        # Around-path variants: still move with the path, but bias slightly to either
        # side.  These are not independent escape behaviors; they are local rollout
        # variants around the same nominal path direction.
        side_speed = min(max(speed * 0.78, 0.10), max_speed * 0.70)
        lateral_gain = 0.34 if primary_progress >= -0.02 else 0.20
        for side_name, side_vec in (("left", left), ("right", left * -1.0)):
            direction = self._norm_vec(base_dir * 0.95 + side_vec * lateral_gain)
            cand = self._rollout(f"local_astar:path_lateral_forward_{side_name}", direction * side_speed, state.position)
            cand.metadata["path_variant"] = f"lateral_forward_{side_name}"
            variants.append(cand)

        return variants

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
        back_speed = min(0.11, max(0.06, self.config.max_speed * 0.14))
        critical = bool(nearest_clearance < max(0.36, self.config.static_emergency_clearance * 0.75))

        directions = [
            ("local_escape:lateral_latched", preferred, speed),
            ("local_escape:lateral_forward_latched", self._norm_vec(preferred * 0.85 + goal_dir * 0.45), speed * 0.92),
            ("local_escape:back_latched", self._norm_vec(back * 0.25 + preferred), speed * 0.45),
        ]
        if critical:
            directions.append(("local_escape:back", back, back_speed))
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
