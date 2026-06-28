#!/usr/bin/env python3
"""Fallback commands used when no candidate is safely executable."""

import math
from typing import Optional, Sequence, Tuple

from .common import CandidateEvaluation, PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3
from .dynamic_safety import dynamic_escape_command, dynamic_ttc_snapshot


class FallbackManager:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._escape_side = 0.0
        self._escape_direction = Vec3()
        self._escape_forward = Vec3()
        self._dynamic_escape_until = 0.0
        self._dynamic_escape_cmd = Vec3()

    def should_fallback(self, best: Optional[CandidateEvaluation], evaluations: Sequence[CandidateEvaluation], perception: PerceptionData) -> bool:
        if best is None:
            return True
        if not best.safety.feasible:
            return True
        return not any(item.safety.safe for item in evaluations)

    def command(self, state: PlannerState, perception: PerceptionData, evaluations: Sequence[CandidateEvaluation]) -> PlannerCommand:
        """Fallback used when the normal planner has no safely executable command.

        Static scenes can often use hover as a final fallback.  Dynamic scenes
        cannot: if an obstacle is still approaching, hover may keep the drone in
        the collision corridor.  Dynamic TTC escape is therefore checked before
        static recovery and is latched for a short time to avoid the pattern
        "escape two steps -> hover -> get hit".
        """
        best = self._best_escape(evaluations)
        now = float(getattr(state, "stamp", 0.0))

        dynamic_escape = dynamic_escape_command(state, perception, self.config)
        if dynamic_escape is not None:
            hold = max(0.9, float(getattr(self.config, "dynamic_escape_hold_sec", 1.35)))
            self._dynamic_escape_until = max(self._dynamic_escape_until, now + hold)
            self._dynamic_escape_cmd = dynamic_escape
            min_ttc, closing, clearance, _ = dynamic_ttc_snapshot(state, perception, self.config)
            return PlannerCommand(
                dynamic_escape,
                "fallback_escape",
                "dynamic_ttc_escape",
                f"dynamic_ttc_escape:ttc={min_ttc:.2f}:closing={closing:.2f}:clearance={clearance:.2f}",
            )

        latched = self._latched_dynamic_escape(state, perception, now)
        if latched is not None:
            min_ttc, closing, clearance, _ = dynamic_ttc_snapshot(state, perception, self.config)
            return PlannerCommand(
                latched,
                "fallback_escape",
                "dynamic_ttc_escape_latched",
                f"dynamic_ttc_escape_latched:ttc={min_ttc:.2f}:closing={closing:.2f}:clearance={clearance:.2f}",
            )

        if perception.near_field_danger:
            self._escape_side = 0.0
            self._escape_direction = Vec3()
            self._escape_forward = Vec3()
            return PlannerCommand(Vec3(), "fallback_hover", "hover", "near_field_lidar_stop")

        explicit = self._best_explicit_escape(evaluations)
        if explicit is not None:
            speed = min(0.16, max(0.10, self.config.max_speed * 0.24))
            return PlannerCommand(
                explicit.trajectory.velocity.limit_norm(speed),
                "fallback_escape",
                explicit.trajectory.name,
                f"fallback_explicit_escape:{explicit.safety.reason}",
            )

        mpc_recovery = self._best_mpc_feasible_recovery(evaluations)
        if mpc_recovery is not None:
            speed = min(0.12, max(0.055, self.config.max_speed * 0.16))
            return PlannerCommand(
                mpc_recovery.trajectory.velocity.limit_norm(speed),
                "fallback_recovery",
                mpc_recovery.trajectory.name,
                f"fallback_mpc_feasible:{mpc_recovery.safety.reason}",
            )

        fallback = self._lateral_unstick_command(state, perception, best)
        if fallback is not None:
            reason = best.safety.reason if best is not None else "no_safe_candidate"
            return PlannerCommand(fallback, "fallback_escape", "fallback_lateral_unstick", f"fallback_unstick:{reason}")

        if best is not None:
            return PlannerCommand(Vec3(), "fallback_hover", "hover", f"fallback_stop:{best.safety.reason}")
        return PlannerCommand(Vec3(), "fallback_hover", "hover", "no_safe_candidate")

    def _latched_dynamic_escape(self, state: PlannerState, perception: PerceptionData, now: float) -> Optional[Vec3]:
        if now >= self._dynamic_escape_until or self._dynamic_escape_cmd.norm() <= 1.0e-6:
            self._dynamic_escape_cmd = Vec3()
            self._dynamic_escape_until = 0.0
            return None
        direction = self._dynamic_escape_cmd.normalized(Vec3())
        # Do not keep a latched escape if it immediately runs into a static wall.
        # This is a lightweight joint static/dynamic safety check for the fallback
        # command; MPC/SafetyChecker still handle normal candidate safety.
        if self._short_escape_clearance(state, perception, direction) < 0.14:
            self._dynamic_escape_cmd = Vec3()
            self._dynamic_escape_until = 0.0
            return None
        return self._dynamic_escape_cmd.limit_norm(min(float(getattr(self.config, "max_speed", 0.75)), 0.38))

    def _lateral_unstick_command(
        self,
        state: PlannerState,
        perception: PerceptionData,
        best: Optional[CandidateEvaluation],
    ) -> Optional[Vec3]:
        if best is not None and best.safety.safe:
            self._escape_side = 0.0
            self._escape_direction = Vec3()
            self._escape_forward = Vec3()
            return None
        nearest = self._nearest_obstacle_body(state, perception)
        if nearest is None:
            return None
        obs_x, obs_y, clearance, away_x, away_y = nearest

        # Only unstick when the obstacle is in front and there is still enough
        # physical room.  If we are near contact, hover wins for static scenes;
        # dynamic scenes were handled by dynamic_ttc_escape above.
        if not (0.25 <= obs_x <= 4.5 and abs(obs_y) <= 1.8):
            if self._escape_direction.norm() > 1.0e-6 and clearance > 0.52 and best is not None and not best.safety.safe:
                lateral_speed = min(0.18, max(0.12, self.config.max_speed * 0.24))
                return self._escape_direction.limit_norm(1.0) * lateral_speed
            return None
        if clearance < 0.50:
            return None
        if self._escape_direction.norm() > 1.0e-6 and clearance > 0.52 and best is not None and not best.safety.safe:
            lateral_speed = min(0.18, max(0.12, self.config.max_speed * 0.24))
            return self._escape_direction.limit_norm(1.0) * lateral_speed
        if best is not None and best.safety.min_clearance < -0.05 and clearance < 0.52:
            return None

        if self._escape_forward.norm() <= 1.0e-6:
            ch = math.cos(state.heading)
            sh = math.sin(state.heading)
            self._escape_forward = Vec3(ch, sh, 0.0).normalized(Vec3(1.0, 0.0, 0.0))
        forward = self._escape_forward
        left = Vec3(-forward.y, forward.x, 0.0)

        # Keep a side latch, but choose the initial side from the world-frame
        # clearance gradient. Body-frame left/right can flip after an escape turn.
        if abs(self._escape_side) < 0.5:
            away = Vec3(away_x, away_y, 0.0)
            if away.norm() > 1.0e-6:
                self._escape_side = 1.0 if (left.x * away.x + left.y * away.y) >= 0.0 else -1.0
            elif abs(obs_y) < 0.10:
                self._escape_side = 1.0
            else:
                self._escape_side = -1.0 if obs_y > 0.0 else 1.0

        self._refresh_escape_direction_from_clearance(state, perception, forward, left)
        lateral = self._escape_direction.limit_norm(1.0) if self._escape_direction.norm() > 1.0e-6 else (left * self._escape_side)

        if clearance > 0.90:
            lateral_speed = min(0.20, max(0.14, self.config.max_speed * 0.28))
        else:
            lateral_speed = min(0.18, max(0.12, self.config.max_speed * 0.24))
        return lateral.limit_norm(1.0) * lateral_speed

    def _refresh_escape_direction_from_clearance(self, state: PlannerState, perception: PerceptionData, forward: Vec3, left: Vec3) -> None:
        candidates = []
        for side in (-1.0, 1.0):
            for forward_gain in (0.0, 0.35):
                direction = (left * side + forward * forward_gain).normalized(Vec3())
                score = self._short_escape_clearance(state, perception, direction) + 0.03 * forward_gain
                candidates.append((score, side, direction))
        if not candidates:
            return
        best_score, best_side, best_direction = max(candidates, key=lambda item: item[0])
        current_direction = self._escape_direction.normalized(Vec3()) if self._escape_direction.norm() > 1.0e-6 else Vec3()
        current_score = self._short_escape_clearance(state, perception, current_direction) if current_direction.norm() > 1.0e-6 else -999.0
        if self._escape_direction.norm() <= 1.0e-6 or current_score < 0.42 or best_score > current_score + 0.08:
            self._escape_side = best_side
            self._escape_direction = best_direction

    def _short_escape_clearance(self, state: PlannerState, perception: PerceptionData, direction: Vec3) -> float:
        if direction.norm() <= 1.0e-6:
            return -999.0
        min_clearance = 999.0
        speed = min(0.18, max(0.12, self.config.max_speed * 0.24))
        for step in range(1, 7):
            t = step * 0.25
            probe = state.position + direction * (speed * t)
            for obs in perception.obstacles:
                rel = obs.position_at(t) - probe
                clearance = math.hypot(rel.x, rel.y) - (self.config.drone_radius + float(obs.radius))
                if getattr(obs, "source", "") == "boundary" and clearance < 0.75:
                    clearance -= (0.75 - clearance) * 2.0
                min_clearance = min(min_clearance, clearance)
        return float(min_clearance)

    def _nearest_obstacle_body(self, state: PlannerState, perception: PerceptionData) -> Optional[Tuple[float, float, float, float, float]]:
        nearest = None
        nearest_clearance = 999.0
        ch = math.cos(state.heading)
        sh = math.sin(state.heading)
        for obs in perception.obstacles:
            rel = obs.position_at(0.0) - state.position
            rel_xy = Vec3(rel.x, rel.y, 0.0)
            dist = rel_xy.norm()
            if dist <= 1.0e-6:
                continue
            clearance = dist - (self.config.drone_radius + float(obs.radius))
            if clearance >= nearest_clearance:
                continue
            nearest_clearance = clearance
            body_x = ch * rel.x + sh * rel.y
            body_y = -sh * rel.x + ch * rel.y
            nearest = (float(body_x), float(body_y), float(clearance), float(-rel.x), float(-rel.y))
        return nearest

    @staticmethod
    def _best_explicit_escape(evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
        allowed = []
        for item in evaluations:
            name = item.trajectory.name
            if not (
                name.startswith("local_escape:")
                or name.startswith("corridor_escape:")
                or name in {"side_left", "side_right", "back", "back_left", "back_right", "stop"}
            ):
                continue
            if item.safety.reason in {"hard_clearance_violation", "static_clearance_violation", "ttc_violation"}:
                continue
            if not item.safety.feasible and item.safety.min_clearance < -0.02:
                continue
            allowed.append(item)
        if not allowed:
            return None
        return min(
            allowed,
            key=lambda item: (
                0 if item.safety.feasible else 1,
                item.costs.get("reverse", 999.0),
                item.costs.get("progress", 999.0),
                item.costs.get("lateral", 999.0),
                -float(item.safety.min_static_clearance),
                -float(item.safety.min_clearance),
                item.score,
            ),
        )

    @staticmethod
    def _best_mpc_feasible_recovery(evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
        allowed = []
        for item in evaluations:
            metadata = item.trajectory.metadata or {}
            if metadata.get("planner") != "sampling_mpc":
                continue
            if bool(metadata.get("is_stop", False)):
                continue
            if item.trajectory.velocity.norm() <= 0.035:
                continue
            if not item.safety.feasible:
                continue
            if item.safety.reason in {"hard_clearance_violation", "static_clearance_violation", "ttc_violation"}:
                continue
            if item.safety.min_ttc < 0.80:
                continue
            if item.safety.min_clearance < 0.18 or item.safety.min_static_clearance < 0.18:
                continue
            reason = str(item.safety.reason)
            allowed_reason = (
                "mpc_feasible_motion" in reason
                or reason in {"emergency_margin_violation", "escaping_hard_zone", "rejoining_corridor"}
            )
            if not allowed_reason:
                continue
            allowed.append(item)
        if not allowed:
            return None
        return min(
            allowed,
            key=lambda item: (
                0 if str(item.safety.reason) == "escaping_hard_zone" else 1,
                float(item.trajectory.metadata.get("sample_cost", item.score)),
                item.costs.get("reverse", 999.0),
                item.costs.get("progress", 999.0),
                -float(item.safety.min_clearance),
                item.score,
            ),
        )

    @staticmethod
    def _best_escape(evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
        if not evaluations:
            return None
        return max(
            evaluations,
            key=lambda item: (
                item.safety.safe,
                item.safety.feasible,
                item.safety.min_clearance,
                -item.score,
            ),
        )
