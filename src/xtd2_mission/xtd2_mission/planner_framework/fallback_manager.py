#!/usr/bin/env python3
"""Fallback commands used when no candidate is safely executable."""

import math
import time
from typing import Optional, Sequence

from .common import CandidateEvaluation, PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3


class FallbackManager:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._latched_escape_until = 0.0
        self._latched_escape_velocity = Vec3()
        self._latched_escape_reason = ""

    def should_fallback(self, best: Optional[CandidateEvaluation], evaluations: Sequence[CandidateEvaluation], perception: PerceptionData) -> bool:
        if best is None:
            return True
        if perception.near_field_danger and best.safety.min_static_clearance < self.config.emergency_clearance:
            return True
        if not best.safety.feasible:
            return True
        return not any(item.safety.safe for item in evaluations)

    def command(self, state: PlannerState, perception: PerceptionData, evaluations: Sequence[CandidateEvaluation]) -> PlannerCommand:
        """Stop-first fallback.

        The debug log showed many frames like
        ``mode=fallback_escape traj=local_astar:* reason=emergency_margin_violation``.
        That means the A* candidate was not safe, but fallback still reused its
        velocity. This feeds the drone around the obstacle in alternating
        directions and creates the observed circular motion. From now on,
        local_astar candidates are never executed by fallback unless they were
        already safe enough to be selected by the normal planner path.
        """
        near_escape = self._nearest_obstacle_escape(state, perception)
        if near_escape is not None:
            return near_escape

        best = self._best_escape(evaluations)
        if best is not None and best.trajectory.name.startswith("local_astar:"):
            return PlannerCommand(
                Vec3(),
                "fallback_hover",
                "hover",
                f"stop_first_blocked_astar:{best.safety.reason}",
            )

        if perception.near_field_danger:
            return PlannerCommand(Vec3(), "fallback_hover", "hover", "near_field_lidar_danger_stop_first")
        return PlannerCommand(Vec3(), "fallback_hover", "hover", "no_safe_candidate")

    def _nearest_obstacle_escape(self, state: PlannerState, perception: PerceptionData) -> Optional[PlannerCommand]:
        """Only use moving escape for true near-contact LiDAR danger.

        Soft emergency-margin violations should stop instead of drifting around
        the obstacle. Moving escape is reserved for very close points where a
        pure hover may keep the vehicle inside the hard safety shell.
        """
        if not perception.near_field_danger:
            self._latched_escape_until = 0.0
            return None

        now = time.time()
        if now < self._latched_escape_until and self._latched_escape_velocity.norm() > 0.05:
            return PlannerCommand(
                self._latched_escape_velocity,
                "fallback_escape",
                "latched_lidar_escape",
                self._latched_escape_reason,
            )

        close_count = 0
        pressure = Vec3()
        nearest_clearance = 999.0
        c = math.cos(state.heading)
        s = math.sin(state.heading)
        for obs in perception.obstacles:
            if obs.source != "lidar_near_field":
                continue
            clearance = state.position.distance_to(obs.position) - float(obs.radius) - self.config.drone_radius
            rel_x = obs.position.x - state.position.x
            rel_y = obs.position.y - state.position.y
            body_x = c * rel_x + s * rel_y
            body_y = -s * rel_x + c * rel_y
            hard_contact = clearance <= max(0.05, 0.5 * self.config.hard_clearance)
            if not hard_contact:
                continue
            away = Vec3(
                state.position.x - obs.position.x,
                state.position.y - obs.position.y,
                0.0,
            )
            distance = max(away.norm(), 1.0e-3)
            weight = 1.0 / max(0.12, clearance + 0.35)
            pressure = pressure + away * (weight / distance)
            close_count += 1
            if clearance < nearest_clearance:
                nearest_clearance = clearance
        if close_count <= 0:
            return None

        fallback_dir = self._latched_escape_velocity.normalized(Vec3(-1.0, 0.0, 0.0))
        away = pressure.normalized(fallback_dir)
        if self._latched_escape_velocity.norm() > 0.05:
            prev = self._latched_escape_velocity.normalized()
            if away.x * prev.x + away.y * prev.y < 0.25:
                away = prev
        speed = min(0.25, self.config.max_speed)
        velocity = away * speed
        reason = f"hard_contact_escape_clearance={nearest_clearance:.2f}, close_points={close_count}"
        self._latched_escape_velocity = velocity
        self._latched_escape_reason = reason
        self._latched_escape_until = now + 0.8
        return PlannerCommand(
            velocity,
            "fallback_escape",
            "latched_lidar_escape",
            reason,
        )

    def _best_escape(self, evaluations: Sequence[CandidateEvaluation]) -> Optional[CandidateEvaluation]:
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
