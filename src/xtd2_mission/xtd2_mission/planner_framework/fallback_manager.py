#!/usr/bin/env python3
"""Fallback commands used when no candidate is safely executable."""

import math
from typing import Optional, Sequence, Tuple

from .common import CandidateEvaluation, PerceptionData, PlannerCommand, PlannerConfig, PlannerState, Vec3


class FallbackManager:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._escape_side = 0.0

    def should_fallback(self, best: Optional[CandidateEvaluation], evaluations: Sequence[CandidateEvaluation], perception: PerceptionData) -> bool:
        if best is None:
            return True
        if not best.safety.feasible:
            return True
        return not any(item.safety.safe for item in evaluations)

    def command(self, state: PlannerState, perception: PerceptionData, evaluations: Sequence[CandidateEvaluation]) -> PlannerCommand:
        """Fallback used when the normal planner has no feasible command.

        Pure hover is safe, but it creates a deadlock in the offline single-pillar
        smoke test: the vehicle stops before the obstacle, sees the same invalid
        forward candidate forever, and never gives local A* a new viewpoint.  This
        fallback therefore allows a low-speed lateral step when there is still
        enough physical margin.  Backward motion is only added when the obstacle is
        closer; when clearance is small, hover remains the hard stop.
        """
        best = self._best_escape(evaluations)
        if perception.near_field_danger:
            self._escape_side = 0.0
            return PlannerCommand(Vec3(), "fallback_hover", "hover", "near_field_lidar_stop")

        fallback = self._lateral_unstick_command(state, perception, best)
        if fallback is not None:
            reason = best.safety.reason if best is not None else "no_safe_candidate"
            return PlannerCommand(fallback, "fallback_escape", "fallback_lateral_unstick", f"fallback_unstick:{reason}")

        self._escape_side = 0.0
        if best is not None:
            return PlannerCommand(Vec3(), "fallback_hover", "hover", f"fallback_stop:{best.safety.reason}")
        return PlannerCommand(Vec3(), "fallback_hover", "hover", "no_safe_candidate")

    def _lateral_unstick_command(
        self,
        state: PlannerState,
        perception: PerceptionData,
        best: Optional[CandidateEvaluation],
    ) -> Optional[Vec3]:
        nearest = self._nearest_obstacle_body(state, perception)
        if nearest is None:
            return None
        obs_x, obs_y, clearance = nearest

        # Only unstick when the obstacle is in front and there is still enough
        # physical room.  If we are near contact, hover wins.
        if not (0.25 <= obs_x <= 4.5 and abs(obs_y) <= 1.8):
            return None
        if clearance < 0.50:
            return None
        if best is not None and best.safety.min_clearance < -0.05:
            return None

        # Keep a side latch so repeated fallback calls do not alternate left/right.
        if abs(self._escape_side) < 0.5:
            if abs(obs_y) < 0.10:
                self._escape_side = 1.0
            else:
                # If obstacle appears on the left side of the body frame, move right.
                self._escape_side = -1.0 if obs_y > 0.0 else 1.0

        ch = math.cos(state.heading)
        sh = math.sin(state.heading)
        forward = Vec3(ch, sh, 0.0)
        left = Vec3(-sh, ch, 0.0)
        lateral = left * self._escape_side

        # Distance-based unstick:
        # - far from the obstacle: pure lateral motion to change the viewpoint;
        # - medium distance: lateral plus a tiny backward component;
        # - close: no command, handled by the hover branch above.
        if clearance > 0.90:
            lateral_speed = min(0.20, max(0.14, self.config.max_speed * 0.28))
            back_speed = 0.0
        else:
            lateral_speed = min(0.18, max(0.12, self.config.max_speed * 0.24))
            back_speed = min(0.025, max(0.0, self.config.max_speed * 0.035))
        return lateral * lateral_speed - forward * back_speed

    def _nearest_obstacle_body(self, state: PlannerState, perception: PerceptionData) -> Optional[Tuple[float, float, float]]:
        nearest = None
        nearest_dist = 999.0
        ch = math.cos(state.heading)
        sh = math.sin(state.heading)
        for obs in perception.obstacles:
            rel = obs.position_at(0.0) - state.position
            rel_xy = Vec3(rel.x, rel.y, 0.0)
            dist = rel_xy.norm()
            if dist <= 1.0e-6 or dist >= nearest_dist:
                continue
            nearest_dist = dist
            body_x = ch * rel.x + sh * rel.y
            body_y = -sh * rel.x + ch * rel.y
            clearance = dist - (self.config.drone_radius + float(obs.radius))
            nearest = (float(body_x), float(body_y), float(clearance))
        return nearest

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
