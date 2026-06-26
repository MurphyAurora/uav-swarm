#!/usr/bin/env python3
"""Lightweight corridor/mode selector for local planning.

This module is intentionally simple: it does not replace trajectory optimization
or local A*, but provides a short-horizon side/corridor preference so the
frontend does not make an independent left/right decision every frame.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3


@dataclass
class CorridorDecision:
    active: bool
    side: float
    locked: bool
    until: float
    left_cost: float
    right_cost: float
    reason: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "active": bool(self.active),
            "side": float(self.side),
            "locked": bool(self.locked),
            "until": float(self.until),
            "left_cost": float(self.left_cost),
            "right_cost": float(self.right_cost),
            "reason": self.reason,
        }


class CorridorSelector:
    """Short-term left/right corridor selector.

    A reactive local planner can oscillate in constrained passages because small
    changes in nearest-obstacle geometry flip the locally best side.  This class
    evaluates two lateral corridors and latches the preferred side for a short
    duration unless the locked side becomes much worse than the alternative.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._locked_side = 0.0
        self._lock_until = 0.0
        self._last_decision = CorridorDecision(False, 0.0, False, 0.0, 0.0, 0.0, "init")

    def update(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData, risk: Optional[dict] = None) -> CorridorDecision:
        left_cost, right_cost, relevant = self._score_sides(state, local_goal, perception)
        now = float(state.stamp)
        risk_value = float((risk or {}).get("risk", 0.0) or 0.0)
        front_obstacle = bool((risk or {}).get("front_obstacle", False))

        if not relevant and risk_value < 0.25:
            if now > self._lock_until:
                self._locked_side = 0.0
            self._last_decision = CorridorDecision(False, self._locked_side, now <= self._lock_until, self._lock_until, left_cost, right_cost, "low_relevance")
            return self._last_decision

        preferred = 1.0 if left_cost <= right_cost else -1.0
        best = min(left_cost, right_cost)
        worst = max(left_cost, right_cost)
        score_gap = abs(left_cost - right_cost)

        locked = abs(self._locked_side) > 0.5 and now <= self._lock_until
        reason = "new_lock"
        if locked:
            locked_cost = left_cost if self._locked_side > 0.0 else right_cost
            other_cost = right_cost if self._locked_side > 0.0 else left_cost
            # Switch only when the current side is clearly worse.  This hysteresis
            # prevents side flipping in S-shaped or bounded maps.
            if other_cost + 1.20 < locked_cost and risk_value > 0.55:
                self._locked_side = -self._locked_side
                self._lock_until = now + 1.8
                reason = "switch_locked_side"
            else:
                preferred = self._locked_side
                reason = "keep_locked_side"
        else:
            # If both sides are nearly equivalent and there was a previous side,
            # keep it to reduce dithering.
            if score_gap < 0.35 and abs(self._locked_side) > 0.5:
                preferred = self._locked_side
                reason = "reuse_previous_side"
            self._locked_side = preferred
            self._lock_until = now + (2.2 if front_obstacle or risk_value > 0.35 else 1.2)

        self._last_decision = CorridorDecision(
            active=True,
            side=float(self._locked_side),
            locked=now <= self._lock_until,
            until=float(self._lock_until),
            left_cost=float(left_cost),
            right_cost=float(right_cost),
            reason=reason,
        )
        return self._last_decision

    def reset(self) -> None:
        self._locked_side = 0.0
        self._lock_until = 0.0
        self._last_decision = CorridorDecision(False, 0.0, False, 0.0, 0.0, 0.0, "reset")

    def _score_sides(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData) -> Tuple[float, float, bool]:
        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        goal_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        lateral_left = Vec3(-goal_dir.y, goal_dir.x, 0.0)

        lookahead = max(3.0, min(5.5, float(self.config.local_goal_distance) + 2.0))
        side_offset = max(0.75, min(1.35, float(self.config.drone_radius) + 0.85))
        corridor_width = max(0.55, float(self.config.drone_radius) + 0.45)
        relevant = False
        left_cost = 0.0
        right_cost = 0.0

        for obs in perception.obstacles:
            obs_pos = obs.position_at(0.0)
            rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)
            forward = rel.x * goal_dir.x + rel.y * goal_dir.y
            lateral = rel.x * lateral_left.x + rel.y * lateral_left.y
            if forward < -0.4 or forward > lookahead:
                continue

            radius = self._execution_radius(obs)
            # Strongly consider obstacles close to the forward tube.  Boundary
            # obstacles are also included; they make the outside corridor expensive.
            if abs(lateral) <= 2.4 or obs.source == "boundary":
                relevant = True

            forward_weight = 1.0 + max(0.0, (lookahead - forward) / lookahead)
            left_clear = abs(lateral - side_offset) - radius
            right_clear = abs(lateral + side_offset) - radius
            left_cost += forward_weight * self._clearance_penalty(left_clear, corridor_width)
            right_cost += forward_weight * self._clearance_penalty(right_clear, corridor_width)

            # Penalize choosing the same side as a close obstacle in front.
            if 0.0 <= forward <= 3.0 and abs(lateral) <= 1.25:
                if lateral > 0.0:
                    left_cost += 0.35
                elif lateral < 0.0:
                    right_cost += 0.35
                else:
                    left_cost += 0.15
                    right_cost += 0.15

        return left_cost, right_cost, relevant

    def _execution_radius(self, obs) -> float:
        if obs.source == "lidar_near_field":
            return self.config.drone_radius + float(obs.radius) + 0.20
        if obs.source == "static":
            return self.config.drone_radius + float(obs.radius) + min(self.config.obstacle_margin, 0.10)
        return self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin

    @staticmethod
    def _clearance_penalty(clearance: float, width: float) -> float:
        if clearance >= width:
            return 0.0
        if clearance <= 0.0:
            return 2.5 + abs(clearance) * 4.0
        ratio = (width - clearance) / max(width, 1.0e-6)
        return ratio * ratio
