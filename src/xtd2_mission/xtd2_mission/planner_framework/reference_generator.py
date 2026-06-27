#!/usr/bin/env python3
"""Reference generation interfaces for local planner frontends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3
from .local_astar_passable import LocalAStarPlanner


@dataclass
class LocalReference:
    source: str
    points: List[Vec3] = field(default_factory=list)
    direction: Vec3 = field(default_factory=lambda: Vec3(1.0, 0.0, 0.0))
    speed: float = 0.0
    confidence: float = 0.0
    side_hint: float = 0.0
    reason: str = ""


class ReferenceGenerator(Protocol):
    def generate(self, state: PlannerState, local_goal: Vec3, perception: Optional[PerceptionData] = None) -> LocalReference:
        ...


class LocalAStarReferenceGenerator:
    """Generate Sampling-MPC references from Local A* without coupling MPC to A*."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.local_astar = LocalAStarPlanner(self.config)
        self.diagnostics = {}

    def generate(self, state: PlannerState, local_goal: Vec3, perception: Optional[PerceptionData] = None) -> LocalReference:
        obstacles = perception.obstacles if perception is not None else []
        candidate = self.local_astar.plan_candidate(state, local_goal, obstacles)
        if candidate is None:
            reference = self._goal_direction_reference(state, local_goal, "local_astar_failed")
            self.diagnostics = {
                "source": reference.source,
                "reason": reference.reason,
                "local_astar": dict(getattr(self.local_astar, "diagnostics", {})),
            }
            return reference

        points = [point.position for point in candidate.points]
        direction = self._reference_direction(state, points, candidate.velocity, local_goal)
        speed = min(candidate.velocity.norm(), float(getattr(self.config, "max_speed", 0.75)))
        confidence = self._confidence(candidate)
        side_hint = self._side_hint(candidate, direction, state, local_goal)
        reason = str(candidate.metadata.get("replan_reason", candidate.metadata.get("reason", "local_astar")) or "local_astar")
        reference = LocalReference(
            source="local_astar",
            points=points,
            direction=direction,
            speed=float(speed),
            confidence=float(confidence),
            side_hint=float(side_hint),
            reason=reason,
        )
        self.diagnostics = {
            "source": reference.source,
            "reason": reference.reason,
            "confidence": reference.confidence,
            "side_hint": reference.side_hint,
            "candidate": candidate.name,
            "local_astar": dict(getattr(self.local_astar, "diagnostics", {})),
        }
        return reference

    def _goal_direction_reference(self, state: PlannerState, local_goal: Vec3, reason: str) -> LocalReference:
        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        distance = to_goal.norm()
        direction = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        cruise = min(float(getattr(self.config, "cruise_speed", 0.55)), float(getattr(self.config, "max_speed", 0.75)))
        speed = min(cruise, max(0.08, distance / max(float(getattr(self.config, "horizon", 2.5)), 1.0e-6))) if distance > 1.0e-6 else 0.0
        return LocalReference(
            source="goal_direction_fallback",
            points=[state.position, local_goal],
            direction=direction,
            speed=float(speed),
            confidence=0.35,
            side_hint=0.0,
            reason=reason,
        )

    def _reference_direction(self, state: PlannerState, points: List[Vec3], velocity: Vec3, local_goal: Vec3) -> Vec3:
        if len(points) >= 2:
            for point in points[1:]:
                delta = Vec3(point.x - state.position.x, point.y - state.position.y, 0.0)
                if delta.norm() > 0.08:
                    return delta.normalized(Vec3(1.0, 0.0, 0.0))
        vel_xy = Vec3(velocity.x, velocity.y, 0.0)
        if vel_xy.norm() > 1.0e-6:
            return vel_xy.normalized(Vec3(1.0, 0.0, 0.0))
        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        return to_goal.normalized(Vec3(1.0, 0.0, 0.0))

    def _confidence(self, candidate) -> float:
        margin = float(candidate.metadata.get("min_path_clearance", 0.0) or 0.0)
        if candidate.name == "local_astar:direct":
            base = 0.82
        else:
            base = 0.70
        return max(0.15, min(1.0, base + 0.20 * max(0.0, min(1.0, margin))))

    def _side_hint(self, candidate, direction: Vec3, state: PlannerState, local_goal: Vec3) -> float:
        metadata = candidate.metadata or {}
        side = float(metadata.get("side_latch", metadata.get("bypass_side", 0.0)) or 0.0)
        if abs(side) > 0.5:
            return 1.0 if side > 0.0 else -1.0
        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        goal_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        left = Vec3(-goal_dir.y, goal_dir.x, 0.0)
        lateral = direction.x * left.x + direction.y * left.y
        if abs(lateral) <= 0.08:
            return 0.0
        return 1.0 if lateral > 0.0 else -1.0
