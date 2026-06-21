#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent safety checker for candidate local trajectories."""

import math
from typing import List, Optional

from .common import CandidateTrajectory, Obstacle, PerceptionData, PlannerConfig, PlannerState, SafetyReport, Vec3, safe_div


class SafetyChecker:
    """Reject unsafe trajectories before backend scoring can select them."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def evaluate(self, trajectory: CandidateTrajectory, state: PlannerState, perception: PerceptionData) -> SafetyReport:
        min_clearance = 999.0
        min_static_clearance = 999.0
        min_source = "none"
        min_ttc = 999.0

        for point in trajectory.points:
            for obs in perception.obstacles:
                obs_pos = obs.position_at(point.t)
                clearance = point.position.distance_to(obs_pos) - self._safety_radius(obs)
                if clearance < min_clearance:
                    min_clearance = clearance
                    min_source = obs.source
                if obs.source in ("static", "lidar_near_field"):
                    min_static_clearance = min(min_static_clearance, clearance)

                ttc = self._estimate_ttc(state.position, trajectory.velocity, obs, point.t)
                min_ttc = min(min_ttc, ttc)

        hard_ok = min_clearance >= self.config.hard_clearance
        static_ok = min_static_clearance >= self.config.static_hard_clearance
        near_ok = not perception.near_field_danger or min_static_clearance >= self.config.hard_clearance
        feasible = bool(hard_ok and static_ok and near_ok)
        safe = bool(
            feasible
            and min_clearance >= self.config.emergency_clearance
            and min_static_clearance >= self.config.static_emergency_clearance
        )

        reason = "ok"
        if not hard_ok:
            reason = "hard_clearance_violation"
        elif not static_ok:
            reason = "static_clearance_violation"
        elif not near_ok:
            reason = "near_field_lidar_danger"
        elif not safe:
            reason = "emergency_margin_violation"

        return SafetyReport(
            safe=safe,
            feasible=feasible,
            min_clearance=float(min_clearance),
            min_clearance_source=min_source,
            min_static_clearance=float(min_static_clearance),
            min_ttc=float(min_ttc),
            reason=reason,
        )

    def filter_safe(
        self,
        candidates: List[CandidateTrajectory],
        state: PlannerState,
        perception: PerceptionData,
    ) -> List[SafetyReport]:
        return [self.evaluate(candidate, state, perception) for candidate in candidates]

    def _safety_radius(self, obs: Obstacle) -> float:
        return self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin

    def _estimate_ttc(self, uav_position: Vec3, uav_velocity: Vec3, obs: Obstacle, horizon_t: float) -> float:
        obs_position = obs.position_at(0.0)
        rel_pos = obs_position - uav_position
        rel_vel = obs.velocity - uav_velocity
        closing_speed = -safe_div(
            rel_pos.x * rel_vel.x + rel_pos.y * rel_vel.y + rel_pos.z * rel_vel.z,
            max(rel_pos.norm(), 1.0e-6),
            fallback=0.0,
        )
        if closing_speed <= 1.0e-6:
            return 999.0
        return max(0.0, (rel_pos.norm() - self._safety_radius(obs)) / closing_speed)
