#!/usr/bin/env python3
"""Hard safety checks for candidate trajectories."""

from typing import List, Optional

from .common import CandidateTrajectory, Obstacle, PerceptionData, PlannerConfig, PlannerState, SafetyReport, Vec3, safe_div


class SafetyChecker:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def evaluate(self, trajectory: CandidateTrajectory, state: PlannerState, perception: PerceptionData) -> SafetyReport:
        min_clearance = 999.0
        min_static_clearance = 999.0
        min_source = "none"
        min_ttc = 999.0
        for point in trajectory.points:
            for obs in perception.obstacles:
                clearance = point.position.distance_to(obs.position_at(point.t)) - self._safety_radius(obs)
                if clearance < min_clearance:
                    min_clearance = clearance
                    min_source = obs.source
                if obs.source == "static":
                    min_static_clearance = min(min_static_clearance, clearance)
                min_ttc = min(min_ttc, self._estimate_ttc(state.position, trajectory.velocity, obs))

        emergency_margin = self._emergency_margin(min_source)
        static_emergency_margin = max(self.config.static_emergency_clearance, 0.8)
        hard_margin = self._hard_margin(min_source)
        static_hard_margin = min(self.config.static_hard_clearance, 0.45)

        hard_ok = min_clearance >= hard_margin
        static_ok = min_static_clearance >= static_hard_margin
        feasible = bool(hard_ok and static_ok)
        safe = bool(feasible and min_clearance >= emergency_margin and min_static_clearance >= static_emergency_margin)

        reason = "ok"
        if not hard_ok:
            reason = "hard_clearance_violation"
        elif not static_ok:
            reason = "static_clearance_violation"
        elif not safe:
            reason = "emergency_margin_violation"
        return SafetyReport(safe, feasible, float(min_clearance), min_source, float(min_static_clearance), float(min_ttc), reason)

    def filter_safe(self, candidates: List[CandidateTrajectory], state: PlannerState, perception: PerceptionData) -> List[SafetyReport]:
        return [self.evaluate(candidate, state, perception) for candidate in candidates]

    def _safety_radius(self, obs: Obstacle) -> float:
        if obs.source == "lidar_near_field":
            # LiDAR obstacles are already inflated in the local A* grid.  Keep a
            # smaller execution margin here so the SafetyChecker does not veto
            # every passable side gap after A* has found a route.
            return self.config.drone_radius + float(obs.radius) + 0.20
        return self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin

    def _hard_margin(self, min_source: str) -> float:
        if min_source == "lidar_near_field":
            return min(self.config.hard_clearance, 0.18)
        return self.config.hard_clearance

    def _emergency_margin(self, min_source: str) -> float:
        if min_source == "lidar_near_field":
            return max(min(self.config.emergency_clearance, 0.50), 0.38)
        return max(self.config.emergency_clearance, 0.60)

    def _estimate_ttc(self, uav_position: Vec3, uav_velocity: Vec3, obs: Obstacle) -> float:
        rel_pos = obs.position_at(0.0) - uav_position
        rel_vel = obs.velocity - uav_velocity
        closing_speed = -safe_div(rel_pos.x * rel_vel.x + rel_pos.y * rel_vel.y + rel_pos.z * rel_vel.z, max(rel_pos.norm(), 1.0e-6))
        if closing_speed <= 1.0e-6:
            return 999.0
        return max(0.0, (rel_pos.norm() - self._safety_radius(obs)) / closing_speed)
