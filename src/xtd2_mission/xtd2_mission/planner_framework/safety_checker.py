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
        current_clearance = 999.0
        current_static_clearance = 999.0
        future_min_clearance = 999.0
        future_min_static_clearance = 999.0
        final_clearance = 999.0
        final_static_clearance = 999.0
        for point in trajectory.points:
            for obs in perception.obstacles:
                clearance = point.position.distance_to(obs.position_at(point.t)) - self._safety_radius(obs)
                if clearance < min_clearance:
                    min_clearance = clearance
                    min_source = obs.source
                if obs.source == "static":
                    min_static_clearance = min(min_static_clearance, clearance)
                if point.t <= 1.0e-6:
                    current_clearance = min(current_clearance, clearance)
                    if obs.source == "static":
                        current_static_clearance = min(current_static_clearance, clearance)
                else:
                    future_min_clearance = min(future_min_clearance, clearance)
                    if obs.source == "static":
                        future_min_static_clearance = min(future_min_static_clearance, clearance)
                min_ttc = min(min_ttc, self._estimate_ttc(state.position, trajectory.velocity, obs))
        if trajectory.points:
            final_point = trajectory.points[-1]
            for obs in perception.obstacles:
                clearance = final_point.position.distance_to(obs.position_at(final_point.t)) - self._safety_radius(obs)
                final_clearance = min(final_clearance, clearance)
                if obs.source == "static":
                    final_static_clearance = min(final_static_clearance, clearance)

        emergency_margin = self._emergency_margin(min_source)
        static_emergency_margin = max(min(self.config.static_emergency_clearance, 0.55), 0.35)
        hard_margin = self._hard_margin(min_source)
        static_hard_margin = max(min(self.config.static_hard_clearance, 0.35), 0.28)

        hard_ok = min_clearance >= hard_margin
        static_ok = min_static_clearance >= static_hard_margin
        escape_ok = self._escaping_from_hard_zone(
            current_clearance,
            current_static_clearance,
            future_min_clearance,
            future_min_static_clearance,
            final_clearance,
            final_static_clearance,
            hard_margin,
            static_hard_margin,
        )
        rejoin_ok = self._forward_rejoin_ok(trajectory, min_clearance, min_static_clearance)
        if escape_ok or rejoin_ok:
            hard_ok = True
            static_ok = True

        # TTC constraint: if the commanded velocity is closing on an obstacle and
        # predicted contact is imminent, the trajectory is not feasible even if
        # point-sampled clearance has not crossed the hard margin yet.  This is
        # intentionally conservative and only activates in the near shell, so it
        # does not veto wide tangential bypasses.
        ttc_violation = self._ttc_hard_violation(trajectory, min_ttc, min_clearance, min_static_clearance)
        if ttc_violation:
            hard_ok = False
            static_ok = False
            escape_ok = False
            rejoin_ok = False

        feasible = bool(hard_ok and static_ok)
        safe = bool(feasible and min_clearance >= emergency_margin and min_static_clearance >= static_emergency_margin)

        reason = "ok"
        if ttc_violation:
            reason = "ttc_violation"
        elif rejoin_ok and not safe:
            reason = "rejoining_corridor"
        elif escape_ok and not safe:
            reason = "escaping_hard_zone"
        elif not hard_ok:
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
            return self.config.drone_radius + float(obs.radius) + 0.20
        if obs.source == "static":
            return self.config.drone_radius + float(obs.radius) + min(self.config.obstacle_margin, 0.10)
        return self.config.drone_radius + float(obs.radius) + self.config.obstacle_margin

    def _hard_margin(self, min_source: str) -> float:
        if min_source == "lidar_near_field":
            return min(self.config.hard_clearance, 0.18)
        if min_source == "static":
            return max(min(self.config.static_hard_clearance, 0.35), 0.28)
        return self.config.hard_clearance

    def _emergency_margin(self, min_source: str) -> float:
        if min_source == "lidar_near_field":
            return max(min(self.config.emergency_clearance, 0.50), 0.38)
        if min_source == "static":
            return max(min(self.config.static_emergency_clearance, 0.55), 0.35)
        return max(self.config.emergency_clearance, 0.60)

    @staticmethod
    def _escaping_from_hard_zone(
        current_clearance: float,
        current_static_clearance: float,
        future_min_clearance: float,
        future_min_static_clearance: float,
        final_clearance: float,
        final_static_clearance: float,
        hard_margin: float,
        static_hard_margin: float,
    ) -> bool:
        in_dynamic_hard_zone = current_clearance < hard_margin
        in_static_hard_zone = current_static_clearance < static_hard_margin
        if not in_dynamic_hard_zone and not in_static_hard_zone:
            return False

        current = min(current_clearance, current_static_clearance)
        future_min = min(future_min_clearance, future_min_static_clearance)
        final = min(final_clearance, final_static_clearance)
        if current >= 998.0 or future_min >= 998.0 or final >= 998.0:
            return False

        contact_floor = -0.05
        if future_min < contact_floor or final < contact_floor:
            return False
        if future_min < current - 0.005:
            return False
        return final >= current + 0.08

    def _forward_rejoin_ok(self, trajectory: CandidateTrajectory, min_clearance: float, min_static_clearance: float) -> bool:
        if not trajectory.name.startswith("local_astar:"):
            return False
        meta = trajectory.metadata or {}
        if bool(meta.get("front_blocked", False)) or bool(meta.get("corridor_blocked", False)):
            return False
        path_clearance = float(meta.get("path_clearance", meta.get("min_path_clearance", 0.0)) or 0.0)
        if path_clearance < max(0.70, self.config.drone_radius + 0.20):
            return False
        body_velocity = meta.get("body_velocity") or {}
        body_x = float(body_velocity.get("x", 0.0) or 0.0)
        if body_x < 0.04:
            return False
        return min_clearance >= 0.05 and min_static_clearance >= 0.05

    @staticmethod
    def _ttc_hard_violation(trajectory: CandidateTrajectory, min_ttc: float, min_clearance: float, min_static_clearance: float) -> bool:
        if trajectory.velocity.norm() <= 1.0e-6:
            return False
        if min_ttc >= 0.80:
            return False
        near_clearance = min(min_clearance, min_static_clearance)
        return near_clearance < 0.45

    def _estimate_ttc(self, uav_position: Vec3, uav_velocity: Vec3, obs: Obstacle) -> float:
        rel_pos = obs.position_at(0.0) - uav_position
        rel_vel = obs.velocity - uav_velocity
        closing_speed = -safe_div(rel_pos.x * rel_vel.x + rel_pos.y * rel_vel.y + rel_pos.z * rel_vel.z, max(rel_pos.norm(), 1.0e-6))
        if closing_speed <= 1.0e-6:
            return 999.0
        return max(0.0, (rel_pos.norm() - self._safety_radius(obs)) / closing_speed)
