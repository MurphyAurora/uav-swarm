#!/usr/bin/env python3
"""Local goal and waypoint manager."""

from typing import Iterable, List, Optional, Sequence

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3


class LocalGoalManager:
    def __init__(self, config: Optional[PlannerConfig] = None, waypoints: Optional[Sequence[Vec3]] = None):
        self.config = config or PlannerConfig()
        self._waypoints: List[Vec3] = list(waypoints or [])
        self._active_waypoint_index = 0

    @staticmethod
    def parse_waypoints(raw: str) -> List[Vec3]:
        if not raw:
            return []
        waypoints = []
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [item.strip() for item in chunk.split(",") if item.strip()]
            if len(parts) not in (2, 3):
                raise ValueError(f'Invalid waypoint "{chunk}"; expected x,y or x,y,z')
            waypoints.append(Vec3(float(parts[0]), float(parts[1]), float(parts[2]) if len(parts) == 3 else 0.0))
        return waypoints

    def set_waypoints(self, waypoints: Iterable[Vec3], reset: bool = True) -> None:
        self._waypoints = list(waypoints)
        if reset:
            self._active_waypoint_index = 0

    def active_waypoint_index(self) -> int:
        return int(self._active_waypoint_index)

    def active_waypoint(self, final_goal: Vec3) -> Vec3:
        if not self._waypoints:
            return final_goal
        return self._waypoints[min(self._active_waypoint_index, len(self._waypoints) - 1)]

    def update_active_waypoint(self, state: PlannerState, final_goal: Vec3) -> Vec3:
        if not self._waypoints:
            return final_goal
        while self._active_waypoint_index < len(self._waypoints) - 1:
            if state.position.distance_to(self._waypoints[self._active_waypoint_index]) > self.config.local_goal_reached_radius:
                break
            self._active_waypoint_index += 1
        return self.active_waypoint(final_goal)

    def compute_local_goal(self, state: PlannerState, final_goal: Vec3, perception: Optional[PerceptionData] = None) -> Vec3:
        active_goal = self.update_active_waypoint(state, final_goal)
        local_goal = self._short_horizon_goal(state, active_goal)
        if perception and perception.obstacles:
            local_goal = self._nudge_goal_away_from_blocked_direction(state, local_goal, perception)
        return local_goal

    def mission_reached(self, state: PlannerState, final_goal: Vec3) -> bool:
        if self._waypoints and self._active_waypoint_index < len(self._waypoints) - 1:
            return False
        return state.position.distance_to(final_goal) <= self.config.local_goal_reached_radius

    def _short_horizon_goal(self, state: PlannerState, active_goal: Vec3) -> Vec3:
        to_goal = active_goal - state.position
        dist = to_goal.norm()
        if dist <= self.config.local_goal_distance:
            return active_goal
        direction = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        local_goal = state.position + direction * self.config.local_goal_distance
        dz = active_goal.z - state.position.z
        max_dz = max(0.5, self.config.vertical_speed * self.config.horizon)
        if abs(dz) > max_dz:
            local_goal = Vec3(local_goal.x, local_goal.y, state.position.z + max_dz * (1.0 if dz > 0.0 else -1.0))
        return local_goal

    def _nudge_goal_away_from_blocked_direction(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData) -> Vec3:
        forward = (local_goal - state.position).normalized(Vec3(1.0, 0.0, 0.0))
        lateral = Vec3(-forward.y, forward.x, 0.0).normalized(Vec3(0.0, 1.0, 0.0))
        shift_score = 0.0
        for obs in perception.obstacles:
            rel = obs.position - state.position
            along = rel.x * forward.x + rel.y * forward.y + rel.z * forward.z
            if along < 0.0 or along > self.config.local_goal_distance:
                continue
            side = rel.x * lateral.x + rel.y * lateral.y
            clearance = abs(side) - (obs.radius + self.config.drone_radius + self.config.obstacle_margin)
            if clearance < 0.6:
                shift_score += -1.0 if side > 0.0 else 1.0
        if abs(shift_score) < 1.0e-6:
            return local_goal
        return local_goal + lateral * (0.8 * (1.0 if shift_score > 0.0 else -1.0))
