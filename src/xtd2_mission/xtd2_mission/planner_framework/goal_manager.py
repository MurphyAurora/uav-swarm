#!/usr/bin/env python3
"""Local goal and waypoint manager."""

import time
from typing import Iterable, List, Optional, Sequence

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3


class LocalGoalManager:
    def __init__(self, config: Optional[PlannerConfig] = None, waypoints: Optional[Sequence[Vec3]] = None):
        self.config = config or PlannerConfig()
        self._waypoints: List[Vec3] = list(waypoints or [])
        self._active_waypoint_index = 0
        self._corridor_side = 0.0
        self._corridor_until = 0.0
        self._last_reason = "direct"

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
        """Return a short-horizon goal aligned with the mission direction.

        For the local-A* bring-up, the goal manager must not also decide left vs
        right obstacle bypass.  That produced conflicts such as corridor_right
        from this module while local_astar latched left.  The final/waypoint goal
        now only provides the global progress direction; the frontend owns local
        obstacle bypassing.
        """
        active_goal = self.update_active_waypoint(state, final_goal)
        local_goal = self._short_horizon_goal(state, active_goal)
        self._corridor_side = 0.0
        self._corridor_until = 0.0
        if perception and perception.obstacles:
            self._last_reason = "direct_astar"
        else:
            self._last_reason = "direct"
        return local_goal

    def diagnostics(self) -> dict:
        return {
            "goal_reason": self._last_reason,
            "corridor_side": float(self._corridor_side),
            "corridor_left_sec": max(0.0, self._corridor_until - time.time()),
        }

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
