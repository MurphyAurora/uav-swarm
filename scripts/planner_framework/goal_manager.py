#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local goal and waypoint manager for long-route/diagonal-route stability.

Waypoints here are task-level guide points, not an obstacle map.  They do not
break an unknown-environment setting because obstacle locations are still only
obtained online from perception.  The planner may be given a rough route such as
"go through these task points", while collision avoidance remains local and
sensor-driven.
"""

from typing import Iterable, List, Optional, Sequence

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3


class LocalGoalManager:
    """Convert final goals or manual task waypoints into short-horizon local goals.

    The planner should not be pulled directly by a far diagonal target.  This
    module gives the local planner a reachable subgoal and can later be replaced
    by a richer corridor/topology manager without touching the safety checker.
    """

    def __init__(self, config: Optional[PlannerConfig] = None, waypoints: Optional[Sequence[Vec3]] = None):
        self.config = config or PlannerConfig()
        self._last_local_goal: Optional[Vec3] = None
        self._waypoints: List[Vec3] = list(waypoints or [])
        self._active_waypoint_index = 0

    @staticmethod
    def parse_waypoints(raw: str) -> List[Vec3]:
        """Parse "x,y,z;x,y,z" or "x,y;x,y" waypoint strings.

        If z is omitted for a waypoint, z=0.0 is used.  The caller may also
        replace z later with the mission altitude.
        """
        if not raw:
            return []
        waypoints: List[Vec3] = []
        for chunk in raw.split(';'):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [item.strip() for item in chunk.split(',') if item.strip()]
            if len(parts) not in (2, 3):
                raise ValueError(f'Invalid waypoint "{chunk}"; expected x,y or x,y,z')
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2]) if len(parts) == 3 else 0.0
            waypoints.append(Vec3(x, y, z))
        return waypoints

    def set_waypoints(self, waypoints: Iterable[Vec3], reset: bool = True) -> None:
        self._waypoints = list(waypoints)
        if reset:
            self._active_waypoint_index = 0

    def has_waypoints(self) -> bool:
        return bool(self._waypoints)

    def active_waypoint_index(self) -> int:
        return int(self._active_waypoint_index)

    def active_waypoint(self, final_goal: Vec3) -> Vec3:
        if not self._waypoints:
            return final_goal
        idx = min(self._active_waypoint_index, len(self._waypoints) - 1)
        return self._waypoints[idx]

    def update_active_waypoint(self, state: PlannerState, final_goal: Vec3) -> Vec3:
        """Advance waypoint index when the active waypoint has been reached."""
        if not self._waypoints:
            return final_goal
        while self._active_waypoint_index < len(self._waypoints) - 1:
            active = self._waypoints[self._active_waypoint_index]
            if state.position.distance_to(active) > self.config.local_goal_reached_radius:
                break
            self._active_waypoint_index += 1
        return self.active_waypoint(final_goal)

    def compute_local_goal(
        self,
        state: PlannerState,
        final_goal: Vec3,
        perception: Optional[PerceptionData] = None,
    ) -> Vec3:
        active_goal = self.update_active_waypoint(state, final_goal)
        local_goal = self._short_horizon_goal(state, active_goal)

        if perception and perception.obstacles:
            local_goal = self._nudge_goal_away_from_blocked_direction(state, local_goal, perception)

        self._last_local_goal = local_goal
        return local_goal

    def reached(self, state: PlannerState, local_goal: Vec3) -> bool:
        return state.position.distance_to(local_goal) <= self.config.local_goal_reached_radius

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

        # Keep the desired altitude trend but avoid huge vertical jumps in one
        # local planning cycle.
        dz = active_goal.z - state.position.z
        max_dz = max(0.5, self.config.vertical_speed * self.config.horizon)
        if abs(dz) > max_dz:
            local_goal = Vec3(
                local_goal.x,
                local_goal.y,
                state.position.z + max_dz * (1.0 if dz > 0.0 else -1.0),
            )
        return local_goal

    def _nudge_goal_away_from_blocked_direction(
        self,
        state: PlannerState,
        local_goal: Vec3,
        perception: PerceptionData,
    ) -> Vec3:
        """Lightweight obstacle-aware local-goal shift.

        If the straight segment toward the local goal is close to an obstacle,
        shift the local goal laterally.  This is intentionally simple and acts
        as a bridge before a full corridor/topology layer is added.
        """
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
                # Positive side means obstacle lies on left; shift right.
                shift_score += -1.0 if side > 0.0 else 1.0

        if abs(shift_score) < 1.0e-6:
            return local_goal
        shift = lateral * (0.8 * (1.0 if shift_score > 0.0 else -1.0))
        return local_goal + shift
