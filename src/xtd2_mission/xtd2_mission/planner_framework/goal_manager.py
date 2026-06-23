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
        active_goal = self.update_active_waypoint(state, final_goal)
        local_goal = self._short_horizon_goal(state, active_goal)
        if perception and perception.obstacles:
            corridor_goal = self._corridor_goal(state, local_goal, perception)
            if corridor_goal is not None:
                return corridor_goal
            if self.config.local_goal_nudge_enable:
                local_goal = self._nudge_goal_away_from_blocked_direction(state, local_goal, perception)
                self._last_reason = "nudge"
            else:
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

    def _nudge_goal_away_from_blocked_direction(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData) -> Vec3:
        forward = (local_goal - state.position).normalized(Vec3(1.0, 0.0, 0.0))
        lateral = Vec3(-forward.y, forward.x, 0.0).normalized(Vec3(0.0, 1.0, 0.0))
        best_shift = 0.0
        best_clearance = None
        for obs in perception.obstacles:
            rel = obs.position - state.position
            along = rel.x * forward.x + rel.y * forward.y + rel.z * forward.z
            if along < 0.0 or along > self.config.local_goal_distance:
                continue
            if obs.source == "static" and along > 1.5:
                continue
            side = rel.x * lateral.x + rel.y * lateral.y
            clearance = abs(side) - (obs.radius + self.config.drone_radius + self.config.obstacle_margin)
            if clearance >= 0.35:
                continue
            if best_clearance is None or clearance < best_clearance:
                best_clearance = clearance
                best_shift = -1.0 if side > 0.0 else 1.0
        if abs(best_shift) < 1.0e-6:
            self._last_reason = "direct"
            return local_goal
        shift_mag = 0.45 if best_clearance is not None and best_clearance >= 0.0 else 0.7
        return local_goal + lateral * (shift_mag * best_shift)

    def _corridor_goal(self, state: PlannerState, local_goal: Vec3, perception: PerceptionData) -> Optional[Vec3]:
        if not self.config.corridor_enable:
            return None
        to_goal = local_goal - state.position
        forward = Vec3(to_goal.x, to_goal.y, 0.0).normalized(Vec3(1.0, 0.0, 0.0))
        lateral = Vec3(-forward.y, forward.x, 0.0).normalized(Vec3(0.0, 1.0, 0.0))
        if forward.norm() <= 1.0e-6:
            return None

        lookahead = max(self.config.local_goal_distance, self.config.corridor_lookahead)
        half_width = max(
            self.config.corridor_half_width,
            self.config.drone_radius + self.config.obstacle_margin + self.config.astar_latch_clearance,
        )
        blockers = []
        left_nearest = 999.0
        right_nearest = 999.0
        for obs in perception.obstacles:
            if obs.source not in ("static", "lidar_near_field"):
                continue
            rel = obs.position - state.position
            along = rel.x * forward.x + rel.y * forward.y
            if along < 0.25 or along > lookahead:
                continue
            side = rel.x * lateral.x + rel.y * lateral.y
            radius = float(obs.radius) + self.config.drone_radius + self.config.obstacle_margin
            if side >= 0.0:
                left_nearest = min(left_nearest, max(0.0, abs(side) - radius))
            else:
                right_nearest = min(right_nearest, max(0.0, abs(side) - radius))
            if abs(side) <= half_width + radius:
                blockers.append((along, side, radius))
        if not blockers:
            if time.time() > self._corridor_until:
                self._corridor_side = 0.0
            return None

        blockers.sort(key=lambda item: item[0])
        first_along = blockers[0][0]
        now = time.time()
        preferred_side = 1.0 if left_nearest >= right_nearest else -1.0
        if now < self._corridor_until and abs(self._corridor_side) > 0.5:
            current_clear = left_nearest if self._corridor_side > 0.0 else right_nearest
            other_clear = right_nearest if self._corridor_side > 0.0 else left_nearest
            if other_clear <= current_clear + 0.55:
                preferred_side = self._corridor_side
        self._corridor_side = preferred_side
        self._corridor_until = now + self.config.corridor_latch_sec

        forward_dist = min(
            self.config.local_goal_distance + 1.5,
            max(1.6, first_along + self.config.corridor_forward_offset),
        )
        side_offset = max(self.config.corridor_side_offset, half_width + 0.55)
        dz = local_goal.z - state.position.z
        corridor = state.position + forward * forward_dist + lateral * (preferred_side * side_offset)
        self._last_reason = "corridor_left" if preferred_side > 0.0 else "corridor_right"
        return Vec3(corridor.x, corridor.y, state.position.z + dz)
