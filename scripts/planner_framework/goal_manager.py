#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local goal manager for long-route and diagonal-route stability."""

from typing import Optional

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3


class LocalGoalManager:
    """Convert a far final goal into a short-horizon local goal.

    The planner should not be pulled directly by a far diagonal target.  This
    module gives the local planner a reachable subgoal and can later be replaced
    by a richer waypoint/corridor manager without touching the safety checker.
    """

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._last_local_goal: Optional[Vec3] = None

    def compute_local_goal(
        self,
        state: PlannerState,
        final_goal: Vec3,
        perception: Optional[PerceptionData] = None,
    ) -> Vec3:
        to_goal = final_goal - state.position
        dist = to_goal.norm()
        if dist <= self.config.local_goal_distance:
            self._last_local_goal = final_goal
            return final_goal

        direction = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        local_goal = state.position + direction * self.config.local_goal_distance

        # Keep the desired altitude trend but avoid huge vertical jumps in one
        # local planning cycle.
        dz = final_goal.z - state.position.z
        max_dz = max(0.5, self.config.vertical_speed * self.config.horizon)
        if abs(dz) > max_dz:
            local_goal = Vec3(local_goal.x, local_goal.y, state.position.z + max_dz * (1.0 if dz > 0.0 else -1.0))

        if perception and perception.obstacles:
            local_goal = self._nudge_goal_away_from_blocked_direction(state, local_goal, perception)

        self._last_local_goal = local_goal
        return local_goal

    def reached(self, state: PlannerState, local_goal: Vec3) -> bool:
        return state.position.distance_to(local_goal) <= self.config.local_goal_reached_radius

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
