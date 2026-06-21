#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Swarm trajectory-sharing interface placeholder.

This module is intentionally lightweight in the first framework commit.  It is
where EGO-Swarm/MADER-like committed trajectory exchange can be added after the
single-UAV long-route planner is stable.
"""

from typing import Dict, Iterable, List

from .common import CandidateTrajectory, Obstacle, PlannerConfig, PlannerState, Vec3


class SwarmInterface:
    """Convert other UAV states or committed trajectories into obstacles."""

    def __init__(self, config: PlannerConfig = None):
        self.config = config or PlannerConfig()
        self._committed_trajs: Dict[int, CandidateTrajectory] = {}

    def update_committed_trajectory(self, drone_id: int, trajectory: CandidateTrajectory) -> None:
        self._committed_trajs[int(drone_id)] = trajectory

    def clear_stale(self, valid_drone_ids: Iterable[int]) -> None:
        valid = {int(item) for item in valid_drone_ids}
        self._committed_trajs = {
            drone_id: traj for drone_id, traj in self._committed_trajs.items() if drone_id in valid
        }

    def other_states_as_obstacles(self, states: Iterable[PlannerState], own_id: int) -> List[Obstacle]:
        obstacles = []
        for state in states:
            if int(state.drone_id) == int(own_id):
                continue
            obstacles.append(
                Obstacle(
                    position=state.position,
                    velocity=state.velocity,
                    radius=self.config.drone_radius,
                    source="drone",
                    obstacle_id=f"x500_{state.drone_id}",
                )
            )
        return obstacles

    def committed_trajs_as_obstacles(self, own_id: int) -> List[Obstacle]:
        obstacles = []
        for drone_id, traj in self._committed_trajs.items():
            if int(drone_id) == int(own_id):
                continue
            predictions = [(point.t, point.position) for point in traj.points]
            start = predictions[0][1] if predictions else Vec3()
            obstacles.append(
                Obstacle(
                    position=start,
                    velocity=traj.velocity,
                    radius=self.config.drone_radius,
                    source="drone",
                    obstacle_id=f"committed_x500_{drone_id}",
                    predictions=predictions,
                )
            )
        return obstacles
