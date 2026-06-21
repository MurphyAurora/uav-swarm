#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perception adapter for the EGO-like planner framework.

This module converts existing JSON obstacle/state messages into planner-native
Obstacle and PerceptionData objects.  It deliberately does not decide commands;
it only answers what the planner should consider unsafe.
"""

import time
from typing import Dict, Iterable, List, Optional

from .common import Obstacle, PerceptionData, Vec3


class PerceptionInterface:
    """Collect and normalize static, dynamic, and near-field obstacles."""

    def __init__(
        self,
        drone_radius: float = 0.35,
        default_obstacle_radius: float = 0.3,
        near_field_distance: float = 1.2,
        frame_id: str = "world",
    ):
        self.drone_radius = float(drone_radius)
        self.default_obstacle_radius = float(default_obstacle_radius)
        self.near_field_distance = float(near_field_distance)
        self.frame_id = str(frame_id)
        self._static_obstacles: List[Obstacle] = []
        self._dynamic_obstacles: List[Obstacle] = []
        self._swarm_obstacles: List[Obstacle] = []
        self._near_field_obstacles: List[Obstacle] = []
        self._last_stamp = 0.0

    def update_static_tracks(self, tracks: Iterable[Dict], stamp: Optional[float] = None) -> None:
        self._static_obstacles = [
            self._obstacle_from_track(track, source="static") for track in tracks
        ]
        self._last_stamp = float(stamp if stamp is not None else time.time())

    def update_dynamic_tracks(self, tracks: Iterable[Dict], stamp: Optional[float] = None) -> None:
        self._dynamic_obstacles = [
            self._obstacle_from_track(track, source="dynamic") for track in tracks
        ]
        self._last_stamp = float(stamp if stamp is not None else time.time())

    def update_swarm_states(self, states: Iterable[Dict], own_id: int, state_timeout: float = 2.0) -> None:
        """Treat other UAVs as predicted dynamic obstacles.

        This is the first multi-UAV extension point.  The simple implementation
        uses constant-velocity prediction so it can be used before a committed
        trajectory protocol is implemented.
        """
        now = time.time()
        obstacles = []
        for raw in states:
            drone_id = int(raw.get("id", raw.get("drone_id", 0)))
            if drone_id == int(own_id):
                continue
            stamp = float(raw.get("stamp", now))
            if now - stamp > float(state_timeout):
                continue
            obstacles.append(
                Obstacle(
                    position=Vec3.from_mapping(raw),
                    velocity=Vec3(
                        float(raw.get("vx", 0.0)),
                        float(raw.get("vy", 0.0)),
                        float(raw.get("vz", 0.0)),
                    ),
                    radius=self.drone_radius,
                    source="drone",
                    obstacle_id=f"x500_{drone_id}",
                )
            )
        self._swarm_obstacles = obstacles
        self._last_stamp = now

    def update_near_field_from_lidar_summary(
        self,
        nearest_distance: float,
        nearest_direction: Optional[Vec3] = None,
        current_position: Optional[Vec3] = None,
        stamp: Optional[float] = None,
    ) -> None:
        """Create a hard near-field obstacle from a LiDAR safety summary.

        The old branch often computes nearest distance/risk flags directly.
        This adapter lets that result become a planner obstacle without moving
        the old LiDAR parser into the new framework immediately.
        """
        dist = float(nearest_distance)
        if dist >= self.near_field_distance:
            self._near_field_obstacles = []
        else:
            direction = nearest_direction.normalized(Vec3(1.0, 0.0, 0.0)) if nearest_direction else Vec3(1.0, 0.0, 0.0)
            origin = current_position or Vec3()
            obs_pos = origin + direction * max(0.05, dist)
            self._near_field_obstacles = [
                Obstacle(
                    position=obs_pos,
                    radius=max(0.05, self.near_field_distance - dist),
                    velocity=Vec3(),
                    source="lidar_near_field",
                    obstacle_id="near_field",
                )
            ]
        self._last_stamp = float(stamp if stamp is not None else time.time())

    def build(self, current_position: Optional[Vec3] = None) -> PerceptionData:
        obstacles = []
        obstacles.extend(self._static_obstacles)
        obstacles.extend(self._dynamic_obstacles)
        obstacles.extend(self._swarm_obstacles)
        obstacles.extend(self._near_field_obstacles)

        nearest = 999.0
        if current_position is not None:
            for obs in obstacles:
                clearance = current_position.distance_to(obs.position) - float(obs.radius)
                nearest = min(nearest, clearance)

        near_field_danger = any(obs.source == "lidar_near_field" for obs in obstacles)
        return PerceptionData(
            obstacles=obstacles,
            nearest_distance=float(nearest),
            near_field_danger=near_field_danger,
            frame_id=self.frame_id,
            stamp=self._last_stamp,
        )

    def _obstacle_from_track(self, track: Dict, source: str) -> Obstacle:
        predictions = []
        for pred in list(track.get("predictions") or []):
            predictions.append(
                (
                    float(pred.get("t", pred.get("time", 0.0))),
                    Vec3.from_mapping(pred),
                )
            )
        return Obstacle(
            position=Vec3.from_mapping(track),
            radius=float(track.get("radius", self.default_obstacle_radius)),
            velocity=Vec3(
                float(track.get("vx", 0.0)),
                float(track.get("vy", 0.0)),
                float(track.get("vz", 0.0)),
            ),
            source=source,
            obstacle_id=str(track.get("id", track.get("track_id", ""))),
            predictions=predictions,
        )
