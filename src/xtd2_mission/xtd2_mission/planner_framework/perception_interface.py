#!/usr/bin/env python3
"""Perception adapter for static, dynamic, and swarm obstacles."""

import math
import time
from typing import Dict, Iterable, List, Optional

from .common import Obstacle, PerceptionData, Vec3


class PerceptionInterface:
    def __init__(self, drone_radius: float = 0.35, default_obstacle_radius: float = 0.3, near_field_distance: float = 1.2, frame_id: str = "world"):
        self.drone_radius = float(drone_radius)
        self.default_obstacle_radius = float(default_obstacle_radius)
        self.near_field_distance = float(near_field_distance)
        self.frame_id = str(frame_id)
        self._static_obstacles: List[Obstacle] = []
        self._dynamic_obstacles: List[Obstacle] = []
        self._swarm_obstacles: List[Obstacle] = []
        self._last_stamp = 0.0
        self._min_lidar_range = 1.25

    def update_static_tracks(self, tracks: Iterable[Dict], stamp: Optional[float] = None) -> None:
        """Do not inject map/world-truth static tracks into the local planner."""
        self._static_obstacles = []
        self._last_stamp = float(stamp if stamp is not None else time.time())

    def update_dynamic_tracks(self, tracks: Iterable[Dict], stamp: Optional[float] = None) -> None:
        self._dynamic_obstacles = [self._obstacle_from_track(track, "dynamic") for track in tracks]
        self._last_stamp = float(stamp if stamp is not None else time.time())

    def update_swarm_states(self, states: Iterable[Dict], own_id: int, state_timeout: float = 2.0) -> None:
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
                    velocity=Vec3(float(raw.get("vx", 0.0)), float(raw.get("vy", 0.0)), float(raw.get("vz", 0.0))),
                    radius=self.drone_radius,
                    source="drone",
                    obstacle_id=f"x500_{drone_id}",
                )
            )
        self._swarm_obstacles = obstacles
        self._last_stamp = now

    def update_lidar_obstacles(self, obstacles: Iterable[Obstacle], stamp: Optional[float] = None) -> None:
        self._lidar_obstacles = list(obstacles)
        self._last_stamp = float(stamp if stamp is not None else time.time())

    def build(self, current_position: Optional[Vec3] = None) -> PerceptionData:
        lidar_obstacles = getattr(self, "_lidar_obstacles", [])
        if current_position is not None:
            # Several logs showed a dense false ring at about 1.10 m even in open
            # space.  Do this second-stage filter here as well, so launch files
            # cannot reintroduce old --lidar-min-range values.
            lidar_obstacles = [obs for obs in lidar_obstacles if self._horizontal_distance(current_position, obs.position) >= self._min_lidar_range]
        obstacles = [*self._static_obstacles, *self._dynamic_obstacles, *self._swarm_obstacles, *lidar_obstacles]
        nearest = 999.0
        nearest_lidar_clearance = 999.0
        if current_position is not None:
            for obs in obstacles:
                clearance = current_position.distance_to(obs.position) - float(obs.radius)
                nearest = min(nearest, clearance)
                if obs.source == "lidar_near_field":
                    nearest_lidar_clearance = min(nearest_lidar_clearance, clearance - self.drone_radius)
        near_field_danger = nearest_lidar_clearance <= self.near_field_distance
        return PerceptionData(
            obstacles=obstacles,
            nearest_distance=float(nearest),
            near_field_danger=near_field_danger,
            frame_id=self.frame_id,
            stamp=self._last_stamp,
        )

    @staticmethod
    def _horizontal_distance(a: Vec3, b: Vec3) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _obstacle_from_track(self, track: Dict, source: str) -> Obstacle:
        predictions = []
        for pred in list(track.get("predictions") or []):
            predictions.append((float(pred.get("t", pred.get("time", 0.0))), Vec3.from_mapping(pred)))
        return Obstacle(
            position=Vec3.from_mapping(track),
            radius=float(track.get("radius", self.default_obstacle_radius)),
            velocity=Vec3(float(track.get("vx", 0.0)), float(track.get("vy", 0.0)), float(track.get("vz", 0.0))),
            source=source,
            obstacle_id=str(track.get("id", track.get("track_id", ""))),
            predictions=predictions,
        )
