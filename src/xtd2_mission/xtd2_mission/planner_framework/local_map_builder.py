#!/usr/bin/env python3
"""Local body-frame occupancy map built from framework perception data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .common import Obstacle, PlannerConfig, PlannerState


GridIndex = Tuple[int, int]
BodyPoint = Tuple[float, float]


@dataclass
class LocalOccupancyMap:
    occupied: List[List[bool]]
    points: List[BodyPoint]
    raw_count: int
    config: PlannerConfig

    def shape(self) -> Tuple[int, int]:
        return len(self.occupied), len(self.occupied[0]) if self.occupied else 0

    def body_to_grid(self, x: float, y: float) -> Optional[GridIndex]:
        if x < -self.config.astar_grid_back or x > self.config.astar_grid_forward:
            return None
        if y < -self.config.astar_grid_right or y > self.config.astar_grid_left:
            return None
        ix = int(round((x + self.config.astar_grid_back) / self.config.astar_resolution))
        iy = int(round((y + self.config.astar_grid_right) / self.config.astar_resolution))
        nx, ny = self.shape()
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix, iy
        return None

    def grid_to_body(self, idx: GridIndex) -> BodyPoint:
        ix, iy = idx
        return (
            ix * self.config.astar_resolution - self.config.astar_grid_back,
            iy * self.config.astar_resolution - self.config.astar_grid_right,
        )


class LocalMapBuilder:
    """Converts world-frame obstacles into a small inflated body-frame grid."""

    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()

    def build(self, state: PlannerState, obstacles: Sequence[Obstacle]) -> LocalOccupancyMap:
        points = self._extract_body_points(state, obstacles)
        occupied, raw_count = self._build_grid(points)
        return LocalOccupancyMap(occupied=occupied, points=points, raw_count=raw_count, config=self.config)

    def _extract_body_points(self, state: PlannerState, obstacles: Sequence[Obstacle]) -> List[BodyPoint]:
        c = math.cos(state.heading)
        s = math.sin(state.heading)
        points: List[BodyPoint] = []
        for obs in obstacles:
            if obs.source not in ("static", "lidar_near_field"):
                continue
            rel_x = obs.position.x - state.position.x
            rel_y = obs.position.y - state.position.y
            rel_z = obs.position.z - state.position.z
            body_x = c * rel_x + s * rel_y
            body_y = -s * rel_x + c * rel_y
            horizontal = math.hypot(body_x, body_y)
            if horizontal < self.config.astar_min_range or horizontal > self.config.astar_max_range:
                continue
            if rel_z < self.config.astar_z_min or rel_z > self.config.astar_z_max:
                continue
            if body_x < -self.config.astar_grid_back or body_x > self.config.astar_grid_forward:
                continue
            if body_y < -self.config.astar_grid_right or body_y > self.config.astar_grid_left:
                continue
            points.append((body_x, body_y))
        return points

    def _grid_shape(self) -> Tuple[int, int]:
        nx = int(round((self.config.astar_grid_forward + self.config.astar_grid_back) / self.config.astar_resolution)) + 1
        ny = int(round((self.config.astar_grid_left + self.config.astar_grid_right) / self.config.astar_resolution)) + 1
        return nx, ny

    def _build_grid(self, points: Sequence[BodyPoint]) -> Tuple[List[List[bool]], int]:
        nx, ny = self._grid_shape()
        occupied = [[False for _ in range(ny)] for _ in range(nx)]
        raw_count = 0
        inflate_cells = int(math.ceil(self.config.astar_inflation_radius / self.config.astar_resolution))
        probe = LocalOccupancyMap(occupied, [], 0, self.config)
        for x, y in points:
            cell = probe.body_to_grid(x, y)
            if cell is None:
                continue
            raw_count += 1
            cx, cy = cell
            for dx in range(-inflate_cells, inflate_cells + 1):
                for dy in range(-inflate_cells, inflate_cells + 1):
                    if dx * dx + dy * dy > inflate_cells * inflate_cells:
                        continue
                    ix = cx + dx
                    iy = cy + dy
                    if 0 <= ix < nx and 0 <= iy < ny:
                        occupied[ix][iy] = True

        start = probe.body_to_grid(0.0, 0.0)
        clear_cells = max(1, int(round(0.55 / self.config.astar_resolution)))
        if start is not None:
            sx, sy = start
            for dx in range(-clear_cells, clear_cells + 1):
                for dy in range(-clear_cells, clear_cells + 1):
                    ix = sx + dx
                    iy = sy + dy
                    if 0 <= ix < nx and 0 <= iy < ny:
                        occupied[ix][iy] = False
        return occupied, raw_count
