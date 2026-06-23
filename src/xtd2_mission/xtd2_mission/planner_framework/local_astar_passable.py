#!/usr/bin/env python3
"""Passability-tuned local A* planner for fast static-obstacle bring-up."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from .common import PlannerConfig
from .local_astar_tracker import LocalAStarPlanner as TrackerLocalAStarPlanner
from .local_map_builder import BodyPoint, LocalMapBuilder, LocalOccupancyMap


class PassableLocalMapBuilder(LocalMapBuilder):
    """Build an occupancy grid whose inflation matches the A* clearance gate.

    The previous chain used a very small grid inflation but then rejected paths
    with a much larger raw-point clearance threshold.  That made A* find paths
    and then throw them away.  This map uses a moderate grid inflation and a
    smaller start-cell clear zone so genuine side gaps remain searchable.
    """

    def _build_grid(self, points: Sequence[BodyPoint]) -> Tuple[list, int]:
        nx, ny = self._grid_shape()
        occupied = [[False for _ in range(ny)] for _ in range(nx)]
        raw_count = 0
        effective_inflation = max(float(self.config.astar_inflation_radius), 0.50)
        inflate_cells = int(math.ceil(effective_inflation / self.config.astar_resolution))
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
        clear_cells = max(1, int(round(0.35 / self.config.astar_resolution)))
        if start is not None:
            sx, sy = start
            for dx in range(-clear_cells, clear_cells + 1):
                for dy in range(-clear_cells, clear_cells + 1):
                    ix = sx + dx
                    iy = sy + dy
                    if 0 <= ix < nx and 0 <= iy < ny:
                        occupied[ix][iy] = False
        return occupied, raw_count


class LocalAStarPlanner(TrackerLocalAStarPlanner):
    def __init__(self, config: PlannerConfig | None = None):
        super().__init__(config)
        self.map_builder = PassableLocalMapBuilder(self.config)

    def _hard_required_clearance(self) -> float:
        # This is the raw point clearance after grid inflation.  It should not be
        # larger than the actual passable gap; the grid already carries safety.
        return max(0.72, self.config.drone_radius + 0.18 + 0.18)

    def _safe_required_clearance(self) -> float:
        return max(1.02, self._hard_required_clearance() + 0.28)

    def _front_blocked(self, points: Sequence[BodyPoint]) -> bool:
        width = max(0.62, self.config.drone_radius + 0.25)
        return any(0.25 <= x <= 2.8 and abs(y) <= width for x, y in points)

    def _target_corridor_clear(self, points: Sequence[BodyPoint], target_body: BodyPoint) -> bool:
        tx, ty = target_body
        dist = math.hypot(tx, ty)
        if dist < 1.0e-6:
            return True
        ux, uy = tx / dist, ty / dist
        length = min(3.4, max(1.6, dist))
        half_width = max(0.62, self.config.drone_radius + 0.25)
        for x, y in points:
            along = x * ux + y * uy
            if 0.25 <= along <= length and abs(-uy * x + ux * y) <= half_width:
                return False
        return True
