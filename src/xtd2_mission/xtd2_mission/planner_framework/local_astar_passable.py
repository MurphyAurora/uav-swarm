#!/usr/bin/env python3
"""Passability-tuned local A* planner for fast static-obstacle bring-up."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

from .common import CandidateTrajectory, PlannerConfig, PlannerState, Vec3
from .local_astar_tracker import LocalAStarPlanner as TrackerLocalAStarPlanner
from .local_map_builder import BodyPoint, LocalMapBuilder, LocalOccupancyMap


class PassableLocalMapBuilder(LocalMapBuilder):
    """Build an occupancy grid whose inflation matches the A* clearance gate."""

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

    def plan_candidate(self, state: PlannerState, local_goal: Vec3, obstacles) -> CandidateTrajectory | None:
        # Keep straight tracking straight.  The inherited candidate sorter used
        # maximum-clearance as the first key, so even when the goal corridor was
        # clear it could pick +30/+60 degree side paths and create the early drift
        # seen in the 20:09 log.  If the direct corridor is clear, bypass A* side
        # candidates and track a direct short path to the local goal.
        local_map = self.map_builder.build(state, obstacles)
        target_body = self._world_to_body(local_goal.x - state.position.x, local_goal.y - state.position.y, state.heading)
        front_blocked = self._front_blocked(local_map.points)
        corridor_clear = self._target_corridor_clear(local_map.points, target_body)
        if not self._bypass_active and not front_blocked and corridor_clear:
            self._side_latch = 0.0
            self._side_latch_until = 0.0
            body_path = self._direct_body_path(target_body)
            clear = self._sampled_path_clearance(body_path, local_map.points)
            nearest = min((math.hypot(x, y) for x, y in local_map.points), default=999.0)
            candidate = self.tracker.track(
                "local_astar:direct",
                state,
                body_path,
                local_goal,
                nearest,
                clear,
                {
                    "frontend": "local_astar_passable",
                    "direct_tracking": True,
                    "path_len": len(body_path),
                    "raw_points": local_map.raw_count,
                    "min_path_clearance": float(clear),
                    "nearest": float(nearest),
                    "front_blocked": front_blocked,
                    "corridor_blocked": False,
                    "mission_corridor_blocked": False,
                    "side_latch": 0.0,
                    "label": "direct",
                    "replan_reason": "direct_corridor_clear",
                    "bypass_active": False,
                    "bypass_side": 0.0,
                    "bypass_mode": "normal",
                    "bypass_target_body": {"x": float(target_body[0]), "y": float(target_body[1])},
                },
            )
            self.diagnostics = {
                "status": "direct_tracking",
                "replan_reason": "direct_corridor_clear",
                "raw_points": local_map.raw_count,
                "nearest": nearest,
                "front_blocked": front_blocked,
                "corridor_blocked": False,
                "mission_corridor_blocked": False,
                "side_latch": 0.0,
                "candidate_goals": 1,
                "scored_paths": 1,
                "best_path_clearance": float(clear),
                "hard_required_clearance": self._hard_required_clearance(),
                "safe_required_clearance": self._safe_required_clearance(),
                "bypass_active": False,
                "bypass_side": 0.0,
                "bypass_mode": "normal",
                "bypass_target_body": {"x": float(target_body[0]), "y": float(target_body[1])},
                "label": "direct",
                "path_len": len(body_path),
                "min_path_clearance": float(clear),
                "lookahead_body": dict(candidate.metadata.get("lookahead_body", {})),
                "speed": float(candidate.metadata.get("tracker_speed", 0.0)),
                "tracker": candidate.metadata.get("tracker", ""),
                "body_velocity": dict(candidate.metadata.get("body_velocity", {})),
            }
            return candidate
        return super().plan_candidate(state, local_goal, obstacles)

    def _direct_body_path(self, target_body: BodyPoint):
        tx, ty = target_body
        dist = max(math.hypot(tx, ty), 1.0e-6)
        length = min(max(1.4, dist), max(2.2, self.config.astar_local_goal_dist))
        ux, uy = tx / dist, ty / dist
        return [(0.0, 0.0), (0.55 * length * ux, 0.55 * length * uy), (length * ux, length * uy)]

    def _hard_required_clearance(self) -> float:
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
