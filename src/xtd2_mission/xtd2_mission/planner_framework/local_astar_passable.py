#!/usr/bin/env python3
"""Passability-tuned local A* planner for fast static-obstacle bring-up."""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

from .common import CandidateTrajectory, PlannerConfig, PlannerState, Vec3
from .local_astar_tracker import LocalAStarPlanner as TrackerLocalAStarPlanner
from .local_map_builder import BodyPoint, GridIndex, LocalMapBuilder, LocalOccupancyMap
from .local_subgoal_selector import LocalSubgoalSelector


class PassableLocalMapBuilder(LocalMapBuilder):
    """Build an occupancy grid whose inflation matches the A* clearance gate."""

    def _build_grid(self, points: Sequence[BodyPoint]) -> Tuple[list, int]:
        nx, ny = self._grid_shape()
        occupied = [[False for _ in range(ny)] for _ in range(nx)]
        raw_count = 0
        effective_inflation = max(float(self.config.astar_inflation_radius), 0.42)
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
        self.subgoal_selector = LocalSubgoalSelector(self.config)
        self._single_obstacle_mode = False

    def plan_candidate(self, state: PlannerState, local_goal: Vec3, obstacles) -> CandidateTrajectory | None:
        # Logic split:
        # - Single obstacle smoke tests are best handled by the stable bypass state
        #   machine from local_astar_tracker: pick one side, hold it, then rejoin.
        # - Multi-obstacle / corridor tests use LocalSubgoalSelector.
        # This prevents the multi-subgoal search from replacing the simple and
        # previously working single-pillar behavior.
        real_obstacle_count = sum(1 for obs in obstacles if getattr(obs, "source", "") not in ("boundary", "lidar_near_field"))
        self._single_obstacle_mode = real_obstacle_count <= 1

        local_map = self.map_builder.build(state, obstacles)
        target_body = self._world_to_body(local_goal.x - state.position.x, local_goal.y - state.position.y, state.heading)
        front_blocked = self._front_blocked(local_map.points)
        corridor_clear = self._target_corridor_clear(local_map.points, target_body)
        direct_body_path = self._direct_body_path(target_body)
        direct_clear = self._sampled_path_clearance(direct_body_path, local_map.points)
        direct_clear_enough = direct_clear >= self._safe_required_clearance()
        rejoin_ready = (not front_blocked and corridor_clear and direct_clear >= self._hard_required_clearance())
        if rejoin_ready:
            self._side_latch = 0.0
            self._side_latch_until = 0.0
            if self._bypass_active:
                self._clear_bypass(getattr(state, "stamp", 0.0), force_rejoin=True)
        if not self._bypass_active and not front_blocked and corridor_clear and direct_clear_enough:
            self._side_latch = 0.0
            self._side_latch_until = 0.0
            body_path = direct_body_path
            clear = direct_clear
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
                    "subgoal_debug": {"mode": "direct_corridor_clear", "single_obstacle_mode": self._single_obstacle_mode},
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
                "subgoal_debug": {"mode": "direct_corridor_clear", "single_obstacle_mode": self._single_obstacle_mode},
            }
            return candidate
        return super().plan_candidate(state, local_goal, obstacles)

    def _select_path(self, local_map: LocalOccupancyMap, start: GridIndex, target_body: BodyPoint) -> Optional[Dict[str, object]]:
        if getattr(self, "_single_obstacle_mode", False):
            # In single-obstacle mode the parent tracker already provides a stable
            # committed bypass/rejoin path.  Do not let the multi-subgoal selector
            # keep choosing side subgoals and stall beside the obstacle.
            return super()._select_path(local_map, start, target_body)

        side_hint = self._side_latch if abs(self._side_latch) > 0.5 else 0.0
        selected = self.subgoal_selector.select(
            local_map=local_map,
            start=start,
            target_body=target_body,
            astar_fn=self._astar,
            shortcut_fn=self._shortcut,
            path_clearance_fn=self._sampled_path_clearance,
            path_length_fn=self._path_length,
            nearest_free_fn=self._nearest_free,
            hard_clearance=self._hard_required_clearance(),
            safe_clearance=self._safe_required_clearance(),
            side_hint=side_hint,
        )
        debug = dict(self.subgoal_selector.last_debug)
        self._last_candidate_count = int(debug.get("subgoal_candidates", 0))
        self._last_scored_count = int(debug.get("subgoal_reachable", 0))
        if selected is not None:
            self._last_best_clearance = float(selected.get("min_path_clearance", 999.0))
            self._last_subgoal_debug = debug
            return selected
        self._last_subgoal_debug = debug
        return None

    def _bypass_target(self, points, mission_target_body: Tuple[float, float], now: float,
                       front_blocked: bool, mission_corridor_blocked: bool):
        if getattr(self, "_single_obstacle_mode", False):
            return super()._bypass_target(points, mission_target_body, now, front_blocked, mission_corridor_blocked)

        direct_clear = self._sampled_path_clearance(self._direct_body_path(mission_target_body), points)
        corridor_clear = not mission_corridor_blocked and direct_clear >= self._hard_required_clearance()
        side_hint = self._side_latch if abs(self._side_latch) > 0.5 else 0.0

        if self._bypass_active:
            min_hold = now - self._bypass_started >= 0.80
            timed_out = now - self._bypass_started > 3.2
            if min_hold and (corridor_clear or timed_out):
                self._clear_bypass(now, force_rejoin=True)
                return None, "committed_rejoin"
            self._bypass_until = max(self._bypass_until, now + 0.35)
            return self._make_bypass_goal(self._bypass_side), "committed_bypass_hold"

        if now < self._rejoin_forced_until and corridor_clear:
            return None, "committed_rejoin"

        if (front_blocked or mission_corridor_blocked or direct_clear < self._safe_required_clearance()) and abs(side_hint) > 0.5:
            self._bypass_active = True
            self._bypass_side = side_hint
            self._bypass_started = now
            self._bypass_until = now + 1.8
            return self._make_bypass_goal(side_hint), "committed_bypass_start"

        return None, "subgoal_search"

    def _diag(self, *args, **kwargs):
        out = super()._diag(*args, **kwargs)
        out["subgoal_debug"] = dict(getattr(self, "_last_subgoal_debug", {}))
        out["single_obstacle_mode"] = bool(getattr(self, "_single_obstacle_mode", False))
        return out

    def _direct_body_path(self, target_body: BodyPoint):
        tx, ty = target_body
        dist = max(math.hypot(tx, ty), 1.0e-6)
        length = min(max(1.2, dist), max(2.0, self.config.astar_local_goal_dist))
        ux, uy = tx / dist, ty / dist
        return [(0.0, 0.0), (0.55 * length * ux, 0.55 * length * uy), (length * ux, length * uy)]

    def _hard_required_clearance(self) -> float:
        return max(0.45, self.config.drone_radius + 0.20)

    def _safe_required_clearance(self) -> float:
        return max(0.70, self._hard_required_clearance() + 0.20)

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
