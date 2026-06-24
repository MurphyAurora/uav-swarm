#!/usr/bin/env python3
"""Free-space subgoal selector for local A* planning.

This follows the mature local-planner pattern: when the direct corridor is not
usable, sample a small lattice of local free-space goals, test reachability with
A*, then score the reachable goals by progress, clearance, path length, and
side consistency.  The planner no longer depends on one hand-written bypass
point.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .local_map_builder import BodyPoint, GridIndex, LocalOccupancyMap


class LocalSubgoalSelector:
    def __init__(self, config):
        self.config = config
        self.last_debug: Dict[str, object] = {}

    def select(
        self,
        local_map: LocalOccupancyMap,
        start: GridIndex,
        target_body: BodyPoint,
        astar_fn: Callable[[LocalOccupancyMap, GridIndex, GridIndex], Optional[List[GridIndex]]],
        shortcut_fn: Callable[[LocalOccupancyMap, List[GridIndex]], List[GridIndex]],
        path_clearance_fn: Callable[[Sequence[BodyPoint], Sequence[BodyPoint]], float],
        path_length_fn: Callable[[Sequence[GridIndex]], float],
        nearest_free_fn: Callable[[LocalOccupancyMap, GridIndex], Optional[GridIndex]],
        hard_clearance: float,
        safe_clearance: float,
        side_hint: float = 0.0,
    ) -> Optional[Dict[str, object]]:
        raw_candidates = self._sample_candidate_cells(local_map, target_body, nearest_free_fn, side_hint)
        scored: List[Dict[str, object]] = []
        target_angle = math.atan2(target_body[1], target_body[0])
        target_norm = max(math.hypot(target_body[0], target_body[1]), 1.0e-6)
        target_unit = (target_body[0] / target_norm, target_body[1] / target_norm)

        for cell, label in raw_candidates:
            path = astar_fn(local_map, start, cell)
            if not path:
                continue
            path = shortcut_fn(local_map, path)
            if len(path) < 2:
                continue
            body = [local_map.grid_to_body(idx) for idx in path]
            clear = path_clearance_fn(body, local_map.points)
            if clear < hard_clearance:
                continue
            forward_score = self._path_forward_score(body, target_unit)
            if forward_score < 0.20:
                continue
            turn_cost = self._turn_cost(body)
            end_x, end_y = body[-1]
            progress = end_x * target_unit[0] + end_y * target_unit[1]
            if progress < 0.25:
                continue
            end_angle = math.atan2(end_y, end_x)
            angle_error = abs(self._angle_diff(end_angle, target_angle))
            lateral = abs(-target_unit[1] * end_x + target_unit[0] * end_y)
            if abs(side_hint) > 0.5 and lateral < 0.45:
                continue
            length = path_length_fn(path)
            safe_deficit = max(0.0, safe_clearance - clear)
            side_penalty = 0.0
            if abs(side_hint) > 0.5:
                if end_y * side_hint < 0.15:
                    side_penalty += 25.0
                else:
                    side_penalty -= 1.5
            # Score resembles mature local planners: prefer reachable progress,
            # adequate clearance, short/smooth path, and avoid unnecessary lateral
            # detours when there is a good forward option.
            score = (
                1.00 * length
                + 1.40 * angle_error
                + 0.35 * lateral
                + 0.45 * turn_cost
                + 7.50 * safe_deficit
                - 0.55 * progress
                - 0.35 * forward_score
                - 0.10 * min(clear, 3.0)
                + side_penalty
            )
            scored.append(
                {
                    "score": float(score),
                    "path": path,
                    "body_path": body,
                    "label": label,
                    "min_path_clearance": float(clear),
                    "safe_class": bool(clear >= safe_clearance),
                    "progress": float(progress),
                    "forward_score": float(forward_score),
                    "turn_cost": float(turn_cost),
                    "lateral": float(lateral),
                    "length": float(length),
                }
            )

        scored.sort(key=lambda item: (not bool(item["safe_class"]), float(item["score"])))
        self.last_debug = {
            "subgoal_candidates": len(raw_candidates),
            "subgoal_reachable": len(scored),
            "subgoal_best": self._short_item(scored[0]) if scored else None,
            "side_hint": float(side_hint),
        }
        return scored[0] if scored else None

    def _sample_candidate_cells(
        self,
        local_map: LocalOccupancyMap,
        target_body: BodyPoint,
        nearest_free_fn: Callable[[LocalOccupancyMap, GridIndex], Optional[GridIndex]],
        side_hint: float,
    ) -> List[Tuple[GridIndex, str]]:
        target_angle = math.atan2(target_body[1], target_body[0])
        radii = [1.4, 2.0, 2.8, 3.6, min(4.5, max(3.2, self.config.astar_local_goal_dist))]
        if abs(side_hint) > 0.5:
            offsets = [int(side_hint * deg) for deg in (35, 50, 70, 90)]
        else:
            offsets = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
        seen = set()
        out: List[Tuple[GridIndex, str]] = []
        for radius in radii:
            for deg in offsets:
                ang = target_angle + math.radians(deg)
                x = math.cos(ang) * radius
                y = math.sin(ang) * radius
                x = max(-self.config.astar_grid_back + 0.3, min(self.config.astar_grid_forward - 0.3, x))
                y = max(-self.config.astar_grid_right + 0.3, min(self.config.astar_grid_left - 0.3, y))
                if x < 0.20:
                    continue
                cell = local_map.body_to_grid(x, y)
                if cell is None:
                    continue
                free = nearest_free_fn(local_map, cell)
                if free is None or free in seen:
                    continue
                bx, by = local_map.grid_to_body(free)
                if bx < 0.15 or math.hypot(bx, by) < 0.8:
                    continue
                seen.add(free)
                out.append((free, f"subgoal:{deg:+d}deg/{radius:.1f}m"))
        return out

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    @staticmethod
    def _short_item(item: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not item:
            return None
        return {
            "label": item.get("label"),
            "score": float(item.get("score", 0.0)),
            "clearance": float(item.get("min_path_clearance", 0.0)),
            "progress": float(item.get("progress", 0.0)),
            "forward_score": float(item.get("forward_score", 0.0)),
            "turn_cost": float(item.get("turn_cost", 0.0)),
            "lateral": float(item.get("lateral", 0.0)),
            "length": float(item.get("length", 0.0)),
            "safe_class": bool(item.get("safe_class", False)),
        }

    @staticmethod
    def _path_forward_score(body_path: Sequence[BodyPoint], target_unit: BodyPoint) -> float:
        if len(body_path) < 2:
            return 0.0
        score = 0.0
        length = 0.0
        last = body_path[0]
        for point in body_path[1:]:
            dx = point[0] - last[0]
            dy = point[1] - last[1]
            seg = math.hypot(dx, dy)
            if seg > 1.0e-6:
                score += max(-0.5, min(1.0, (dx * target_unit[0] + dy * target_unit[1]) / seg)) * seg
                length += seg
            last = point
        return score / max(length, 1.0e-6)

    @staticmethod
    def _turn_cost(body_path: Sequence[BodyPoint]) -> float:
        if len(body_path) < 3:
            return 0.0
        cost = 0.0
        last_heading = None
        last = body_path[0]
        for point in body_path[1:]:
            dx = point[0] - last[0]
            dy = point[1] - last[1]
            if math.hypot(dx, dy) > 1.0e-6:
                heading = math.atan2(dy, dx)
                if last_heading is not None:
                    cost += abs(math.atan2(math.sin(heading - last_heading), math.cos(heading - last_heading)))
                last_heading = heading
            last = point
        return cost
