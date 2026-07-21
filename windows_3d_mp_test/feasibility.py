from collections import deque
import math

import numpy as np


class StaticSceneFeasibility:
    """Lightweight 2D grid precheck for static cylinder maps.

    This does not replace the motion primitive planner. It only answers whether
    the XY projection of a generated static map has a reasonable start-goal
    corridor after inflating obstacles by the UAV safety radius.
    """

    def __init__(self, resolution=0.15, safety_margin=0.8):
        self.resolution = float(resolution)
        self.safety_margin = float(safety_margin)

    def _bounds_tuple(self, bounds, start, goal, obstacles):
        if bounds is not None:
            if isinstance(bounds, dict):
                return (
                    bounds.get("x_min", bounds.get("xmin", -math.inf)),
                    bounds.get("x_max", bounds.get("xmax", math.inf)),
                    bounds.get("y_min", bounds.get("ymin", -math.inf)),
                    bounds.get("y_max", bounds.get("ymax", math.inf)),
                )
            if all(hasattr(bounds, name) for name in ("xmin", "xmax", "ymin", "ymax")):
                return bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax
            return tuple(bounds)

        xs = [float(start[0]), float(goal[0])]
        ys = [float(start[1]), float(goal[1])]
        for obs in obstacles:
            xs.extend([float(obs.x - obs.radius), float(obs.x + obs.radius)])
            ys.extend([float(obs.y - obs.radius), float(obs.y + obs.radius)])
        pad = max(2.0, 2.0 * self.safety_margin)
        return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad

    def _to_index(self, x, y, x_min, y_min):
        ix = int(round((float(x) - x_min) / self.resolution))
        iy = int(round((float(y) - y_min) / self.resolution))
        return ix, iy

    def _inside_grid(self, ix, iy, nx, ny):
        return 0 <= ix < nx and 0 <= iy < ny

    def _is_blocked(self, x, y, obstacles):
        for obs in obstacles:
            clearance = math.hypot(x - float(obs.x), y - float(obs.y)) - float(obs.radius)
            if clearance < self.safety_margin:
                return True
        return False

    def _point_clearance(self, x, y, obstacles):
        if not obstacles:
            return math.inf
        return min(math.hypot(x - float(obs.x), y - float(obs.y)) - float(obs.radius) for obs in obstacles)

    def evaluate(self, start, goal, obstacles, bounds=None):
        x_min, x_max, y_min, y_max = self._bounds_tuple(bounds, start, goal, obstacles)
        nx = int(math.floor((x_max - x_min) / self.resolution)) + 1
        ny = int(math.floor((y_max - y_min) / self.resolution)) + 1

        if nx <= 1 or ny <= 1 or nx * ny > 600000:
            return {
                "feasible_xy": False,
                "reason": "invalid_or_too_large_grid",
                "grid_cells": int(max(0, nx * ny)),
                "xy_path_length": math.inf,
                "xy_min_clearance": -math.inf,
            }

        start_idx = self._to_index(start[0], start[1], x_min, y_min)
        goal_idx = self._to_index(goal[0], goal[1], x_min, y_min)
        if not self._inside_grid(*start_idx, nx, ny):
            return self._blocked_result("start_outside_bounds", nx * ny)
        if not self._inside_grid(*goal_idx, nx, ny):
            return self._blocked_result("goal_outside_bounds", nx * ny)

        blocked = np.zeros((nx, ny), dtype=bool)
        for ix in range(nx):
            x = x_min + ix * self.resolution
            for iy in range(ny):
                y = y_min + iy * self.resolution
                blocked[ix, iy] = self._is_blocked(x, y, obstacles)

        if blocked[start_idx]:
            return self._blocked_result("start_inside_inflated_obstacle", nx * ny)
        if blocked[goal_idx]:
            return self._blocked_result("goal_inside_inflated_obstacle", nx * ny)

        parent = {}
        q = deque([start_idx])
        parent[start_idx] = None
        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        while q:
            current = q.popleft()
            if current == goal_idx:
                break
            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not self._inside_grid(*nxt, nx, ny):
                    continue
                if blocked[nxt] or nxt in parent:
                    continue
                parent[nxt] = current
                q.append(nxt)

        if goal_idx not in parent:
            return self._blocked_result("no_xy_corridor", nx * ny)

        grid_path = []
        cur = goal_idx
        while cur is not None:
            grid_path.append(cur)
            cur = parent[cur]
        grid_path.reverse()

        xy_points = [(x_min + ix * self.resolution, y_min + iy * self.resolution) for ix, iy in grid_path]
        xy_length = 0.0
        for p0, p1 in zip(xy_points[:-1], xy_points[1:]):
            xy_length += math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        min_clearance = min(self._point_clearance(x, y, obstacles) for x, y in xy_points)

        return {
            "feasible_xy": True,
            "reason": "ok",
            "grid_cells": int(nx * ny),
            "xy_path_length": float(xy_length),
            "xy_min_clearance": float(min_clearance),
        }

    def _blocked_result(self, reason, grid_cells):
        return {
            "feasible_xy": False,
            "reason": reason,
            "grid_cells": int(grid_cells),
            "xy_path_length": math.inf,
            "xy_min_clearance": -math.inf,
        }
