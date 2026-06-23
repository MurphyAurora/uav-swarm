#!/usr/bin/env python3
"""Stable local A* frontend for the planner framework.

This is intentionally conservative for the single-UAV static-obstacle bring-up:
A* remains a frontend path generator, but it no longer uses extreme side goals
or reverse/away actions.  The final safety shield may brake/project, not invent
motion.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .common import CandidateTrajectory, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .local_map_builder import BodyPoint, GridIndex, LocalMapBuilder, LocalOccupancyMap


class LocalAStarPlanner:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self.map_builder = LocalMapBuilder(self.config)
        self._committed_world_path: List[Vec3] = []
        self._committed_label = ""
        self._committed_until = 0.0
        self._last_replan_time = 0.0
        self._last_goal = Vec3()
        self._side_latch = 0.0
        self._side_latch_until = 0.0
        self._last_cmd_body: BodyPoint = (0.0, 0.0)
        self.diagnostics: Dict[str, object] = {}

    def plan_candidate(self, state: PlannerState, local_goal: Vec3, obstacles) -> Optional[CandidateTrajectory]:
        now = time.time()
        local_map = self.map_builder.build(state, obstacles)
        start = local_map.body_to_grid(0.0, 0.0)
        if start is None:
            self.diagnostics = {"status": "no_start_cell"}
            return None

        target_body = self._world_to_body(local_goal.x - state.position.x, local_goal.y - state.position.y, state.heading)
        nearest = min((math.hypot(x, y) for x, y in local_map.points), default=999.0)
        front_blocked = self._front_blocked(local_map.points)
        corridor_blocked = not self._target_corridor_clear(local_map.points, target_body)
        self._update_side_latch(local_map.points, target_body, now, front_blocked, corridor_blocked)

        replan_reason = self._replan_reason(now, state, local_goal, local_map, corridor_blocked)
        if replan_reason:
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self._committed_world_path = []
                self.diagnostics = self._diag("no_astar_path", replan_reason, local_map, nearest, front_blocked, corridor_blocked, None)
                return None
            self._commit_path(state, selected, local_goal, now)

        body_path = self._committed_body_path(state)
        if len(body_path) < 2 or not self._path_is_valid(local_map, body_path):
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self._committed_world_path = []
                self.diagnostics = self._diag("committed_path_invalid", "invalid_path_no_replacement", local_map, nearest, front_blocked, corridor_blocked, None)
                return None
            self._commit_path(state, selected, local_goal, now)
            body_path = self._committed_body_path(state)

        body_path = self._trim_passed_points(body_path)
        look_x, look_y = self._lookahead(body_path)
        look_x = max(0.28, look_x)
        clear = self._sampled_path_clearance(body_path, local_map.points)
        speed = min(self.config.cruise_speed, self.config.max_speed, 0.42)
        if nearest < 2.0 or clear < self._safe_required_clearance():
            speed = min(speed, 0.16)
        if nearest < 1.55:
            speed = min(speed, 0.12)

        norm = max(math.hypot(look_x, look_y), 1.0e-6)
        bx = speed * look_x / norm
        by = speed * look_y / norm
        bx, by = self._smooth_cmd_body(bx, by)
        vx, vy = self._body_to_world(bx, by, state.heading)
        dz = local_goal.z - state.position.z
        vz = max(-0.05, min(0.05, 0.25 * dz)) if abs(dz) >= 0.25 else 0.0
        velocity = Vec3(vx, vy, vz).limit_norm(self.config.max_speed)

        self.diagnostics = self._diag(
            "tracking_committed_path", replan_reason or "latched", local_map, nearest, front_blocked, corridor_blocked,
            {
                "label": self._committed_label,
                "committed_left_sec": max(0.0, self._committed_until - now),
                "path_len": len(body_path),
                "min_path_clearance": float(clear),
                "lookahead_body": {"x": float(look_x), "y": float(look_y)},
                "speed": float(speed),
            },
        )
        return self._path_rollout(
            f"local_astar:{self._committed_label}", velocity, state, body_path, speed,
            {
                "frontend": "local_astar_stable",
                "path_len": len(body_path),
                "raw_points": local_map.raw_count,
                "min_path_clearance": float(clear),
                "nearest": nearest,
                "front_blocked": front_blocked,
                "corridor_blocked": corridor_blocked,
                "side_latch": self._side_latch,
            },
        )

    def _diag(self, status: str, reason: str, local_map: LocalOccupancyMap, nearest: float,
              front_blocked: bool, corridor_blocked: bool, extra: Optional[Dict]) -> Dict[str, object]:
        out: Dict[str, object] = {
            "status": status,
            "replan_reason": reason,
            "raw_points": local_map.raw_count,
            "nearest": nearest,
            "front_blocked": front_blocked,
            "corridor_blocked": corridor_blocked,
            "side_latch": self._side_latch,
            "side_latch_left_sec": max(0.0, self._side_latch_until - time.time()),
            "candidate_goals": int(getattr(self, "_last_candidate_count", 0)),
            "scored_paths": int(getattr(self, "_last_scored_count", 0)),
            "best_path_clearance": float(getattr(self, "_last_best_clearance", 999.0)),
            "hard_required_clearance": self._hard_required_clearance(),
            "safe_required_clearance": self._safe_required_clearance(),
        }
        if extra:
            out.update(extra)
        return out

    def _replan_reason(self, now: float, state: PlannerState, local_goal: Vec3,
                       local_map: LocalOccupancyMap, corridor_blocked: bool) -> str:
        if not self._committed_world_path:
            return "no_committed_path"
        if now >= self._committed_until:
            return "path_latch_expired"
        if now - self._last_replan_time >= self.config.astar_replan_interval:
            body_path = self._committed_body_path(state)
            if not self._path_is_valid(local_map, body_path):
                return "committed_path_blocked"
            if corridor_blocked:
                return "corridor_blocked_refresh"
        if local_goal.distance_to(self._last_goal) > max(1.0, 0.5 * self.config.astar_local_goal_dist):
            return "local_goal_changed"
        return ""

    def _commit_path(self, state: PlannerState, selected: Dict[str, object], local_goal: Vec3, now: float) -> None:
        body_path = list(selected["body_path"])
        self._committed_world_path = [self._body_point_to_world(state, x, y) for x, y in body_path]
        self._committed_label = str(selected["label"])
        self._committed_until = now + self.config.astar_path_latch_sec
        self._last_replan_time = now
        self._last_goal = local_goal
        self._last_best_clearance = float(selected.get("min_path_clearance", 999.0))

    def _candidate_goals(self, local_map: LocalOccupancyMap, target_body: BodyPoint) -> List[Tuple[GridIndex, str]]:
        target_angle = math.atan2(target_body[1], target_body[0])
        front_blocked = self._front_blocked(local_map.points)
        if abs(self._side_latch) > 1.0e-6:
            offsets = [int(self._side_latch * a) for a in (30, 45, 60, 75, 85)]
        elif front_blocked or not self._target_corridor_clear(local_map.points, target_body):
            side = self._pick_side(local_map.points, target_body)
            offsets = [int(side * a) for a in (30, 45, 60, 75, 85)] + [int(-side * a) for a in (45, 60)]
        else:
            offsets = [0, 15, -15, 30, -30, 45, -45, 60, -60]
        radii = [self.config.astar_local_goal_dist, 4.5, 3.6, 2.8]
        out: List[Tuple[GridIndex, str]] = []
        seen = set()
        for deg in offsets:
            for radius in radii:
                ang = target_angle + math.radians(deg)
                x = max(-self.config.astar_grid_back + 0.4, min(self.config.astar_grid_forward - 0.4, math.cos(ang) * radius))
                y = max(-self.config.astar_grid_right + 0.4, min(self.config.astar_grid_left - 0.4, math.sin(ang) * radius))
                if x < 0.45:
                    continue
                if (front_blocked or abs(self._side_latch) > 1.0e-6) and abs(y) < 0.85 and x > 1.0:
                    continue
                if abs(self._side_latch) > 1.0e-6 and y * self._side_latch < 0.30 and x > 0.5:
                    continue
                cell = local_map.body_to_grid(x, y)
                if cell is None:
                    continue
                free = self._nearest_free(local_map, cell)
                if free is None or free in seen:
                    continue
                bx, by = local_map.grid_to_body(free)
                if bx < 0.35 or math.hypot(bx, by) < 1.5:
                    continue
                seen.add(free)
                out.append((free, f"{deg:+d}deg/{radius:.1f}m"))
        self._last_candidate_count = len(out)
        return out

    def _select_path(self, local_map: LocalOccupancyMap, start: GridIndex, target_body: BodyPoint) -> Optional[Dict[str, object]]:
        scored: List[Dict[str, object]] = []
        for goal, label in self._candidate_goals(local_map, target_body):
            path = self._astar(local_map, start, goal)
            if not path:
                continue
            path = self._shortcut(local_map, path)
            if len(path) < 2:
                continue
            body = [local_map.grid_to_body(cell) for cell in path]
            clear = self._sampled_path_clearance(body, local_map.points)
            if clear < self._hard_required_clearance():
                continue
            end_x, end_y = body[-1]
            target_angle = math.atan2(target_body[1], target_body[0])
            end_angle = math.atan2(end_y, end_x)
            progress = end_x * math.cos(target_angle) + end_y * math.sin(target_angle)
            score = self._path_length(path) + 1.8 * abs(self._angle_diff(end_angle, target_angle)) - 0.35 * progress
            score += max(0.0, self._safe_required_clearance() - clear) * 10.0
            if abs(self._side_latch) > 1.0e-6 and end_y * self._side_latch < 0.35:
                score += 40.0
            scored.append({"score": score, "path": path, "body_path": body, "label": label, "min_path_clearance": clear, "safe_class": clear >= self._safe_required_clearance()})
        self._last_scored_count = len(scored)
        if not scored:
            return None
        scored.sort(key=lambda item: (not bool(item["safe_class"]), -float(item["min_path_clearance"]), float(item["score"])))
        self._last_best_clearance = float(scored[0]["min_path_clearance"])
        return scored[0]

    def _nearest_free(self, local_map: LocalOccupancyMap, preferred: GridIndex, max_r: int = 10) -> Optional[GridIndex]:
        nx, ny = local_map.shape()
        px, py = preferred
        best = None
        best_score = 1.0e9
        for r in range(0, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    cell = (px + dx, py + dy)
                    ix, iy = cell
                    if not (0 <= ix < nx and 0 <= iy < ny) or local_map.occupied[ix][iy]:
                        continue
                    bx, by = local_map.grid_to_body(cell)
                    clear = self._clearance_to_points((bx, by), local_map.points)
                    score = dx * dx + dy * dy - 0.25 * clear
                    if score < best_score:
                        best, best_score = cell, score
            if best is not None:
                return best
        return None

    def _astar(self, local_map: LocalOccupancyMap, start: GridIndex, goal: GridIndex) -> Optional[List[GridIndex]]:
        nx, ny = local_map.shape()
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        q: List[Tuple[float, GridIndex]] = [(0.0, start)]
        parent: Dict[GridIndex, Optional[GridIndex]] = {start: None}
        cost: Dict[GridIndex, float] = {start: 0.0}
        while q:
            _, cur = heapq.heappop(q)
            if cur == goal:
                path = [cur]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            cx, cy = cur
            for dx, dy, step in moves:
                nxt = (cx + dx, cy + dy)
                ix, iy = nxt
                if not (0 <= ix < nx and 0 <= iy < ny) or local_map.occupied[ix][iy]:
                    continue
                if dx != 0 and dy != 0 and (local_map.occupied[cx + dx][cy] or local_map.occupied[cx][cy + dy]):
                    continue
                bx, by = local_map.grid_to_body(nxt)
                clear = self._clearance_to_points((bx, by), local_map.points)
                penalty = max(0.0, self._safe_required_clearance() - clear) * 3.0
                new_cost = cost[cur] + step + penalty
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    heapq.heappush(q, (new_cost + math.hypot(ix - goal[0], iy - goal[1]), nxt))
                    parent[nxt] = cur
        return None

    def _line_free(self, local_map: LocalOccupancyMap, a: Optional[GridIndex], b: Optional[GridIndex]) -> bool:
        if a is None or b is None:
            return False
        ax, ay = a
        bx, by = b
        steps = max(abs(bx - ax), abs(by - ay), 1)
        for i in range(steps + 1):
            t = i / float(steps)
            ix = int(round(ax + (bx - ax) * t))
            iy = int(round(ay + (by - ay) * t))
            if local_map.occupied[ix][iy]:
                return False
        return True

    def _shortcut(self, local_map: LocalOccupancyMap, path: List[GridIndex]) -> List[GridIndex]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_free(local_map, path[i], path[j]):
                    body = [local_map.grid_to_body(path[i]), local_map.grid_to_body(path[j])]
                    if self._sampled_path_clearance(body, local_map.points) >= self._hard_required_clearance():
                        break
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _path_is_valid(self, local_map: LocalOccupancyMap, body_path: Sequence[BodyPoint]) -> bool:
        if len(body_path) < 2:
            return False
        cells = [local_map.body_to_grid(x, y) for x, y in body_path]
        if any(cell is None for cell in cells):
            return False
        for cell in cells:
            if local_map.occupied[cell[0]][cell[1]]:
                return False
        return self._sampled_path_clearance(body_path, local_map.points) >= self._hard_required_clearance()

    def _update_side_latch(self, points: Sequence[BodyPoint], target_body: BodyPoint, now: float, front_blocked: bool, corridor_blocked: bool) -> None:
        if abs(self._side_latch) > 1.0e-6:
            if now >= self._side_latch_until and not front_blocked and not corridor_blocked:
                self._side_latch = 0.0
            else:
                self._side_latch_until = max(self._side_latch_until, now + 0.5)
            return
        if front_blocked or corridor_blocked:
            self._side_latch = self._pick_side(points, target_body)
            self._side_latch_until = now + 3.0

    def _pick_side(self, points: Sequence[BodyPoint], target_body: BodyPoint) -> float:
        left = self._sector_clearance(points, 0.25, self.config.astar_grid_left, -0.1, 4.5)
        right = self._sector_clearance(points, -self.config.astar_grid_right, -0.25, -0.1, 4.5)
        target_side = 1.0 if target_body[1] >= 0.0 else -1.0
        if abs(left - right) < 0.35:
            return target_side
        return 1.0 if left > right else -1.0

    def _front_blocked(self, points: Sequence[BodyPoint]) -> bool:
        width = max(0.85, self.config.drone_radius + self.config.obstacle_margin + 0.25)
        return any(0.25 <= x <= 3.0 and abs(y) <= width for x, y in points)

    def _target_corridor_clear(self, points: Sequence[BodyPoint], target_body: BodyPoint) -> bool:
        tx, ty = target_body
        dist = math.hypot(tx, ty)
        if dist < 1.0e-6:
            return True
        ux, uy = tx / dist, ty / dist
        length = min(3.8, max(1.8, dist))
        half_width = max(0.9, self.config.drone_radius + self.config.obstacle_margin + 0.25)
        for x, y in points:
            along = x * ux + y * uy
            if 0.25 <= along <= length and abs(-uy * x + ux * y) <= half_width:
                return False
        return True

    def _sector_clearance(self, points: Sequence[BodyPoint], y_min: float, y_max: float, x_min: float, x_max: float) -> float:
        vals = [math.hypot(x, y) for x, y in points if x_min <= x <= x_max and y_min <= y <= y_max]
        return min(vals) if vals else 999.0

    @staticmethod
    def _clearance_to_points(point: BodyPoint, points: Sequence[BodyPoint]) -> float:
        if not points:
            return 999.0
        px, py = point
        return min(math.hypot(px - x, py - y) for x, y in points)

    def _sampled_path_clearance(self, body_path: Sequence[BodyPoint], points: Sequence[BodyPoint]) -> float:
        if not points:
            return 999.0
        if len(body_path) < 2:
            return self._clearance_to_points(body_path[0], points) if body_path else 999.0
        best = 999.0
        last = body_path[0]
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            steps = max(1, int(math.ceil(seg / 0.08)))
            for i in range(1, steps + 1):
                r = i / float(steps)
                sample = (last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1]))
                best = min(best, self._clearance_to_points(sample, points))
            last = point
        return best

    def _hard_required_clearance(self) -> float:
        raw = self.config.drone_radius + self.config.obstacle_margin + self.config.hard_clearance + 0.18
        return max(1.22, raw)

    def _safe_required_clearance(self) -> float:
        raw = self.config.drone_radius + self.config.obstacle_margin + max(0.60, self.config.emergency_clearance) + 0.18
        return max(1.55, raw)

    @staticmethod
    def _world_to_body(dx: float, dy: float, heading: float) -> BodyPoint:
        c, s = math.cos(heading), math.sin(heading)
        return c * dx + s * dy, -s * dx + c * dy

    @staticmethod
    def _body_to_world(bx: float, by: float, heading: float) -> BodyPoint:
        c, s = math.cos(heading), math.sin(heading)
        return c * bx - s * by, s * bx + c * by

    def _body_point_to_world(self, state: PlannerState, bx: float, by: float) -> Vec3:
        wx, wy = self._body_to_world(bx, by, state.heading)
        return Vec3(state.position.x + wx, state.position.y + wy, state.position.z)

    def _world_point_to_body(self, state: PlannerState, point: Vec3) -> BodyPoint:
        return self._world_to_body(point.x - state.position.x, point.y - state.position.y, state.heading)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _committed_body_path(self, state: PlannerState) -> List[BodyPoint]:
        return [self._world_point_to_body(state, p) for p in self._committed_world_path]

    def _trim_passed_points(self, body_path: List[BodyPoint]) -> List[BodyPoint]:
        if len(body_path) <= 2:
            return body_path
        idx = min(range(len(body_path)), key=lambda i: math.hypot(body_path[i][0], body_path[i][1]))
        idx = min(idx, len(body_path) - 2)
        if idx > 0:
            self._committed_world_path = self._committed_world_path[idx:]
            return body_path[idx:]
        return body_path

    def _lookahead(self, body_path: List[BodyPoint]) -> BodyPoint:
        last = body_path[0]
        acc = 0.0
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            if acc + seg >= self.config.astar_lookahead_dist:
                r = (self.config.astar_lookahead_dist - acc) / max(seg, 1.0e-6)
                return last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1])
            acc += seg
            last = point
        return body_path[-1]

    def _smooth_cmd_body(self, bx: float, by: float) -> BodyPoint:
        ox, oy = self._last_cmd_body
        alpha = max(0.0, min(1.0, self.config.output_alpha))
        sx, sy = alpha * bx + (1.0 - alpha) * ox, alpha * by + (1.0 - alpha) * oy
        mag = math.hypot(sx, sy)
        if mag > self.config.max_speed:
            sx *= self.config.max_speed / mag
            sy *= self.config.max_speed / mag
        self._last_cmd_body = (sx, sy)
        return sx, sy

    @staticmethod
    def _path_length(path: Sequence[GridIndex]) -> float:
        total = 0.0
        last = path[0]
        for cur in path[1:]:
            total += math.hypot(cur[0] - last[0], cur[1] - last[1])
            last = cur
        return total

    def _path_rollout(self, name: str, velocity: Vec3, state: PlannerState, body_path: Sequence[BodyPoint], speed: float, metadata: Dict) -> CandidateTrajectory:
        points = [TrajectoryPoint(t=0.0, position=state.position)]
        steps = max(1, int(round(self.config.horizon / max(self.config.dt, 1.0e-6))))
        total = self._body_path_length(body_path)
        for idx in range(1, steps + 1):
            t = idx * self.config.dt
            bx, by = self._point_at_body_distance(body_path, min(total, max(0.0, speed) * t))
            wx, wy = self._body_to_world(bx, by, state.heading)
            points.append(TrajectoryPoint(t=t, position=Vec3(state.position.x + wx, state.position.y + wy, state.position.z + velocity.z * t)))
        return CandidateTrajectory(name=name, velocity=velocity, points=points, metadata=metadata)

    @staticmethod
    def _body_path_length(body_path: Sequence[BodyPoint]) -> float:
        total = 0.0
        last = body_path[0]
        for point in body_path[1:]:
            total += math.hypot(point[0] - last[0], point[1] - last[1])
            last = point
        return total

    @staticmethod
    def _point_at_body_distance(body_path: Sequence[BodyPoint], distance: float) -> BodyPoint:
        if not body_path:
            return 0.0, 0.0
        remaining = distance
        last = body_path[0]
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            if seg <= 1.0e-6:
                last = point
                continue
            if remaining <= seg:
                r = remaining / seg
                return last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1])
            remaining -= seg
            last = point
        return body_path[-1]
