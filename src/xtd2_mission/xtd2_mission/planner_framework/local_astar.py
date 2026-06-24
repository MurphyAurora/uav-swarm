#!/usr/bin/env python3
"""Local A* frontend backend for body-frame LiDAR/static obstacle maps."""

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
        self._last_progress_dist: Optional[float] = None
        self._last_progress_position: Optional[Vec3] = None
        self._last_progress_time = time.time()
        self._last_cmd_body: BodyPoint = (0.0, 0.0)
        self._current_position = Vec3()
        self._side_latch = 0.0
        self._side_latch_until = 0.0
        self._last_candidate_count = 0
        self._last_scored_count = 0
        self._last_best_clearance = 999.0
        self.diagnostics: Dict[str, object] = {}

    def plan_candidate(self, state: PlannerState, local_goal: Vec3, obstacles) -> Optional[CandidateTrajectory]:
        now = time.time()
        self._current_position = state.position
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

        target_dist = state.position.distance_to(local_goal)
        stalled_sec = self._update_progress(now, target_dist)

        replan_reason = self._replan_reason(now, state, local_goal, local_map, corridor_blocked)
        if replan_reason:
            selected = self._select_path(local_map, start, target_body)
            if selected is not None:
                self._commit_path(state, selected, local_goal, now)
            elif not self._committed_world_path:
                self.diagnostics = self._diag(
                    "no_astar_path", now, replan_reason, local_map, stalled_sec, nearest,
                    front_blocked, corridor_blocked, None
                )
                return None

        body_path = self._committed_body_path(state)
        if len(body_path) < 2 or not self._path_is_valid(local_map, body_path):
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self._committed_world_path = []
                self.diagnostics = self._diag(
                    "committed_path_invalid", now, "invalid_path_no_replacement", local_map,
                    stalled_sec, nearest, front_blocked, corridor_blocked, None
                )
                return None
            self._commit_path(state, selected, local_goal, now)
            body_path = self._committed_body_path(state)

        body_path = self._trim_passed_points(body_path)
        look_x, look_y = self._lookahead(body_path)
        if look_x < 0.18:
            look_x = 0.18
            if abs(look_y) < 0.18:
                look_y = 0.18 if (self._side_latch or target_body[1]) >= 0.0 else -0.18

        min_path_clearance = self._sampled_path_clearance(body_path, local_map.points)
        speed = min(self.config.max_speed, self.config.cruise_speed)
        if min_path_clearance < self._safe_required_clearance():
            speed = min(speed, 0.22)
        if nearest < 1.8:
            speed = min(speed, 0.20)
        if stalled_sec > self.config.astar_progress_stall_sec and min_path_clearance < self._safe_required_clearance():
            speed = min(speed, 0.16)

        look_norm = max(math.hypot(look_x, look_y), 1.0e-6)
        bx = speed * look_x / look_norm
        by = speed * look_y / look_norm
        bx, by = self._smooth_cmd_body(bx, by)
        vx, vy = self._body_to_world(bx, by, state.heading)
        dz = local_goal.z - state.position.z
        vz = max(-0.08, min(0.08, 0.35 * dz))
        if abs(dz) < 0.25:
            vz = 0.0
        velocity = Vec3(vx, vy, vz).limit_norm(self.config.max_speed)

        self.diagnostics = self._diag(
            "tracking_committed_path", now, replan_reason or "latched", local_map, stalled_sec,
            nearest, front_blocked, corridor_blocked,
            {
                "label": self._committed_label,
                "committed_left_sec": max(0.0, self._committed_until - now),
                "path_len": len(body_path),
                "min_path_clearance": float(min_path_clearance),
                "lookahead_body": {"x": look_x, "y": look_y},
                "speed": float(speed),
            },
        )
        return self._path_rollout(
            f"local_astar:{self._committed_label}",
            velocity,
            state,
            body_path,
            speed,
            {
                "frontend": "local_astar",
                "committed": True,
                "path_len": len(body_path),
                "smooth_len": len(body_path),
                "raw_points": local_map.raw_count,
                "min_path_clearance": float(min_path_clearance),
                "lookahead_body": {"x": look_x, "y": look_y},
                "replan_reason": replan_reason or "latched",
                "stalled_sec": stalled_sec,
                "nearest": nearest,
                "front_blocked": front_blocked,
                "corridor_blocked": corridor_blocked,
                "side_latch": self._side_latch,
            },
        )

    def _diag(self, status: str, now: float, reason: str, local_map: LocalOccupancyMap, stalled_sec: float,
              nearest: float, front_blocked: bool, corridor_blocked: bool, extra: Optional[Dict]) -> Dict[str, object]:
        out: Dict[str, object] = {
            "status": status,
            "replan_reason": reason,
            "raw_points": local_map.raw_count,
            "stalled_sec": stalled_sec,
            "nearest": nearest,
            "front_blocked": front_blocked,
            "corridor_blocked": corridor_blocked,
            "side_latch": self._side_latch,
            "side_latch_left_sec": max(0.0, self._side_latch_until - now),
            "candidate_goals": self._last_candidate_count,
            "scored_paths": self._last_scored_count,
            "best_path_clearance": self._last_best_clearance,
            "hard_required_clearance": self._hard_required_clearance(),
            "safe_required_clearance": self._safe_required_clearance(),
        }
        if extra:
            out.update(extra)
        return out

    def _update_progress(self, now: float, target_dist: float) -> float:
        moved = 0.0
        if self._last_progress_position is not None:
            moved = self._last_progress_position.distance_to(self._current_position)
        if (
            self._last_progress_dist is None
            or target_dist < self._last_progress_dist - self.config.astar_progress_epsilon
            or moved > self.config.astar_progress_move_epsilon
        ):
            self._last_progress_dist = target_dist
            self._last_progress_position = self._current_position
            self._last_progress_time = now
        return max(0.0, now - self._last_progress_time)

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
            if self._sampled_path_clearance(body_path, local_map.points) < self._hard_required_clearance():
                return "committed_path_low_clearance"
        if state.position.distance_to(self._last_goal) < self.config.local_goal_reached_radius:
            return "local_goal_reached"
        if local_goal.distance_to(self._last_goal) > max(1.0, 0.5 * self.config.astar_local_goal_dist):
            return "local_goal_changed"
        if corridor_blocked and now - self._last_replan_time >= 0.5 * self.config.astar_replan_interval:
            return "corridor_blocked_refresh"
        return ""

    def _commit_path(self, state: PlannerState, selected: Dict[str, object], local_goal: Vec3, now: float) -> None:
        body_path = list(selected["body_path"])
        self._committed_world_path = [self._body_point_to_world(state, x, y) for x, y in body_path]
        self._committed_label = str(selected["label"])
        self._committed_until = now + self.config.astar_path_latch_sec
        self._last_replan_time = now
        self._last_goal = local_goal
        self._last_best_clearance = float(selected.get("min_path_clearance", 999.0))

    def _committed_body_path(self, state: PlannerState) -> List[BodyPoint]:
        return [self._world_point_to_body(state, point) for point in self._committed_world_path]

    def _trim_passed_points(self, body_path: List[BodyPoint]) -> List[BodyPoint]:
        if len(body_path) <= 2:
            return body_path
        best_idx = min(range(len(body_path)), key=lambda idx: math.hypot(body_path[idx][0], body_path[idx][1]))
        best_idx = min(best_idx, len(body_path) - 2)
        if best_idx > 0:
            self._committed_world_path = self._committed_world_path[best_idx:]
            return body_path[best_idx:]
        return body_path

    def _path_is_valid(self, local_map: LocalOccupancyMap, body_path: Sequence[BodyPoint]) -> bool:
        if len(body_path) < 2:
            return False
        cells: List[GridIndex] = []
        for x, y in body_path:
            cell = local_map.body_to_grid(x, y)
            if cell is None:
                return False
            cells.append(cell)
        for cell in cells:
            if local_map.occupied[cell[0]][cell[1]]:
                return False
        if not self._line_free(local_map, local_map.body_to_grid(0.0, 0.0), cells[min(1, len(cells) - 1)]):
            return False
        return self._sampled_path_clearance(body_path, local_map.points) >= self._hard_required_clearance()

    def _update_side_latch(self, points: Sequence[BodyPoint], target_body: BodyPoint, now: float,
                           front_blocked: bool, corridor_blocked: bool) -> None:
        if abs(self._side_latch) > 1.0e-6:
            if now >= self._side_latch_until and not front_blocked and not corridor_blocked:
                self._side_latch = 0.0
                self._side_latch_until = 0.0
            else:
                self._side_latch_until = max(self._side_latch_until, now + 0.5)
            return
        if front_blocked or corridor_blocked:
            self._side_latch = self._pick_escape_side(points, target_body)
            self._side_latch_until = now + max(2.8, self.config.astar_path_latch_sec)

    def _pick_escape_side(self, points: Sequence[BodyPoint], target_body: BodyPoint) -> float:
        left_clear = self._sector_clearance(points, 0.25, self.config.astar_grid_left, x_min=-0.1, x_max=4.5)
        right_clear = self._sector_clearance(points, -self.config.astar_grid_right, -0.25, x_min=-0.1, x_max=4.5)
        target_side = 1.0 if target_body[1] >= 0.0 else -1.0
        if abs(left_clear - right_clear) < 0.30:
            return target_side
        return 1.0 if left_clear > right_clear else -1.0

    @staticmethod
    def _world_to_body(dx: float, dy: float, heading: float) -> BodyPoint:
        c = math.cos(heading)
        s = math.sin(heading)
        return c * dx + s * dy, -s * dx + c * dy

    @staticmethod
    def _body_to_world(bx: float, by: float, heading: float) -> BodyPoint:
        c = math.cos(heading)
        s = math.sin(heading)
        return c * bx - s * by, s * bx + c * by

    def _body_point_to_world(self, state: PlannerState, bx: float, by: float) -> Vec3:
        wx, wy = self._body_to_world(bx, by, state.heading)
        return Vec3(state.position.x + wx, state.position.y + wy, state.position.z)

    def _world_point_to_body(self, state: PlannerState, point: Vec3) -> BodyPoint:
        return self._world_to_body(point.x - state.position.x, point.y - state.position.y, state.heading)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def _candidate_goals(self, local_map: LocalOccupancyMap, target_body: BodyPoint) -> List[Tuple[GridIndex, str]]:
        tx, ty = target_body
        target_angle = math.atan2(ty, tx)
        candidates: List[Tuple[GridIndex, str]] = []
        seen = set()
        front_blocked = self._front_blocked(local_map.points)
        if abs(self._side_latch) > 1.0e-6:
            base_angles = [35, 50, 65, 80, 95, 115, 25]
            angle_offsets_deg = [int(self._side_latch * deg) for deg in base_angles]
        elif front_blocked or not self._target_corridor_clear(local_map.points, target_body):
            side = self._pick_escape_side(local_map.points, target_body)
            primary = [35, 50, 65, 80, 95, 115, 25]
            secondary = [45, 65]
            angle_offsets_deg = [int(side * deg) for deg in primary] + [int(-side * deg) for deg in secondary]
        else:
            angle_offsets_deg = [0, 18, -18, 30, -30, 45, -45, 60, -60]
        radii = [self.config.astar_local_goal_dist, 4.5, 3.6, 2.8, 2.1]
        for deg in angle_offsets_deg:
            for radius in radii:
                ang = target_angle + math.radians(deg)
                x = max(-self.config.astar_grid_back + 0.4, min(self.config.astar_grid_forward - 0.4, math.cos(ang) * radius))
                y = max(-self.config.astar_grid_right + 0.4, min(self.config.astar_grid_left - 0.4, math.sin(ang) * radius))
                if (front_blocked or abs(self._side_latch) > 1.0e-6) and abs(y) < 0.7 and x > 1.0:
                    continue
                if abs(self._side_latch) > 1.0e-6 and y * self._side_latch < 0.15 and x > 0.5:
                    continue
                cell = local_map.body_to_grid(x, y)
                if cell is None:
                    continue
                free = self._nearest_free(local_map, cell, max_r=10)
                if free is None or free in seen:
                    continue
                bx, by = local_map.grid_to_body(free)
                if math.hypot(bx, by) < 1.2 or bx < 0.15:
                    continue
                seen.add(free)
                candidates.append((free, f"{deg:+d}deg/{radius:.1f}m"))
        self._last_candidate_count = len(candidates)
        return candidates

    def _nearest_free(self, local_map: LocalOccupancyMap, preferred: GridIndex, max_r: int = 12) -> Optional[GridIndex]:
        nx, ny = local_map.shape()
        px, py = preferred
        if 0 <= px < nx and 0 <= py < ny and not local_map.occupied[px][py]:
            return preferred
        best = None
        best_score = 1e9
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    ix, iy = px + dx, py + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not local_map.occupied[ix][iy]:
                        bx, by = local_map.grid_to_body((ix, iy))
                        clearance = self._clearance_to_points((bx, by), local_map.points)
                        score = dx * dx + dy * dy - 0.4 * clearance
                        if score < best_score:
                            best, best_score = (ix, iy), score
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    ix, iy = px + dx, py + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not local_map.occupied[ix][iy]:
                        bx, by = local_map.grid_to_body((ix, iy))
                        clearance = self._clearance_to_points((bx, by), local_map.points)
                        score = dx * dx + dy * dy - 0.4 * clearance
                        if score < best_score:
                            best, best_score = (ix, iy), score
            if best is not None:
                return best
        return None

    def _astar(self, local_map: LocalOccupancyMap, start: GridIndex, goal: GridIndex) -> Optional[List[GridIndex]]:
        nx, ny = local_map.shape()
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]
        frontier: List[Tuple[float, GridIndex]] = [(0.0, start)]
        came_from: Dict[GridIndex, Optional[GridIndex]] = {start: None}
        cost_so_far: Dict[GridIndex, float] = {start: 0.0}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                path = [current]
                while came_from[path[-1]] is not None:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            cx, cy = current
            for dx, dy, step_cost in neighbors:
                nxt = (cx + dx, cy + dy)
                nx_i, ny_i = nxt
                if not (0 <= nx_i < nx and 0 <= ny_i < ny):
                    continue
                if local_map.occupied[nx_i][ny_i]:
                    continue
                if dx != 0 and dy != 0:
                    if local_map.occupied[cx + dx][cy] or local_map.occupied[cx][cy + dy]:
                        continue
                bx, by = local_map.grid_to_body(nxt)
                clearance = self._clearance_to_points((bx, by), local_map.points)
                clearance_penalty = 0.0
                hard_required = self._hard_required_clearance()
                if clearance < hard_required:
                    clearance_penalty += (hard_required - clearance) * 12.0
                elif clearance < self._safe_required_clearance():
                    clearance_penalty += (self._safe_required_clearance() - clearance) * 2.0
                new_cost = cost_so_far[current] + step_cost + clearance_penalty
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + math.hypot(nxt[0] - goal[0], nxt[1] - goal[1])
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current
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

    def _score_path(self, local_map: LocalOccupancyMap, path: List[GridIndex], label: str, target_body: BodyPoint) -> Dict[str, object]:
        body_path = [local_map.grid_to_body(cell) for cell in path]
        end_x, end_y = body_path[-1]
        target_angle = math.atan2(target_body[1], target_body[0])
        end_angle = math.atan2(end_y, end_x)
        progress = end_x * math.cos(target_angle) + end_y * math.sin(target_angle)
        angle_cost = abs(self._angle_diff(end_angle, target_angle))
        length = self._path_length(path)
        min_clearance = self._sampled_path_clearance(body_path, local_map.points)
        hard_required = self._hard_required_clearance()
        safe_required = self._safe_required_clearance()
        clearance_cost = max(0.0, safe_required - min_clearance) * 7.0
        score = length + 1.6 * angle_cost - 0.55 * progress + clearance_cost
        if self._front_blocked(local_map.points) and abs(end_y) < 1.0:
            score += 10.0
        if abs(self._side_latch) > 1.0e-6:
            if end_y * self._side_latch < 0.2 and end_x > 0.4:
                score += 30.0
            elif end_y * self._side_latch > 0.7:
                score -= 2.0
        if min_clearance < hard_required:
            score += (hard_required - min_clearance) * 100.0
        return {
            "score": score,
            "path": path,
            "body_path": body_path,
            "label": label,
            "min_path_clearance": min_clearance,
            "safe_class": min_clearance >= safe_required,
        }

    def _select_path(self, local_map: LocalOccupancyMap, start: GridIndex, target_body: BodyPoint) -> Optional[Dict[str, object]]:
        scored: List[Dict[str, object]] = []
        for goal, label in self._candidate_goals(local_map, target_body):
            path = self._astar(local_map, start, goal)
            if path is None:
                continue
            smooth = self._shortcut(local_map, path)
            if len(smooth) < 2:
                continue
            item = self._score_path(local_map, smooth, label, target_body)
            if float(item["min_path_clearance"]) >= self._hard_required_clearance():
                scored.append(item)
        self._last_scored_count = len(scored)
        if not scored:
            return None
        scored.sort(key=lambda item: (not bool(item["safe_class"]), float(item["score"]), -float(item["min_path_clearance"])))
        self._last_best_clearance = float(scored[0]["min_path_clearance"])
        return scored[0]

    def _path_length(self, path: Sequence[GridIndex]) -> float:
        length = 0.0
        last = path[0]
        for cur in path[1:]:
            length += math.hypot(cur[0] - last[0], cur[1] - last[1]) * self.config.astar_resolution
            last = cur
        return length

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
        min_clearance = 999.0
        sample_step = max(0.06, 0.4 * self.config.astar_resolution)
        last = body_path[0]
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            steps = max(1, int(math.ceil(seg / sample_step)))
            for idx in range(1, steps + 1):
                r = idx / float(steps)
                sample = (last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1]))
                min_clearance = min(min_clearance, self._clearance_to_points(sample, points))
            last = point
        return min_clearance

    def _hard_required_clearance(self) -> float:
        lidar_point_radius = 0.18
        # Keep the A* acceptance threshold consistent with the occupancy grid.
        # If this is much larger than astar_inflation_radius, narrow forest
        # gaps are planned as free cells and then rejected after search.
        return max(
            self.config.astar_inflation_radius + 0.25,
            self.config.drone_radius + self.config.hard_clearance + lidar_point_radius,
        )

    def _safe_required_clearance(self) -> float:
        lidar_point_radius = 0.18
        return max(
            self._hard_required_clearance() + 0.25,
            self.config.drone_radius + min(self.config.emergency_clearance, 0.55) + lidar_point_radius,
        )

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

    def _sector_clearance(self, points: Sequence[BodyPoint], y_min: float, y_max: float, x_min: float = -0.3, x_max: float = 2.8) -> float:
        values = [math.hypot(x, y) for x, y in points if x_min <= x <= x_max and y_min <= y <= y_max]
        return min(values) if values else 999.0

    def _front_blocked(self, points: Sequence[BodyPoint]) -> bool:
        front_width = max(0.75, self.config.drone_radius + self.config.obstacle_margin + 0.20)
        front_depth = min(3.2, max(1.8, self.config.astar_lookahead_dist + 1.2))
        return any(0.25 <= x <= front_depth and abs(y) <= front_width for x, y in points)

    def _target_corridor_clear(self, points: Sequence[BodyPoint], target_body: BodyPoint) -> bool:
        tx, ty = target_body
        dist = math.hypot(tx, ty)
        if dist < 1.0e-6:
            return True
        ux = tx / dist
        uy = ty / dist
        length = min(3.8, max(1.8, dist))
        half_width = max(0.85, self.config.drone_radius + self.config.obstacle_margin + 0.20)
        for x, y in points:
            along = x * ux + y * uy
            if along < 0.25 or along > length:
                continue
            lateral = abs(-uy * x + ux * y)
            if lateral <= half_width:
                return False
        return True

    def _smooth_cmd_body(self, bx: float, by: float) -> BodyPoint:
        old_x, old_y = self._last_cmd_body
        alpha = max(0.0, min(1.0, self.config.output_alpha))
        sx = alpha * bx + (1.0 - alpha) * old_x
        sy = alpha * by + (1.0 - alpha) * old_y
        mag = math.hypot(sx, sy)
        if mag > self.config.max_speed:
            sx *= self.config.max_speed / mag
            sy *= self.config.max_speed / mag
        self._last_cmd_body = (sx, sy)
        return sx, sy

    def _rollout(self, name: str, velocity: Vec3, start: Vec3, metadata: Dict) -> CandidateTrajectory:
        points = [TrajectoryPoint(t=0.0, position=start)]
        steps = max(1, int(round(self.config.horizon / max(self.config.dt, 1.0e-6))))
        for idx in range(1, steps + 1):
            t = idx * self.config.dt
            points.append(TrajectoryPoint(t=t, position=start + velocity * t))
        return CandidateTrajectory(name=name, velocity=velocity, points=points, metadata=metadata)

    def _path_rollout(self, name: str, velocity: Vec3, state: PlannerState, body_path: Sequence[BodyPoint], speed: float, metadata: Dict) -> CandidateTrajectory:
        points = [TrajectoryPoint(t=0.0, position=state.position)]
        steps = max(1, int(round(self.config.horizon / max(self.config.dt, 1.0e-6))))
        total_len = self._body_path_length(body_path)
        for idx in range(1, steps + 1):
            t = idx * self.config.dt
            dist = min(total_len, max(0.0, speed) * t)
            bx, by = self._point_at_body_distance(body_path, dist)
            wx, wy = self._body_to_world(bx, by, state.heading)
            points.append(
                TrajectoryPoint(
                    t=t,
                    position=Vec3(
                        state.position.x + wx,
                        state.position.y + wy,
                        state.position.z + velocity.z * t,
                    ),
                )
            )
        return CandidateTrajectory(name=name, velocity=velocity, points=points, metadata=metadata)

    @staticmethod
    def _body_path_length(body_path: Sequence[BodyPoint]) -> float:
        if len(body_path) < 2:
            return 0.0
        length = 0.0
        last = body_path[0]
        for point in body_path[1:]:
            length += math.hypot(point[0] - last[0], point[1] - last[1])
            last = point
        return length

    @staticmethod
    def _point_at_body_distance(body_path: Sequence[BodyPoint], distance: float) -> BodyPoint:
        if not body_path:
            return 0.0, 0.0
        if len(body_path) == 1 or distance <= 0.0:
            return body_path[0]
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
