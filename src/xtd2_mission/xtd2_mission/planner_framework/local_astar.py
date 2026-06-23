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
        self._recovery_until = 0.0
        self._recovery_side = 1.0
        self._last_cmd_body: BodyPoint = (0.0, 0.0)
        self._current_position = Vec3()
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
        target_dist = state.position.distance_to(local_goal)
        stalled_sec = self._update_progress(now, target_dist)
        nearest = min((math.hypot(x, y) for x, y in local_map.points), default=999.0)
        front_blocked = self._front_blocked(local_map.points)
        recovery_allowed = front_blocked and nearest < max(2.2, self.config.astar_latch_clearance + 1.0)
        if stalled_sec > self.config.astar_progress_stall_sec and not recovery_allowed:
            self._last_progress_dist = target_dist
            self._last_progress_position = state.position
            self._last_progress_time = now
            stalled_sec = 0.0
        if stalled_sec > self.config.astar_progress_stall_sec and recovery_allowed and now > self._recovery_until:
            self._recovery_side = self._pick_escape_side(local_map.points, target_body)
            self._recovery_until = now + self.config.astar_recovery_sec
            self._committed_world_path = []

        if now < self._recovery_until and not recovery_allowed:
            self._recovery_until = 0.0
        if now < self._recovery_until:
            return self._recovery_candidate(state, target_body, local_map, now, stalled_sec)

        replan_reason = self._replan_reason(now, state, local_goal, local_map)
        if replan_reason:
            selected = self._select_path(local_map, start, target_body)
            if selected is not None:
                self._commit_path(state, selected, local_goal, now)
            elif not self._committed_world_path:
                self.diagnostics = {
                    "status": "no_astar_path",
                    "replan_reason": replan_reason,
                    "raw_points": local_map.raw_count,
                    "stalled_sec": stalled_sec,
                    "nearest": nearest,
                    "front_blocked": front_blocked,
                    "recovery_allowed": recovery_allowed,
                }
                return None

        body_path = self._committed_body_path(state)
        if len(body_path) < 2 or not self._path_is_valid(local_map, body_path):
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self._committed_world_path = []
                self.diagnostics = {
                    "status": "committed_path_invalid",
                    "replan_reason": "invalid_path_no_replacement",
                    "raw_points": local_map.raw_count,
                    "stalled_sec": stalled_sec,
                    "nearest": nearest,
                    "front_blocked": front_blocked,
                    "recovery_allowed": recovery_allowed,
                }
                return None
            self._commit_path(state, selected, local_goal, now)
            body_path = self._committed_body_path(state)

        body_path = self._trim_passed_points(body_path)
        look_x, look_y = self._lookahead(body_path)
        if look_x < 0.18:
            look_x = 0.18
            if abs(look_y) < 0.18:
                look_y = 0.18 if target_body[1] >= 0.0 else -0.18
        look_norm = max(math.hypot(look_x, look_y), 1.0e-6)
        speed = min(self.config.max_speed, self.config.cruise_speed)
        if nearest < 1.2:
            speed = min(speed, max(0.16, 0.34 * self.config.max_speed))
        bx = speed * look_x / look_norm
        by = speed * look_y / look_norm
        bx, by = self._smooth_cmd_body(bx, by)
        vx, vy = self._body_to_world(bx, by, state.heading)
        dz = local_goal.z - state.position.z
        vz = max(-0.08, min(0.08, 0.35 * dz))
        if abs(dz) < 0.25:
            vz = 0.0
        velocity = Vec3(vx, vy, vz).limit_norm(self.config.max_speed)
        min_path_clearance = self._sampled_path_clearance(body_path, local_map.points)
        self.diagnostics = {
            "status": "tracking_committed_path",
            "label": self._committed_label,
            "replan_reason": replan_reason or "latched",
            "committed_left_sec": max(0.0, self._committed_until - now),
            "stalled_sec": stalled_sec,
            "raw_points": local_map.raw_count,
            "path_len": len(body_path),
            "min_path_clearance": min_path_clearance,
            "lookahead_body": {"x": look_x, "y": look_y},
            "nearest": nearest,
            "front_blocked": front_blocked,
            "recovery_allowed": recovery_allowed,
        }
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
                "recovery_allowed": recovery_allowed,
            },
        )

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

    def _replan_reason(self, now: float, state: PlannerState, local_goal: Vec3, local_map: LocalOccupancyMap) -> str:
        if not self._committed_world_path:
            return "no_committed_path"
        if now >= self._committed_until:
            return "path_latch_expired"
        if now - self._last_replan_time >= self.config.astar_replan_interval:
            body_path = self._committed_body_path(state)
            if not self._path_is_valid(local_map, body_path):
                return "committed_path_blocked"
        if state.position.distance_to(self._last_goal) < self.config.local_goal_reached_radius:
            return "local_goal_reached"
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
        sample_clearance = self._sampled_path_clearance(body_path, local_map.points)
        return sample_clearance >= max(self.config.astar_latch_clearance, self._hard_required_clearance())

    def _pick_escape_side(self, points: Sequence[BodyPoint], target_body: BodyPoint) -> float:
        left_clear = self._sector_clearance(points, 0.25, self.config.astar_grid_left)
        right_clear = self._sector_clearance(points, -self.config.astar_grid_right, -0.25)
        target_side = 1.0 if target_body[1] >= 0.0 else -1.0
        if abs(left_clear - right_clear) < 0.25:
            return target_side
        return 1.0 if left_clear > right_clear else -1.0

    def _recovery_candidate(self, state: PlannerState, target_body: BodyPoint, local_map: LocalOccupancyMap, now: float, stalled_sec: float) -> CandidateTrajectory:
        left_clear = self._sector_clearance(local_map.points, 0.25, self.config.astar_grid_left)
        right_clear = self._sector_clearance(local_map.points, -self.config.astar_grid_right, -0.25)
        back_clear = self._sector_clearance(local_map.points, -0.9, 0.9, x_min=-self.config.astar_grid_back, x_max=-0.2)
        side = self._recovery_side or self._pick_escape_side(local_map.points, target_body)
        lateral = 0.34 * side
        forward = 0.18 if target_body[0] >= -0.2 else 0.0
        if max(left_clear, right_clear) < 0.75:
            lateral = 0.22 * side
            forward = 0.12 if target_body[0] >= -0.2 else 0.0
        bx, by = self._smooth_cmd_body(forward, lateral)
        vx, vy = self._body_to_world(bx, by, state.heading)
        velocity = Vec3(vx, vy, 0.0).limit_norm(self.config.max_speed)
        label = "recovery_left" if side > 0.0 else "recovery_right"
        self.diagnostics = {
            "status": "recovery_escape",
            "label": label,
            "recovery_left_sec": max(0.0, self._recovery_until - now),
            "stalled_sec": stalled_sec,
            "left_clear": left_clear,
            "right_clear": right_clear,
            "back_clear": back_clear,
            "raw_points": local_map.raw_count,
        }
        return self._rollout(
            f"local_astar:{label}",
            velocity,
            state.position,
            {"frontend": "local_astar", "recovery": True, "stalled_sec": stalled_sec, "raw_points": local_map.raw_count},
        )

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
        if self._front_blocked(local_map.points):
            angle_offsets_deg = [30, -30, 45, -45, 60, -60, 75, -75, 18, -18, 0]
        else:
            angle_offsets_deg = [0, 18, -18, 30, -30, 45, -45, 60, -60, 75, -75]
        radii = [self.config.astar_local_goal_dist, 3.6, 2.8, 2.1]
        for deg in angle_offsets_deg:
            for radius in radii:
                ang = target_angle + math.radians(deg)
                x = max(-self.config.astar_grid_back + 0.4, min(self.config.astar_grid_forward - 0.4, math.cos(ang) * radius))
                y = max(-self.config.astar_grid_right + 0.4, min(self.config.astar_grid_left - 0.4, math.sin(ang) * radius))
                cell = local_map.body_to_grid(x, y)
                if cell is None:
                    continue
                free = self._nearest_free(local_map, cell, max_r=8)
                if free is None or free in seen:
                    continue
                bx, by = local_map.grid_to_body(free)
                if math.hypot(bx, by) < 1.0:
                    continue
                if bx < 0.15:
                    continue
                seen.add(free)
                candidates.append((free, f"{deg:+d}deg/{radius:.1f}m"))
        return candidates

    def _nearest_free(self, local_map: LocalOccupancyMap, preferred: GridIndex, max_r: int = 12) -> Optional[GridIndex]:
        nx, ny = local_map.shape()
        px, py = preferred
        if 0 <= px < nx and 0 <= py < ny and not local_map.occupied[px][py]:
            return preferred
        for r in range(1, max_r + 1):
            best = None
            best_score = 1e9
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    ix, iy = px + dx, py + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not local_map.occupied[ix][iy]:
                        score = dx * dx + dy * dy
                        if score < best_score:
                            best, best_score = (ix, iy), score
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    ix, iy = px + dx, py + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not local_map.occupied[ix][iy]:
                        score = dx * dx + dy * dy
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
                new_cost = cost_so_far[current] + step_cost
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
        clearance_cost = 0.0 if min_clearance >= 1.4 else (1.4 - min_clearance) * 3.0
        score = length + 2.0 * angle_cost - 0.65 * progress + clearance_cost
        if self._front_blocked(local_map.points) and abs(end_y) < 1.0:
            score += 8.0
        required_clearance = self._hard_required_clearance()
        if min_clearance < required_clearance:
            score += (required_clearance - min_clearance) * 18.0
        return {
            "score": score,
            "path": path,
            "body_path": body_path,
            "label": label,
            "min_path_clearance": min_clearance,
        }

    def _select_path(self, local_map: LocalOccupancyMap, start: GridIndex, target_body: BodyPoint) -> Optional[Dict[str, object]]:
        scored: List[Dict[str, object]] = []
        for goal, label in self._candidate_goals(local_map, target_body):
            path = self._astar(local_map, start, goal)
            if path is None:
                continue
            smooth = self._shortcut(local_map, path)
            if len(smooth) >= 2:
                scored.append(self._score_path(local_map, smooth, label, target_body))
        if not scored:
            return None
        scored.sort(key=lambda item: float(item["score"]))
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
        sample_step = max(0.08, 0.5 * self.config.astar_resolution)
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
        lidar_point_radius = 0.16
        return self.config.drone_radius + self.config.obstacle_margin + self.config.hard_clearance + lidar_point_radius

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
        front_depth = min(3.0, max(1.6, self.config.astar_lookahead_dist + 1.0))
        return any(0.25 <= x <= front_depth and abs(y) <= front_width for x, y in points)

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
