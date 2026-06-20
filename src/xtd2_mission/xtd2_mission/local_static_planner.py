#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-UAV static local planner for forest3 LiDAR validation.

This is a standalone local planner node, not another patch inside
multi_waypoint2.py.  It implements the minimum complete local-planner loop:

  LiDAR/state freshness check
  -> 2.5D local occupancy grid with obstacle inflation
  -> multiple candidate local goals
  -> local A* for each candidate
  -> path scoring and safety validation
  -> lookahead velocity tracking
  -> near-field escape and stuck recovery
  -> /cmd_vel_ned output

Scope:
  - one UAV, static obstacles only;
  - LiDAR point cloud assumed in the UAV body/local frame;
  - NED velocity command output for xtdrone2/PX4 offboard;
  - no dynamic-obstacle prediction and no multi-UAV coordination yet.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

GridIndex = Tuple[int, int]
BodyPoint = Tuple[float, float]


class LocalStaticPlanner(Node):
    def __init__(
        self,
        drone_id: int = 1,
        state_topic: str = '/xtdrone2/swarm/state_exchange',
        lidar_topic: Optional[str] = None,
        cmd_topic: Optional[str] = None,
        target_x: float = 19.0,
        target_y: float = 30.0,
        target_z: float = -3.0,
        duration: float = 180.0,
        rate_hz: float = 10.0,
        grid_forward: float = 7.0,
        grid_back: float = 2.5,
        grid_left: float = 5.0,
        grid_right: float = 5.0,
        resolution: float = 0.25,
        inflation_radius: float = 0.60,
        z_min: float = -1.2,
        z_max: float = 1.4,
        min_range: float = 0.45,
        max_range: float = 7.5,
        local_goal_dist: float = 4.5,
        lookahead_dist: float = 1.35,
        max_speed: float = 0.55,
        min_speed: float = 0.14,
        goal_tolerance: float = 1.0,
        state_timeout: float = 2.0,
        lidar_timeout: float = 2.0,
    ):
        super().__init__('local_static_planner')
        self.drone_id = int(drone_id)
        self.state_topic = state_topic
        self.lidar_topic = lidar_topic or f'/x500_{self.drone_id}/lidar/points_local'
        self.cmd_topic = cmd_topic or f'/xtdrone2/x500_{self.drone_id}/cmd_vel_ned'
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.target_z = float(target_z)
        self.duration = float(duration)
        self.rate_hz = float(rate_hz)
        self.grid_forward = float(grid_forward)
        self.grid_back = float(grid_back)
        self.grid_left = float(grid_left)
        self.grid_right = float(grid_right)
        self.resolution = float(resolution)
        self.inflation_radius = float(inflation_radius)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.local_goal_dist = float(local_goal_dist)
        self.lookahead_dist = float(lookahead_dist)
        self.max_speed = float(max_speed)
        self.min_speed = float(min_speed)
        self.goal_tolerance = float(goal_tolerance)
        self.state_timeout = float(state_timeout)
        self.lidar_timeout = float(lidar_timeout)

        self._state: Optional[Dict[str, float]] = None
        self._cloud: Optional[PointCloud2] = None
        self._cloud_stamp = 0.0
        self._start_time = time.time()
        self._last_log_time = 0.0
        self._last_path: List[BodyPoint] = []
        self._last_cmd_body: BodyPoint = (0.0, 0.0)
        self._last_progress_dist: Optional[float] = None
        self._last_progress_time = time.time()
        self._recovery_until = 0.0
        self._recovery_side = 1.0
        self._recovery_reason = ''

        self.create_subscription(String, self.state_topic, self._state_cb, 10)
        self.create_subscription(PointCloud2, self.lidar_topic, self._lidar_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.timer = self.create_timer(max(0.02, 1.0 / max(self.rate_hz, 1.0)), self._run)

        self.get_logger().info(
            'local_static_planner started: '
            f'x500_{self.drone_id}, lidar={self.lidar_topic}, cmd={self.cmd_topic}, '
            f'target=({self.target_x:.2f},{self.target_y:.2f},{self.target_z:.2f}), '
            f'grid=[-{self.grid_back:.1f},+{self.grid_forward:.1f}]x[-{self.grid_right:.1f},+{self.grid_left:.1f}], '
            f'res={self.resolution:.2f}, inflation={self.inflation_radius:.2f}, max_speed={self.max_speed:.2f}'
        )

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------

    def _state_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            for raw in data.get('states', []):
                if int(raw.get('id', 0)) != self.drone_id:
                    continue
                self._state = {
                    'x': float(raw.get('world_x', raw.get('x', 0.0))),
                    'y': float(raw.get('world_y', raw.get('y', 0.0))),
                    'z': float(raw.get('z', 0.0)),
                    'vx': float(raw.get('vx', 0.0)),
                    'vy': float(raw.get('vy', 0.0)),
                    'vz': float(raw.get('vz', 0.0)),
                    'heading': float(raw.get('heading', 0.0)),
                    'stamp': float(raw.get('stamp', time.time())),
                }
                return
        except Exception as exc:
            self.get_logger().warn(f'local_static_planner state parse failed: {exc}')

    def _lidar_cb(self, msg: PointCloud2):
        self._cloud = msg
        self._cloud_stamp = time.time()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _limit(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

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

    def _grid_shape(self) -> Tuple[int, int]:
        nx = int(round((self.grid_forward + self.grid_back) / self.resolution)) + 1
        ny = int(round((self.grid_left + self.grid_right) / self.resolution)) + 1
        return nx, ny

    def _body_to_grid(self, x: float, y: float) -> Optional[GridIndex]:
        if x < -self.grid_back or x > self.grid_forward:
            return None
        if y < -self.grid_right or y > self.grid_left:
            return None
        ix = int(round((x + self.grid_back) / self.resolution))
        iy = int(round((y + self.grid_right) / self.resolution))
        nx, ny = self._grid_shape()
        if 0 <= ix < nx and 0 <= iy < ny:
            return ix, iy
        return None

    def _grid_to_body(self, idx: GridIndex) -> BodyPoint:
        ix, iy = idx
        return ix * self.resolution - self.grid_back, iy * self.resolution - self.grid_right

    # ------------------------------------------------------------------
    # Local map
    # ------------------------------------------------------------------

    def _extract_points(self) -> List[BodyPoint]:
        cloud = self._cloud
        if cloud is None:
            return []
        points: List[BodyPoint] = []
        for idx, p in enumerate(point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True)):
            if idx >= 9000:
                break
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])
            horizontal = math.hypot(x, y)
            if horizontal < self.min_range or horizontal > self.max_range:
                continue
            if z < self.z_min or z > self.z_max:
                continue
            if x < -self.grid_back or x > self.grid_forward:
                continue
            if y < -self.grid_right or y > self.grid_left:
                continue
            points.append((x, y))
        return points

    def _build_grid(self, points: Sequence[BodyPoint]) -> Tuple[List[List[bool]], int]:
        nx, ny = self._grid_shape()
        occupied = [[False for _ in range(ny)] for _ in range(nx)]
        raw_count = 0
        inflate_cells = int(math.ceil(self.inflation_radius / self.resolution))
        for x, y in points:
            cell = self._body_to_grid(x, y)
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
        # Keep a small bubble around the vehicle free; close returns and self/noise
        # otherwise make A* unable to leave the start cell.
        start = self._body_to_grid(0.0, 0.0)
        clear_cells = max(1, int(round(0.55 / self.resolution)))
        if start is not None:
            sx, sy = start
            for dx in range(-clear_cells, clear_cells + 1):
                for dy in range(-clear_cells, clear_cells + 1):
                    ix = sx + dx
                    iy = sy + dy
                    if 0 <= ix < nx and 0 <= iy < ny:
                        occupied[ix][iy] = False
        return occupied, raw_count

    def _nearest_free(self, occupied: List[List[bool]], preferred: GridIndex, max_r: int = 12) -> Optional[GridIndex]:
        nx, ny = self._grid_shape()
        px, py = preferred
        if 0 <= px < nx and 0 <= py < ny and not occupied[px][py]:
            return preferred
        best = None
        best_score = 1e9
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    ix, iy = px + dx, py + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not occupied[ix][iy]:
                        score = dx * dx + dy * dy
                        if score < best_score:
                            best, best_score = (ix, iy), score
            for dy in range(-r + 1, r):
                for dx in (-r, r):
                    ix, iy = px + dx, py + dy
                    if 0 <= ix < nx and 0 <= iy < ny and not occupied[ix][iy]:
                        score = dx * dx + dy * dy
                        if score < best_score:
                            best, best_score = (ix, iy), score
            if best is not None:
                return best
        return None

    # ------------------------------------------------------------------
    # Candidate goals and A*
    # ------------------------------------------------------------------

    def _candidate_goals(self, occupied: List[List[bool]], target_body: BodyPoint) -> List[Tuple[GridIndex, str]]:
        tx, ty = target_body
        target_angle = math.atan2(ty, tx)
        candidates: List[Tuple[GridIndex, str]] = []
        seen = set()

        # A complete local planner must not depend on a single local goal.  Try
        # target direction first, then increasingly lateral bypass directions.
        angle_offsets_deg = [0, 18, -18, 35, -35, 55, -55, 75, -75, 95, -95, 120, -120]
        radii = [self.local_goal_dist, 3.6, 2.8, 2.1]
        for deg in angle_offsets_deg:
            for radius in radii:
                ang = target_angle + math.radians(deg)
                x = self._limit(math.cos(ang) * radius, -self.grid_back + 0.4, self.grid_forward - 0.4)
                y = self._limit(math.sin(ang) * radius, -self.grid_right + 0.4, self.grid_left - 0.4)
                cell = self._body_to_grid(x, y)
                if cell is None:
                    continue
                free = self._nearest_free(occupied, cell, max_r=8)
                if free is None or free in seen:
                    continue
                bx, by = self._grid_to_body(free)
                # Avoid choosing goals too close to the vehicle unless we are recovering.
                if math.hypot(bx, by) < 1.0:
                    continue
                seen.add(free)
                candidates.append((free, f'{deg:+d}deg/{radius:.1f}m'))
        return candidates

    def _astar(self, occupied: List[List[bool]], start: GridIndex, goal: GridIndex) -> Optional[List[GridIndex]]:
        nx, ny = self._grid_shape()
        neighbors = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0)),
        ]

        def heuristic(a: GridIndex, b: GridIndex) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        frontier: List[Tuple[float, GridIndex]] = []
        heapq.heappush(frontier, (0.0, start))
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
                if occupied[nx_i][ny_i]:
                    continue
                new_cost = cost_so_far[current] + step_cost
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current
        return None

    def _line_free(self, occupied: List[List[bool]], a: GridIndex, b: GridIndex) -> bool:
        ax, ay = a
        bx, by = b
        steps = max(abs(bx - ax), abs(by - ay), 1)
        for i in range(steps + 1):
            t = i / float(steps)
            ix = int(round(ax + (bx - ax) * t))
            iy = int(round(ay + (by - ay) * t))
            if occupied[ix][iy]:
                return False
        return True

    def _shortcut(self, occupied: List[List[bool]], path: List[GridIndex]) -> List[GridIndex]:
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_free(occupied, path[i], path[j]):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _path_length(self, path: Sequence[GridIndex]) -> float:
        if len(path) <= 1:
            return 0.0
        length = 0.0
        last = path[0]
        for cur in path[1:]:
            length += math.hypot(cur[0] - last[0], cur[1] - last[1]) * self.resolution
            last = cur
        return length

    def _clearance_to_points(self, point: BodyPoint, points: Sequence[BodyPoint]) -> float:
        if not points:
            return 999.0
        px, py = point
        return min(math.hypot(px - x, py - y) for x, y in points)

    def _score_path(
        self,
        path: List[GridIndex],
        label: str,
        target_body: BodyPoint,
        points: Sequence[BodyPoint],
    ) -> Dict[str, object]:
        body_path = [self._grid_to_body(cell) for cell in path]
        end_x, end_y = body_path[-1]
        target_angle = math.atan2(target_body[1], target_body[0])
        end_angle = math.atan2(end_y, end_x)
        progress = end_x * math.cos(target_angle) + end_y * math.sin(target_angle)
        angle_cost = abs(self._angle_diff(end_angle, target_angle))
        length = self._path_length(path)
        min_clearance = min((self._clearance_to_points(p, points) for p in body_path[1:]), default=999.0)
        clearance_cost = 0.0 if min_clearance >= 1.4 else (1.4 - min_clearance) * 3.0
        score = length + 2.0 * angle_cost - 0.65 * progress + clearance_cost
        return {
            'score': score,
            'path': path,
            'body_path': body_path,
            'label': label,
            'progress': progress,
            'angle_cost': angle_cost,
            'length': length,
            'min_path_clearance': min_clearance,
        }

    def _select_path(
        self,
        occupied: List[List[bool]],
        start: GridIndex,
        target_body: BodyPoint,
        points: Sequence[BodyPoint],
    ) -> Optional[Dict[str, object]]:
        candidates = self._candidate_goals(occupied, target_body)
        scored: List[Dict[str, object]] = []
        for goal, label in candidates:
            path = self._astar(occupied, start, goal)
            if path is None:
                continue
            smooth = self._shortcut(occupied, path)
            if len(smooth) < 2:
                continue
            scored.append(self._score_path(smooth, label, target_body, points))
        if not scored:
            return None
        scored.sort(key=lambda item: float(item['score']))
        return scored[0]

    # ------------------------------------------------------------------
    # Tracking, near-field safety, recovery
    # ------------------------------------------------------------------

    def _lookahead(self, body_path: List[BodyPoint]) -> BodyPoint:
        if not body_path:
            return 0.0, 0.0
        last = body_path[0]
        acc = 0.0
        for point in body_path[1:]:
            seg = math.hypot(point[0] - last[0], point[1] - last[1])
            if acc + seg >= self.lookahead_dist:
                r = (self.lookahead_dist - acc) / max(seg, 1e-6)
                return last[0] + r * (point[0] - last[0]), last[1] + r * (point[1] - last[1])
            acc += seg
            last = point
        return body_path[-1]

    def _front_contact(self, points: Sequence[BodyPoint]) -> bool:
        return any(0.05 < x < 0.90 and abs(y) < 0.65 for x, y in points)

    def _sector_clearance(self, points: Sequence[BodyPoint], y_min: float, y_max: float, x_min: float = -0.3, x_max: float = 2.8) -> float:
        values = [math.hypot(x, y) for x, y in points if x_min <= x <= x_max and y_min <= y <= y_max]
        return min(values) if values else 999.0

    def _pick_escape_side(self, points: Sequence[BodyPoint], target_body: BodyPoint) -> Tuple[float, float, float]:
        left_clear = self._sector_clearance(points, 0.25, self.grid_left)
        right_clear = self._sector_clearance(points, -self.grid_right, -0.25)
        target_side = 1.0 if target_body[1] >= 0.0 else -1.0
        if abs(left_clear - right_clear) < 0.25:
            side = target_side
        else:
            side = 1.0 if left_clear > right_clear else -1.0
        return side, left_clear, right_clear

    def _escape_velocity_body(self, points: Sequence[BodyPoint], target_body: BodyPoint, aggressive: bool = False) -> Tuple[float, float, str, float, float]:
        side, left_clear, right_clear = self._pick_escape_side(points, target_body)
        back_clear = self._sector_clearance(points, -0.9, 0.9, x_min=-self.grid_back, x_max=-0.2)
        lateral = (0.40 if aggressive else 0.32) * side
        back = -0.16 if back_clear > 0.55 else 0.0
        if max(left_clear, right_clear) < 0.75:
            lateral = 0.20 * side
            back = -0.22 if back_clear > 0.45 else 0.0
        return back, lateral, 'recovery_escape_left' if side > 0 else 'recovery_escape_right', left_clear, right_clear

    def _smooth_cmd_body(self, bx: float, by: float) -> BodyPoint:
        old_x, old_y = self._last_cmd_body
        alpha = 0.55
        sx = alpha * bx + (1.0 - alpha) * old_x
        sy = alpha * by + (1.0 - alpha) * old_y
        mag = math.hypot(sx, sy)
        if mag > self.max_speed:
            sx *= self.max_speed / mag
            sy *= self.max_speed / mag
        self._last_cmd_body = (sx, sy)
        return sx, sy

    def _publish_cmd(self, vx: float, vy: float, vz: float = 0.0):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._last_cmd_body = (0.0, 0.0)
        self._publish_cmd(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def _run(self):
        now = time.time()
        if now - self._start_time > self.duration:
            self._stop()
            if now - self._last_log_time > 2.0:
                self._last_log_time = now
                self.get_logger().info('local_static_planner done: duration reached, publishing stop')
            return

        state = self._state
        if state is None or now - float(state.get('stamp', 0.0)) > self.state_timeout:
            self._stop()
            return
        if self._cloud is None or now - self._cloud_stamp > self.lidar_timeout:
            self._stop()
            if now - self._last_log_time > 1.5:
                self._last_log_time = now
                self.get_logger().warn('local_static_planner waiting for fresh LiDAR; stop')
            return

        dx = self.target_x - float(state['x'])
        dy = self.target_y - float(state['y'])
        dz = self.target_z - float(state['z'])
        target_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if target_dist < self.goal_tolerance:
            self._stop()
            if now - self._last_log_time > 1.5:
                self._last_log_time = now
                self.get_logger().info(f'local_static_planner reached goal: target_dist={target_dist:.2f}')
            return

        heading = float(state.get('heading', 0.0))
        target_body = self._world_to_body(dx, dy, heading)
        points = self._extract_points()
        occupied, raw_count = self._build_grid(points)
        nearest = min((math.hypot(x, y) for x, y in points), default=999.0)
        left_clear = self._sector_clearance(points, 0.25, self.grid_left)
        right_clear = self._sector_clearance(points, -self.grid_right, -0.25)

        # Progress monitor. If the target distance does not improve, enter a short
        # recovery mode rather than repeatedly choosing the same blocked local path.
        if self._last_progress_dist is None or target_dist < self._last_progress_dist - 0.20:
            self._last_progress_dist = target_dist
            self._last_progress_time = now
        stalled_sec = now - self._last_progress_time
        if stalled_sec > 3.0 and now > self._recovery_until:
            self._recovery_side, _, _ = self._pick_escape_side(points, target_body)
            self._recovery_until = now + 2.2
            self._recovery_reason = f'stalled_{stalled_sec:.1f}s'

        start = self._body_to_grid(0.0, 0.0)
        selected: Optional[Dict[str, object]] = None
        if start is not None:
            selected = self._select_path(occupied, start, target_body, points)

        mode = 'astar_candidates'
        bx = 0.0
        by = 0.0
        path_len = 0
        smooth_len = 0
        selected_label = 'none'
        min_path_clearance = 999.0
        look_x = 0.0
        look_y = 0.0

        if now < self._recovery_until:
            # Hold the chosen recovery side for a short window to actually get out
            # of the local minimum; otherwise per-frame replanning can oscillate.
            back, lateral, mode, left_clear, right_clear = self._escape_velocity_body(points, target_body, aggressive=True)
            bx, by = back, lateral if self._recovery_side > 0 else -abs(lateral)
        elif selected is not None:
            body_path = list(selected['body_path'])
            path_len = len(selected['path'])
            smooth_len = len(body_path)
            selected_label = str(selected['label'])
            min_path_clearance = float(selected['min_path_clearance'])
            self._last_path = body_path
            look_x, look_y = self._lookahead(body_path)
            look_norm = max(math.hypot(look_x, look_y), 1e-6)
            speed = self.max_speed
            if nearest < 1.2:
                speed = max(self.min_speed, min(self.max_speed, 0.24))
            bx = speed * look_x / look_norm
            by = speed * look_y / look_norm

            # Near-field safety should not dead-stop unless there is no escape. If
            # a command still pushes into a close frontal obstacle, switch to a
            # lateral/back escape instead of zeroing the velocity.
            if self._front_contact(points) and bx > 0.03:
                bx, by, mode, left_clear, right_clear = self._escape_velocity_body(points, target_body)
        else:
            # No A* path to any candidate. Use a bounded recovery motion; this is
            # safer than sitting in front of the obstacle forever, but still avoids
            # climbing over vertical obstacles.
            bx, by, mode, left_clear, right_clear = self._escape_velocity_body(points, target_body, aggressive=True)
            self._recovery_until = max(self._recovery_until, now + 1.5)
            self._recovery_reason = 'no_candidate_path'

        bx, by = self._smooth_cmd_body(bx, by)
        vx, vy = self._body_to_world(bx, by, heading)
        vz = self._limit(0.35 * dz, -0.08, 0.08)
        if abs(dz) < 0.25:
            vz = 0.0
        self._publish_cmd(vx, vy, vz)

        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            self.get_logger().info(
                'local_static_planner: '
                f'mode={mode}, target_dist={target_dist:.2f}, nearest={nearest:.2f}, '
                f'left_clear={left_clear:.2f}, right_clear={right_clear:.2f}, stalled={stalled_sec:.1f}s, '
                f'recovery_left={max(0.0, self._recovery_until - now):.1f}s, recovery_reason={self._recovery_reason or "none"}, '
                f'points={raw_count}, path={path_len}, smooth={smooth_len}, label={selected_label}, '
                f'min_path_clearance={min_path_clearance:.2f}, lookahead=({look_x:.2f},{look_y:.2f}), '
                f'cmd_body=({bx:.2f},{by:.2f}), cmd_ned=({vx:.2f},{vy:.2f},{vz:.2f})'
            )


def main():
    parser = argparse.ArgumentParser(description='Single-UAV local static A* planner')
    parser.add_argument('--drone-id', type=int, default=1)
    parser.add_argument('--state-topic', type=str, default='/xtdrone2/swarm/state_exchange')
    parser.add_argument('--lidar-topic', type=str, default='')
    parser.add_argument('--cmd-topic', type=str, default='')
    parser.add_argument('--target-x', type=float, default=19.0)
    parser.add_argument('--target-y', type=float, default=30.0)
    parser.add_argument('--target-z', type=float, default=-3.0)
    parser.add_argument('--duration', type=float, default=180.0)
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--resolution', type=float, default=0.25)
    parser.add_argument('--inflation-radius', type=float, default=0.60)
    parser.add_argument('--max-speed', type=float, default=0.55)
    parser.add_argument('--local-goal-dist', type=float, default=4.5)
    parser.add_argument('--lookahead-dist', type=float, default=1.35)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = LocalStaticPlanner(
        drone_id=args.drone_id,
        state_topic=args.state_topic,
        lidar_topic=args.lidar_topic or None,
        cmd_topic=args.cmd_topic or None,
        target_x=args.target_x,
        target_y=args.target_y,
        target_z=args.target_z,
        duration=args.duration,
        rate_hz=args.rate,
        resolution=args.resolution,
        inflation_radius=args.inflation_radius,
        max_speed=args.max_speed,
        local_goal_dist=args.local_goal_dist,
        lookahead_dist=args.lookahead_dist,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
