#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-UAV static local planner for forest3 LiDAR validation.

This is intentionally a small, standalone planner node, not another patch inside
multi_waypoint2.py.  It uses the current local LiDAR point cloud to build a
2.5D body-frame occupancy grid, runs a short-horizon A* search, tracks a
lookahead point on the local path, and publishes /cmd_vel_ned.

Scope of this first version:
  - one UAV, static obstacles only;
  - local LiDAR point cloud in the UAV body/local frame;
  - NED velocity command output for xtdrone2/PX4 offboard;
  - no dynamic-obstacle prediction and no multi-UAV coordination yet.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
        duration: float = 90.0,
        rate_hz: float = 10.0,
        grid_forward: float = 6.0,
        grid_back: float = 2.0,
        grid_left: float = 4.0,
        grid_right: float = 4.0,
        resolution: float = 0.25,
        inflation_radius: float = 0.70,
        z_min: float = -1.2,
        z_max: float = 1.4,
        min_range: float = 0.45,
        max_range: float = 6.5,
        local_goal_dist: float = 4.2,
        lookahead_dist: float = 1.25,
        max_speed: float = 0.30,
        min_speed: float = 0.12,
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
    def _norm2(x: float, y: float) -> float:
        return math.sqrt(x * x + y * y)

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
    # Local map and A*
    # ------------------------------------------------------------------

    def _extract_points(self) -> List[BodyPoint]:
        cloud = self._cloud
        if cloud is None:
            return []
        points: List[BodyPoint] = []
        for idx, p in enumerate(point_cloud2.read_points(cloud, field_names=('x', 'y', 'z'), skip_nans=True)):
            if idx >= 6000:
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
        # Keep a small bubble around the vehicle free; otherwise close returns
        # and self/noise can make A* unable to leave the start cell.
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

    def _nearest_free(self, occupied: List[List[bool]], preferred: GridIndex, max_r: int = 18) -> Optional[GridIndex]:
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

    def _choose_local_goal(self, occupied: List[List[bool]], target_body: BodyPoint) -> Optional[GridIndex]:
        tx, ty = target_body
        dist = max(math.hypot(tx, ty), 1e-6)
        gx = self._limit(tx / dist * self.local_goal_dist, 1.2, self.grid_forward - 0.4)
        gy = self._limit(ty / dist * self.local_goal_dist, -self.grid_right + 0.4, self.grid_left - 0.4)
        preferred = self._body_to_grid(gx, gy)
        if preferred is None:
            return None
        goal = self._nearest_free(occupied, preferred)
        if goal is not None:
            return goal

        # Fallback: scan angular sectors and select the free cell closest to the
        # final target direction. This is only used when the preferred local goal
        # is completely blocked.
        target_angle = math.atan2(ty, tx)
        best = None
        best_score = 1e9
        for deg in range(-85, 86, 10):
            ang = target_angle + math.radians(deg)
            for radius in (3.8, 3.2, 2.6, 2.0, 1.5):
                x = self._limit(math.cos(ang) * radius, -self.grid_back + 0.2, self.grid_forward - 0.4)
                y = self._limit(math.sin(ang) * radius, -self.grid_right + 0.4, self.grid_left - 0.4)
                cell = self._body_to_grid(x, y)
                if cell is None:
                    continue
                free = self._nearest_free(occupied, cell, max_r=4)
                if free is None:
                    continue
                bx, by = self._grid_to_body(free)
                progress = bx * math.cos(target_angle) + by * math.sin(target_angle)
                angle_cost = abs(math.atan2(math.sin(math.atan2(by, bx) - target_angle), math.cos(math.atan2(by, bx) - target_angle)))
                score = 2.0 * angle_cost - 0.25 * progress
                if score < best_score:
                    best = free
                    best_score = score
        return best

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
                nx_i, ny_i = cx + dx, cy + dy
                nxt = (nx_i, ny_i)
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
        for x, y in points:
            if 0.05 < x < 0.80 and abs(y) < 0.55:
                return True
        return False

    # ------------------------------------------------------------------
    # Main planning loop
    # ------------------------------------------------------------------

    def _publish_cmd(self, vx: float, vy: float, vz: float = 0.0):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        self.cmd_pub.publish(msg)

    def _stop(self):
        self._publish_cmd(0.0, 0.0, 0.0)

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
        start = self._body_to_grid(0.0, 0.0)
        if start is None:
            self._stop()
            return
        goal = self._choose_local_goal(occupied, target_body)
        if goal is None:
            self._stop()
            if now - self._last_log_time > 1.0:
                self._last_log_time = now
                self.get_logger().warn('local_static_planner no free local goal; stop')
            return

        path = self._astar(occupied, start, goal)
        mode = 'astar'
        if path is None:
            self._stop()
            if now - self._last_log_time > 1.0:
                self._last_log_time = now
                self.get_logger().warn('local_static_planner astar failed; stop')
            return

        smooth = self._shortcut(occupied, path)
        body_path = [self._grid_to_body(cell) for cell in smooth]
        self._last_path = body_path
        look_x, look_y = self._lookahead(body_path)
        look_norm = max(math.hypot(look_x, look_y), 1e-6)
        nearest = min((math.hypot(x, y) for x, y in points), default=999.0)

        # A conservative hard contact guard. It only stops if the planned command
        # would still push into a very close frontal obstacle.
        speed = self.max_speed
        if nearest < 1.2:
            speed = max(self.min_speed, min(self.max_speed, 0.18))
        bx = speed * look_x / look_norm
        by = speed * look_y / look_norm
        if self._front_contact(points) and bx > 0.05:
            bx, by = 0.0, 0.0
            mode = 'front_contact_stop'

        vx, vy = self._body_to_world(bx, by, heading)
        vz = self._limit(0.35 * dz, -0.08, 0.08)
        # Keep this first static version near constant altitude.
        if abs(dz) < 0.25:
            vz = 0.0
        self._publish_cmd(vx, vy, vz)

        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            goal_body = self._grid_to_body(goal)
            self.get_logger().info(
                'local_static_planner: '
                f'mode={mode}, target_dist={target_dist:.2f}, nearest={nearest:.2f}, '
                f'points={raw_count}, path={len(path)}, smooth={len(smooth)}, '
                f'local_goal=({goal_body[0]:.2f},{goal_body[1]:.2f}), '
                f'lookahead=({look_x:.2f},{look_y:.2f}), '
                f'cmd_ned=({vx:.2f},{vy:.2f},{vz:.2f})'
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
    parser.add_argument('--duration', type=float, default=90.0)
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--resolution', type=float, default=0.25)
    parser.add_argument('--inflation-radius', type=float, default=0.70)
    parser.add_argument('--max-speed', type=float, default=0.30)
    parser.add_argument('--local-goal-dist', type=float, default=4.2)
    parser.add_argument('--lookahead-dist', type=float, default=1.25)
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
