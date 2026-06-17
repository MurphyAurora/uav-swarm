#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Near-field LiDAR TTC safety filter for UAV primitive velocity commands."""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String


def finite(value, default=0.0):
    try:
        v = float(value)
    except Exception:
        return default
    if not math.isfinite(v):
        return default
    return v


class LidarTtcSafetyFilter(Node):
    def __init__(
        self,
        num_drones=5,
        input_topic_template='/xtdrone2/x500_{id}/primitive_cmd_vel_ned',
        output_topic_template='/xtdrone2/x500_{id}/safe_cmd_vel_ned',
        lidar_topic_template='/x500_{id}/lidar/points_local',
        state_topic='/xtdrone2/swarm/state_exchange',
        warning_ttc=2.0,
        emergency_ttc=0.8,
        min_speed=0.08,
        max_points=2500,
        max_range=6.0,
        min_range=0.18,
        vertical_limit=1.8,
        cone_angle_deg=42.0,
        safety_radius=0.55,
        self_filter_radius=0.65,
        projection_gain=1.0,
        brake_gain=0.15,
        back_speed=0.20,
        climb_speed=0.18,
        cmd_timeout=0.6,
        point_timeout=0.8,
    ):
        super().__init__('lidar_ttc_safety_filter')
        self.num_drones = int(num_drones)
        self.input_topic_template = str(input_topic_template)
        self.output_topic_template = str(output_topic_template)
        self.lidar_topic_template = str(lidar_topic_template)
        self.state_topic = str(state_topic)
        self.warning_ttc = float(warning_ttc)
        self.emergency_ttc = float(emergency_ttc)
        self.min_speed = float(min_speed)
        self.max_points = int(max_points)
        self.max_range = float(max_range)
        self.min_range = float(min_range)
        self.vertical_limit = float(vertical_limit)
        self.cone_cos = math.cos(math.radians(float(cone_angle_deg)))
        self.safety_radius = float(safety_radius)
        self.self_filter_radius = float(self_filter_radius)
        self.projection_gain = float(projection_gain)
        self.brake_gain = float(brake_gain)
        self.back_speed = float(back_speed)
        self.climb_speed = float(climb_speed)
        self.cmd_timeout = float(cmd_timeout)
        self.point_timeout = float(point_timeout)

        self.cmds = {}
        self.clouds = {}
        self.headings = {}
        self.last_log_time = 0.0

        self.create_subscription(String, self.state_topic, self._state_cb, 10)
        self.pubs = {}
        for drone_id in range(1, self.num_drones + 1):
            self.create_subscription(
                Twist,
                self.input_topic_template.format(id=drone_id),
                lambda msg, i=drone_id: self._cmd_cb(i, msg),
                10,
            )
            self.create_subscription(
                PointCloud2,
                self.lidar_topic_template.format(id=drone_id),
                lambda msg, i=drone_id: self._cloud_cb(i, msg),
                5,
            )
            self.pubs[drone_id] = self.create_publisher(
                Twist,
                self.output_topic_template.format(id=drone_id),
                10,
            )

        self.timer = self.create_timer(0.05, self._run)
        self.get_logger().info(
            'lidar_ttc_safety_filter started: '
            f'num={self.num_drones}, warning_ttc={self.warning_ttc:.2f}, '
            f'emergency_ttc={self.emergency_ttc:.2f}, max_range={self.max_range:.2f}, '
            f'cone_cos={self.cone_cos:.2f}, safety_radius={self.safety_radius:.2f}, '
            f'self_filter_radius={self.self_filter_radius:.2f}, '
            f'input={self.input_topic_template}, output={self.output_topic_template}, '
            f'lidar={self.lidar_topic_template}'
        )

    def _state_cb(self, msg):
        try:
            data = json.loads(msg.data)
            for item in data.get('states', []):
                drone_id = int(item.get('id', 0))
                if 1 <= drone_id <= self.num_drones:
                    self.headings[drone_id] = finite(item.get('heading', 0.0))
        except Exception:
            return

    def _cmd_cb(self, drone_id, msg):
        self.cmds[int(drone_id)] = {
            'vx': finite(msg.linear.x),
            'vy': finite(msg.linear.y),
            'vz': finite(msg.linear.z),
            'stamp': time.time(),
        }

    def _cloud_cb(self, drone_id, msg):
        self.clouds[int(drone_id)] = {
            'msg': msg,
            'stamp': time.time(),
        }

    def _run(self):
        now = time.time()
        reports = []
        for drone_id in range(1, self.num_drones + 1):
            cmd = self.cmds.get(drone_id)
            if cmd is None or now - float(cmd['stamp']) > self.cmd_timeout:
                continue
            cloud = self.clouds.get(drone_id)
            if cloud is None or now - float(cloud['stamp']) > self.point_timeout:
                self._publish(drone_id, 0.0, 0.0, 0.0)
                reports.append(self._result(drone_id, 'no_cloud_hold', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0))
                continue

            result = self._filter_command(drone_id, cmd, cloud['msg'])
            self._publish(drone_id, result['vx'], result['vy'], result['vz'])
            reports.append(result)

        if now - self.last_log_time >= 1.5:
            self.last_log_time = now
            active = [r for r in reports if r['mode'] != 'pass']
            if active:
                worst = min(active, key=lambda item: item['ttc'])
                self.get_logger().info(
                    'lidar ttc safety: '
                    f'active={len(active)}/{len(reports)}, '
                    f'worst=x500_{worst["drone_id"]} mode={worst["mode"]} '
                    f'ttc={worst["ttc"]:.2f}, dist={worst["distance"]:.2f}, '
                    f'closing={worst["closing_speed"]:.2f}, '
                    f'point=({worst["px"]:.2f},{worst["py"]:.2f},{worst["pz"]:.2f}), '
                    f'points={worst["points_used"]}'
                )
            else:
                self.get_logger().info(
                    f'lidar ttc safety: pass, filtered_drones={len(reports)}'
                )

    def _filter_command(self, drone_id, cmd, cloud_msg):
        vx = float(cmd['vx'])
        vy = float(cmd['vy'])
        vz = float(cmd['vz'])
        heading = self.headings.get(int(drone_id), 0.0)
        bvx, bvy = self._world_to_body_xy(vx, vy, heading)
        bvz = vz
        speed_xy = math.hypot(bvx, bvy)
        if speed_xy < self.min_speed:
            return self._result(drone_id, 'pass', vx, vy, vz, 999.0, 999.0, 0.0, 0)

        dir_x = bvx / speed_xy
        dir_y = bvy / speed_xy
        worst = None
        used = 0
        for idx, point in enumerate(point_cloud2.read_points(cloud_msg, field_names=('x', 'y', 'z'), skip_nans=True)):
            if idx >= self.max_points:
                break
            px = finite(point[0])
            py = finite(point[1])
            pz = finite(point[2])
            horizontal = math.hypot(px, py)
            if horizontal < self.min_range or horizontal < self.self_filter_radius or horizontal > self.max_range:
                continue
            if abs(pz) > self.vertical_limit:
                continue
            ux = px / max(horizontal, 1e-6)
            uy = py / max(horizontal, 1e-6)
            alignment = ux * dir_x + uy * dir_y
            if alignment < self.cone_cos:
                continue
            closing = bvx * ux + bvy * uy
            if closing <= self.min_speed:
                continue
            effective_dist = max(0.0, horizontal - self.safety_radius)
            ttc = effective_dist / max(closing, 1e-6)
            if ttc > self.warning_ttc:
                continue
            used += 1
            if worst is None or ttc < worst['ttc']:
                worst = {
                    'ttc': ttc,
                    'distance': horizontal,
                    'closing': closing,
                    'ux': ux,
                    'uy': uy,
                    'px': px,
                    'py': py,
                    'pz': pz,
                }

        if worst is None:
            return self._result(drone_id, 'pass', vx, vy, vz, 999.0, 999.0, 0.0, used)

        ux = worst['ux']
        uy = worst['uy']
        toward = bvx * ux + bvy * uy
        if worst['ttc'] <= self.emergency_ttc:
            side_x = bvx - toward * ux
            side_y = bvy - toward * uy
            nbvx = self.brake_gain * side_x - self.back_speed * ux
            nbvy = self.brake_gain * side_y - self.back_speed * uy
            if abs(self.climb_speed) > 1e-6:
                nbvz = min(bvz, -abs(self.climb_speed))
            else:
                nbvz = 0.0
            mode = 'emergency'
        else:
            scale = (self.warning_ttc - worst['ttc']) / max(self.warning_ttc - self.emergency_ttc, 1e-6)
            remove = max(0.0, toward) * min(1.0, max(0.0, scale * self.projection_gain))
            nbvx = bvx - remove * ux
            nbvy = bvy - remove * uy
            nbvz = bvz
            mode = 'project'

        nvx, nvy = self._body_to_world_xy(nbvx, nbvy, heading)
        return self._result(
            drone_id,
            mode,
            nvx,
            nvy,
            nbvz,
            worst['ttc'],
            worst['distance'],
            worst['closing'],
            used,
            worst['px'],
            worst['py'],
            worst['pz'],
        )

    def _publish(self, drone_id, vx, vy, vz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        self.pubs[int(drone_id)].publish(msg)

    @staticmethod
    def _world_to_body_xy(vx, vy, heading):
        c = math.cos(float(heading))
        s = math.sin(float(heading))
        return c * float(vx) + s * float(vy), -s * float(vx) + c * float(vy)

    @staticmethod
    def _body_to_world_xy(vx, vy, heading):
        c = math.cos(float(heading))
        s = math.sin(float(heading))
        return c * float(vx) - s * float(vy), s * float(vx) + c * float(vy)

    @staticmethod
    def _result(drone_id, mode, vx, vy, vz, ttc, distance, closing_speed, points_used, px=999.0, py=999.0, pz=999.0):
        return {
            'drone_id': int(drone_id),
            'mode': str(mode),
            'vx': float(vx),
            'vy': float(vy),
            'vz': float(vz),
            'ttc': float(ttc),
            'distance': float(distance),
            'closing_speed': float(closing_speed),
            'points_used': int(points_used),
            'px': float(px),
            'py': float(py),
            'pz': float(pz),
        }


def main():
    parser = argparse.ArgumentParser(description='LiDAR TTC velocity-cone safety filter')
    parser.add_argument('--num-drones', type=int, default=5)
    parser.add_argument('--input-topic-template', type=str, default='/xtdrone2/x500_{id}/primitive_cmd_vel_ned')
    parser.add_argument('--output-topic-template', type=str, default='/xtdrone2/x500_{id}/safe_cmd_vel_ned')
    parser.add_argument('--lidar-topic-template', type=str, default='/x500_{id}/lidar/points_local')
    parser.add_argument('--state-topic', type=str, default='/xtdrone2/swarm/state_exchange')
    parser.add_argument('--warning-ttc', type=float, default=2.0)
    parser.add_argument('--emergency-ttc', type=float, default=0.8)
    parser.add_argument('--min-speed', type=float, default=0.08)
    parser.add_argument('--max-points', type=int, default=2500)
    parser.add_argument('--max-range', type=float, default=6.0)
    parser.add_argument('--min-range', type=float, default=0.18)
    parser.add_argument('--vertical-limit', type=float, default=1.8)
    parser.add_argument('--cone-angle-deg', type=float, default=42.0)
    parser.add_argument('--safety-radius', type=float, default=0.55)
    parser.add_argument('--self-filter-radius', type=float, default=0.65)
    parser.add_argument('--projection-gain', type=float, default=1.0)
    parser.add_argument('--brake-gain', type=float, default=0.15)
    parser.add_argument('--back-speed', type=float, default=0.20)
    parser.add_argument('--climb-speed', type=float, default=0.18)
    parser.add_argument('--cmd-timeout', type=float, default=0.6)
    parser.add_argument('--point-timeout', type=float, default=0.8)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = LidarTtcSafetyFilter(
        num_drones=args.num_drones,
        input_topic_template=args.input_topic_template,
        output_topic_template=args.output_topic_template,
        lidar_topic_template=args.lidar_topic_template,
        state_topic=args.state_topic,
        warning_ttc=args.warning_ttc,
        emergency_ttc=args.emergency_ttc,
        min_speed=args.min_speed,
        max_points=args.max_points,
        max_range=args.max_range,
        min_range=args.min_range,
        vertical_limit=args.vertical_limit,
        cone_angle_deg=args.cone_angle_deg,
        safety_radius=args.safety_radius,
        self_filter_radius=args.self_filter_radius,
        projection_gain=args.projection_gain,
        brake_gain=args.brake_gain,
        back_speed=args.back_speed,
        climb_speed=args.climb_speed,
        cmd_timeout=args.cmd_timeout,
        point_timeout=args.point_timeout,
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
