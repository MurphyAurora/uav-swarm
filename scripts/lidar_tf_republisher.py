#!/usr/bin/env python3
"""Publish clean per-UAV LiDAR TFs and frame-id-corrected LiDAR topics for RViz."""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import MarkerArray


def yaw_to_quat(yaw):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class LidarTfRepublisher(Node):
    def __init__(self, num_drones, lidar_x, lidar_y, lidar_z):
        super().__init__('lidar_tf_republisher')
        self.num_drones = int(num_drones)
        self.lidar_offset = (float(lidar_x), float(lidar_y), float(lidar_z))
        self.tf_broadcaster = TransformBroadcaster(self)
        self.latest_states = {}

        self.create_subscription(String, '/xtdrone2/swarm/state_exchange', self._state_cb, 10)
        self.create_subscription(
            MarkerArray,
            '/xtdrone2/swarm/drone_markers_rviz',
            self._marker_cb,
            10,
        )

        self.point_pubs = {}
        self.scan_pubs = {}
        for drone_id in range(1, self.num_drones + 1):
            self.point_pubs[drone_id] = self.create_publisher(
                PointCloud2, f'/x500_{drone_id}/lidar/points_local', 10
            )
            self.scan_pubs[drone_id] = self.create_publisher(
                LaserScan, f'/x500_{drone_id}/lidar/scan_local', 10
            )
            self.create_subscription(
                PointCloud2,
                f'/x500_{drone_id}/lidar/points',
                lambda msg, did=drone_id: self._points_cb(did, msg),
                10,
            )
            self.create_subscription(
                LaserScan,
                f'/x500_{drone_id}/lidar/scan',
                lambda msg, did=drone_id: self._scan_cb(did, msg),
                10,
            )

        self.timer = self.create_timer(0.05, self._publish_dynamic_tf)
        self.get_logger().info(
            f'lidar_tf_republisher started: num_drones={self.num_drones}, '
            f'lidar_offset={self.lidar_offset}'
        )

    def _state_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        for state in payload.get('states', []):
            try:
                drone_id = int(state.get('id', -1))
            except Exception:
                continue
            if not (1 <= drone_id <= self.num_drones):
                continue

            ned_x = float(state.get('world_x', state.get('x', 0.0)))
            ned_y = float(state.get('world_y', state.get('y', 0.0)))
            ned_z = float(state.get('world_z', state.get('z', 0.0)))
            heading = float(state.get('heading', 0.0))
            self.latest_states[drone_id] = {
                'stamp': time.time(),
                # RViz/Gazebo ENU map: ENU x = NED y, ENU y = NED x, ENU z = -NED z.
                'enu_x': ned_y,
                'enu_y': ned_x,
                'enu_z': -ned_z,
                # NED heading from north clockwise -> ENU yaw from east counter-clockwise.
                'yaw': math.pi * 0.5 - heading,
            }

    def _marker_cb(self, msg):
        headings = {}
        for marker in msg.markers:
            if marker.ns != 'drone_heading' or len(marker.points) < 2:
                continue
            drone_id = int(marker.id) - 1000
            if not (1 <= drone_id <= self.num_drones):
                continue
            p0 = marker.points[0]
            p1 = marker.points[1]
            headings[drone_id] = math.atan2(p1.y - p0.y, p1.x - p0.x)

        for marker in msg.markers:
            if marker.ns != 'drone_body':
                continue
            drone_id = int(marker.id)
            if not (1 <= drone_id <= self.num_drones):
                continue
            pos = marker.pose.position
            self.latest_states[drone_id] = {
                'stamp': time.time(),
                'enu_x': float(pos.x),
                'enu_y': float(pos.y),
                'enu_z': float(pos.z),
                'yaw': headings.get(drone_id, 0.0),
            }

    def _publish_dynamic_tf(self):
        now = self.get_clock().now().to_msg()
        transforms = []
        ox, oy, oz = self.lidar_offset
        for drone_id, state in sorted(self.latest_states.items()):
            if time.time() - state.get('stamp', 0.0) > 2.0:
                continue
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = 'map'
            t.child_frame_id = f'x500_{drone_id}/base_link'
            t.transform.translation.x = state['enu_x']
            t.transform.translation.y = state['enu_y']
            t.transform.translation.z = state['enu_z']
            qx, qy, qz, qw = yaw_to_quat(state['yaw'])
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            transforms.append(t)

            lidar_t = TransformStamped()
            lidar_t.header.stamp = now
            lidar_t.header.frame_id = f'x500_{drone_id}/base_link'
            lidar_t.child_frame_id = f'x500_{drone_id}/lidar_3d_link'
            lidar_t.transform.translation.x = ox
            lidar_t.transform.translation.y = oy
            lidar_t.transform.translation.z = oz
            lidar_t.transform.rotation.w = 1.0
            transforms.append(lidar_t)
        if transforms:
            self.tf_broadcaster.sendTransform(transforms)

    def _points_cb(self, drone_id, msg):
        # RViz treats stamp=0 as "use the latest available TF", which avoids
        # display flicker when Gazebo sensor time and ROS wall time drift.
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = f'x500_{drone_id}/lidar_3d_link'
        self.point_pubs[drone_id].publish(msg)

    def _scan_cb(self, drone_id, msg):
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0
        msg.header.frame_id = f'x500_{drone_id}/lidar_3d_link'
        self.scan_pubs[drone_id].publish(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-drones', type=int, default=5)
    parser.add_argument('--lidar-x', type=float, default=0.12)
    parser.add_argument('--lidar-y', type=float, default=0.0)
    parser.add_argument('--lidar-z', type=float, default=0.18)
    args = parser.parse_args()

    rclpy.init()
    node = LidarTfRepublisher(args.num_drones, args.lidar_x, args.lidar_y, args.lidar_z)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
