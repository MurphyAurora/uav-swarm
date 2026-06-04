#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mission Controller Node

Replaces start1.sh lines 394-504 (warmup, ARM, OFFBOARD, mission).
State machine: WARMUP -> ARMING -> OFFBOARD_HOLD -> MISSION -> DONE
"""

import os
import sys
import time
import argparse

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Pose
from xtd2_msgs.srv import XTD2Cmd


class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller')

        # --- ROS 2 parameters ---
        self.declare_parameter('num_drones', 1)
        self.declare_parameter('namespaces', ['x500_1'])
        self.declare_parameter('px4_namespaces', ['px4_1'])
        self.declare_parameter('warmup_sec', 25.0)
        self.declare_parameter('takeoff_z', -2.0)
        self.declare_parameter('vertical_takeoff', True)
        self.declare_parameter('arm_offboard_max_retries', 6)
        self.declare_parameter('arm_retry_sleep', 0.6)
        self.declare_parameter('offboard_retry_sleep', 0.8)
        self.declare_parameter('arm_to_offboard_delay', 0.8)
        self.declare_parameter('inter_drone_gap', 0.2)
        self.declare_parameter('post_offboard_hold_sec', 3.0)
        self.declare_parameter('vehicle_status_timeout_sec', 4.0)
        self.declare_parameter('auto_arm_offboard', True)

        # Read parameters
        self.num_drones = self.get_parameter('num_drones').value
        self.namespaces = self.get_parameter('namespaces').value
        self.px4_namespaces = self.get_parameter('px4_namespaces').value
        self.warmup_sec = self.get_parameter('warmup_sec').value
        self.takeoff_z = self.get_parameter('takeoff_z').value
        self.vertical_takeoff = self.get_parameter('vertical_takeoff').value
        self.max_retries = self.get_parameter('arm_offboard_max_retries').value
        self.arm_retry_sleep = self.get_parameter('arm_retry_sleep').value
        self.offboard_retry_sleep = self.get_parameter('offboard_retry_sleep').value
        self.arm_to_offboard_delay = self.get_parameter('arm_to_offboard_delay').value
        self.inter_drone_gap = self.get_parameter('inter_drone_gap').value
        self.post_offboard_hold_sec = self.get_parameter('post_offboard_hold_sec').value
        self.status_timeout = self.get_parameter('vehicle_status_timeout_sec').value
        self.auto_arm_offboard = self.get_parameter('auto_arm_offboard').value

        # --- Warmup publishers ---
        self.warmup_pubs = {}
        for ns in self.namespaces:
            topic = f'/xtdrone2/{ns}/cmd_pose_local_ned'
            self.warmup_pubs[ns] = self.create_publisher(Pose, topic, 10)

        # --- Service clients (created on demand) ---
        self._cmd_clients = {}

        # --- State ---
        self._state = 'WARMUP'
        self._warmup_start = time.time()
        self._warmup_timer = self.create_timer(0.05, self._warmup_callback)  # 20Hz

        self.get_logger().info(
            f'MissionController started: {self.num_drones} drones, '
            f'warmup={self.warmup_sec}s, state=WARMUP'
        )

    # ---- Warmup ----

    def _warmup_callback(self):
        """Publish warmup setpoints at 20Hz for all drones."""
        for ns in self.namespaces:
            pose = Pose()
            if self.vertical_takeoff:
                pose.position.x = 0.0
                pose.position.y = 0.0
            pose.position.z = self.takeoff_z
            pose.orientation.w = 1.0
            self.warmup_pubs[ns].publish(pose)

        elapsed = time.time() - self._warmup_start
        if self._state == 'WARMUP' and elapsed >= self.warmup_sec:
            self.get_logger().info(f'Warmup done ({elapsed:.1f}s), starting ARM sequence')
            self._state = 'ARMING'
            self._warmup_timer.cancel()
            # Run ARM/OFFBOARD in a timer callback to avoid blocking
            self._arm_timer = self.create_timer(0.1, self._arming_step)
            self._arm_drone_idx = 0
            self._arm_attempt = 0

        elif self._state == 'OFFBOARD_HOLD':
            elapsed_hold = time.time() - self._hold_start
            if elapsed_hold >= self.post_offboard_hold_sec:
                self.get_logger().info(
                    f'OFFBOARD hold done ({elapsed_hold:.1f}s), starting mission'
                )
                self._state = 'MISSION'
                self._warmup_timer.cancel()
                self._start_mission()

    # ---- ARM / OFFBOARD ----

    def _get_cmd_client(self, ns):
        if ns not in self._cmd_clients:
            self._cmd_clients[ns] = self.create_client(
                XTD2Cmd, f'/xtdrone2/{ns}/cmd'
            )
        return self._cmd_clients[ns]

    def _arming_step(self):
        """Non-blocking ARM/OFFBOARD state machine, one step per timer tick."""
        self._arm_timer.cancel()

        if self._arm_drone_idx >= self.num_drones:
            # All drones done
            self.get_logger().info('All drones ARM+OFFBOARD sequence complete')
            self._state = 'OFFBOARD_HOLD'
            self._hold_start = time.time()
            self._warmup_timer = self.create_timer(0.05, self._warmup_callback)
            return

        ns = self.px4_namespaces[self._arm_drone_idx]  # service 在 /xtdrone2/px4_i/cmd
        drone_id = self._arm_drone_idx + 1

        if self._arm_attempt == 0:
            self.get_logger().info(f'[ARM] x500_{drone_id} starting ARM attempts')
            self._arm_attempt = 1

        # ARM
        client = self._get_cmd_client(ns)
        if not client.service_is_ready():
            client.wait_for_service(timeout_sec=2.0)

        request = XTD2Cmd.Request()
        request.command = 'ARM'
        future = client.call_async(request)
        future.add_done_callback(lambda f, idx=self._arm_drone_idx, att=self._arm_attempt:
            self._arm_done(f, idx, att))

    def _arm_done(self, future, drone_idx, attempt):
        ns = self.namespaces[drone_idx]
        drone_id = drone_idx + 1
        try:
            result = future.result()
            self.get_logger().info(f'[ARM] x500_{drone_id} ARM response: success={result.success}')
        except Exception as e:
            self.get_logger().warn(f'[ARM] x500_{drone_id} ARM call failed: {e}')

        # Check arming state via DDS topic
        px4_ns = self.px4_namespaces[drone_idx]
        armed = self._check_arming_state(px4_ns)

        if armed:
            self.get_logger().info(f'[ARM] x500_{drone_id} armed, sending OFFBOARD')
            self._send_offboard(drone_idx, attempt)
        else:
            if attempt < self.max_retries:
                self.get_logger().info(
                    f'[ARM] x500_{drone_id} not armed, retry {attempt+1}/{self.max_retries}'
                )
                self._arm_drone_idx = drone_idx
                self._arm_attempt = attempt + 1
                self._arm_timer = self.create_timer(self.arm_retry_sleep, self._arming_step)
            else:
                self.get_logger().warn(
                    f'[ARM] x500_{drone_id} ARM failed after {self.max_retries} retries, moving on'
                )
                self._arm_drone_idx = drone_idx + 1
                self._arm_attempt = 0
                self._arm_timer = self.create_timer(
                    self.arm_to_offboard_delay + self.inter_drone_gap,
                    self._arming_step
                )

    def _send_offboard(self, drone_idx, arm_attempt):
        ns = self.namespaces[drone_idx]
        drone_id = drone_idx + 1
        client = self._get_cmd_client(ns)

        request = XTD2Cmd.Request()
        request.command = 'OFFBOARD'
        future = client.call_async(request)
        future.add_done_callback(
            lambda f, idx=drone_idx, att=arm_attempt: self._offboard_done(f, idx, att)
        )

    def _offboard_done(self, future, drone_idx, arm_attempt):
        ns = self.namespaces[drone_idx]
        drone_id = drone_idx + 1
        try:
            result = future.result()
            self.get_logger().info(
                f'[OFFBOARD] x500_{drone_id} response: success={result.success}'
            )
        except Exception as e:
            self.get_logger().warn(f'[OFFBOARD] x500_{drone_id} call failed: {e}')
            result = None

        px4_ns = self.px4_namespaces[drone_idx]
        offboard_ok = self._check_offboard_state(px4_ns)

        if result and result.success and offboard_ok:
            self.get_logger().info(f'[OFFBOARD] x500_{drone_id} OFFBOARD success')
            self._arm_drone_idx = drone_idx + 1
            self._arm_attempt = 0
            self._arm_timer = self.create_timer(
                self.arm_to_offboard_delay + self.inter_drone_gap,
                self._arming_step
            )
        else:
            if arm_attempt < self.max_retries:
                self.get_logger().info(
                    f'[OFFBOARD] x500_{drone_id} not in OFFBOARD, retry {arm_attempt+1}/{self.max_retries}'
                )
                self._arm_drone_idx = drone_idx
                self._arm_attempt = arm_attempt + 1
                self._arm_timer = self.create_timer(self.offboard_retry_sleep, self._arming_step)
            else:
                self.get_logger().warn(
                    f'[OFFBOARD] x500_{drone_id} failed after {self.max_retries} retries, moving on'
                )
                self._arm_drone_idx = drone_idx + 1
                self._arm_attempt = 0
                self._arm_timer = self.create_timer(
                    self.arm_to_offboard_delay + self.inter_drone_gap,
                    self._arming_step
                )

    def _check_arming_state(self, px4_ns):
        """Check vehicle_status for arming_state == 2 (ARMED)."""
        from px4_msgs.msg import VehicleStatus
        topic = f'/{px4_ns}/fmu/out/vehicle_status_v4'
        msg = self._wait_for_topic(topic, VehicleStatus, self.status_timeout)
        if msg is not None:
            return msg.arming_state == 2
        # Fallback: try vehicle_status (without _v4)
        topic2 = f'/{px4_ns}/fmu/out/vehicle_status'
        msg = self._wait_for_topic(topic2, VehicleStatus, self.status_timeout)
        if msg is not None:
            return msg.arming_state == 2
        return False

    def _check_offboard_state(self, px4_ns):
        """Check vehicle_status for nav_state == 14 (OFFBOARD)."""
        from px4_msgs.msg import VehicleStatus
        topic = f'/{px4_ns}/fmu/out/vehicle_status_v4'
        msg = self._wait_for_topic(topic, VehicleStatus, self.status_timeout)
        if msg is not None:
            return msg.nav_state == 14
        topic2 = f'/{px4_ns}/fmu/out/vehicle_status'
        msg = self._wait_for_topic(topic2, VehicleStatus, self.status_timeout)
        if msg is not None:
            return msg.nav_state == 14
        return False

    def _wait_for_topic(self, topic, msg_type, timeout):
        """Wait for one message on a topic with timeout."""
        received = {'msg': None}
        done = {'flag': False}

        def cb(msg):
            if not done['flag']:
                received['msg'] = msg
                done['flag'] = True

        sub = self.create_subscription(msg_type, topic, cb, 10)
        start = time.time()
        while not done['flag'] and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.destroy_subscription(sub)
        return received['msg']

    # ---- Mission ----

    def _start_mission(self):
        """Start the multi_waypoint2 mission logic."""
        self.get_logger().info('Starting mission via multi_waypoint2')
        from xtd2_mission.multi_waypoint2 import main as mission_main
        try:
            mission_main()
        except Exception as e:
            self.get_logger().error(f'Mission failed: {e}')
        self._state = 'DONE'
        self.get_logger().info('MissionController DONE')


def main():
    rclpy.init(args=sys.argv)

    parser = argparse.ArgumentParser(description='XTDrone2 Mission Controller')
    # All parameters are passed via ROS 2 parameter system (--ros-args),
    # but we also accept a few CLI args for convenience.
    args, _ = parser.parse_known_args()

    node = MissionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
