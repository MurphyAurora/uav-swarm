#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 bridge node for the EGO-like planner framework.

This node is intentionally added under scripts/ so it can be tried without
changing the installed xtd2_mission console scripts or launch files.  It reads
the same JSON-style swarm state / obstacle tracks used by the current branch and
publishes both a JSON planner report and optional Twist commands.

Example:
    python3 scripts/ego_like_planner_framework_node.py --num_drones 1 --publish_cmd 1
"""

import argparse
import json
import time
from typing import Dict, List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from planner_framework import PlannerConfig, PlannerState, Vec3
from planner_framework.perception_interface import PerceptionInterface
from planner_framework.planner_core import EgoLikePlannerCore


class EgoLikePlannerFrameworkNode(Node):
    def __init__(self, args):
        super().__init__('ego_like_planner_framework')
        self.args = args
        self.num_drones = int(args.num_drones)
        self.state_timeout = float(args.state_timeout)
        self.publish_cmd = bool(int(args.publish_cmd))

        self.config = PlannerConfig(
            horizon=float(args.horizon),
            dt=float(args.dt),
            cruise_speed=float(args.cruise_speed),
            lateral_speed=float(args.lateral_speed),
            vertical_speed=float(args.vertical_speed),
            max_speed=float(args.max_speed),
            max_accel=float(args.max_accel),
            drone_radius=float(args.drone_radius),
            obstacle_margin=float(args.obstacle_margin),
            hard_clearance=float(args.hard_clearance),
            emergency_clearance=float(args.emergency_clearance),
            static_hard_clearance=float(args.static_hard_clearance),
            static_emergency_clearance=float(args.static_emergency_clearance),
            local_goal_distance=float(args.local_goal_distance),
            goal_weight=float(args.goal_weight),
            clearance_weight=float(args.clearance_weight),
            ttc_weight=float(args.ttc_weight),
            smooth_weight=float(args.smooth_weight),
            output_alpha=float(args.output_alpha),
        )

        self.states: Dict[int, PlannerState] = {}
        self.static_tracks: List[Dict] = []
        self.dynamic_tracks: List[Dict] = []
        self.planners = {drone_id: EgoLikePlannerCore(self.config) for drone_id in range(1, self.num_drones + 1)}

        self.create_subscription(String, args.state_topic, self._state_cb, 10)
        self.create_subscription(String, args.predicted_topic, self._dynamic_cb, 10)
        self.create_subscription(String, args.static_topic, self._static_cb, 10)
        self.report_pub = self.create_publisher(String, args.output_topic, 10)
        self.cmd_pubs = {
            drone_id: self.create_publisher(Twist, f'/xtdrone2/x500_{drone_id}/ego_like_cmd_vel_ned', 10)
            for drone_id in range(1, self.num_drones + 1)
        }
        self.timer = self.create_timer(float(args.period), self._run)
        self.last_log_time = 0.0
        self.get_logger().info(
            'ego_like_planner_framework started: '
            f'num_drones={self.num_drones}, publish_cmd={int(self.publish_cmd)}, '
            f'output={args.output_topic}, cmd_topic=/xtdrone2/x500_i/ego_like_cmd_vel_ned'
        )

    def _state_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            states = {}
            for raw in data.get('states', []):
                state = PlannerState.from_mapping(raw)
                if 1 <= state.drone_id <= self.num_drones:
                    states[state.drone_id] = state
            self.states = states
        except Exception as exc:
            self.get_logger().warn(f'framework state parse failed: {exc}')

    def _dynamic_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.dynamic_tracks = list(data.get('tracks', []))
        except Exception as exc:
            self.get_logger().warn(f'framework dynamic track parse failed: {exc}')

    def _static_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.static_tracks = list(data.get('tracks', []))
        except Exception as exc:
            self.get_logger().warn(f'framework static track parse failed: {exc}')

    def _target_for_drone(self, drone_id: int) -> Vec3:
        # Keep the same simple formation target convention as the existing
        # primitive selector; later this should be replaced by goal_manager input.
        offset = (drone_id - int(self.args.leader_id)) * float(self.args.target_y_spacing)
        return Vec3(float(self.args.target_x), float(self.args.target_y_base) + offset, float(self.args.target_z))

    def _run(self) -> None:
        now = time.time()
        reports = []
        raw_states = [
            {
                'id': state.drone_id,
                'x': state.position.x,
                'y': state.position.y,
                'z': state.position.z,
                'vx': state.velocity.x,
                'vy': state.velocity.y,
                'vz': state.velocity.z,
                'stamp': state.stamp,
            }
            for state in self.states.values()
        ]

        for drone_id, state in sorted(self.states.items()):
            if now - state.stamp > self.state_timeout:
                continue
            perception = PerceptionInterface(
                drone_radius=self.config.drone_radius,
                default_obstacle_radius=0.3,
                near_field_distance=self.config.emergency_clearance,
            )
            perception.update_static_tracks(self.static_tracks, stamp=now)
            perception.update_dynamic_tracks(self.dynamic_tracks, stamp=now)
            perception.update_swarm_states(raw_states, own_id=drone_id, state_timeout=self.state_timeout)
            perception_data = perception.build(current_position=state.position)
            report = self.planners[drone_id].plan(state, self._target_for_drone(drone_id), perception_data)
            reports.append(report)
            if self.publish_cmd:
                self._publish_cmd(drone_id, report)

        out = String()
        out.data = json.dumps(
            {
                'stamp': now,
                'framework': 'ego_like_planner_framework',
                'num_reports': len(reports),
                'reports': reports,
            },
            ensure_ascii=False,
        )
        self.report_pub.publish(out)

        if now - self.last_log_time > 1.5:
            self.last_log_time = now
            if reports:
                best = min(reports, key=lambda item: float(item.get('best', {}).get('score', 0.0) if item.get('best') else 99999.0))
                cmd = best.get('command', {})
                self.get_logger().info(
                    'ego-like framework: '
                    f'reports={len(reports)}, best=x500_{best.get("drone_id")}, '
                    f'mode={cmd.get("mode")}, source={cmd.get("source_trajectory")}, '
                    f'safe={best.get("safe_count")}, feasible={best.get("feasible_count")}'
                )
            else:
                self.get_logger().info('ego-like framework: no fresh states')

    def _publish_cmd(self, drone_id: int, report: Dict) -> None:
        cmd = report.get('command', {}).get('velocity', {})
        msg = Twist()
        msg.linear.x = float(cmd.get('x', 0.0))
        msg.linear.y = float(cmd.get('y', 0.0))
        msg.linear.z = float(cmd.get('z', 0.0))
        pub = self.cmd_pubs.get(int(drone_id))
        if pub:
            pub.publish(msg)


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Run the EGO-like planner framework bridge node.')
    parser.add_argument('--num_drones', type=int, default=1)
    parser.add_argument('--state_topic', default='/xtdrone2/swarm/state_exchange')
    parser.add_argument('--predicted_topic', default='/xtdrone2/swarm/filtered_predicted_dynamic_obstacles')
    parser.add_argument('--static_topic', default='/xtdrone2/swarm/tracked_static_obstacles')
    parser.add_argument('--output_topic', default='/xtdrone2/swarm/ego_like_planner_framework')
    parser.add_argument('--publish_cmd', default='0', help='1 publishes /xtdrone2/x500_i/ego_like_cmd_vel_ned; default only publishes report')
    parser.add_argument('--period', type=float, default=0.1)
    parser.add_argument('--state_timeout', type=float, default=2.0)
    parser.add_argument('--target_x', type=float, default=30.0)
    parser.add_argument('--target_y_base', type=float, default=26.0)
    parser.add_argument('--target_y_spacing', type=float, default=1.8)
    parser.add_argument('--target_z', type=float, default=-3.0)
    parser.add_argument('--leader_id', type=int, default=3)
    parser.add_argument('--horizon', type=float, default=2.5)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--cruise_speed', type=float, default=0.7)
    parser.add_argument('--lateral_speed', type=float, default=0.45)
    parser.add_argument('--vertical_speed', type=float, default=0.35)
    parser.add_argument('--max_speed', type=float, default=1.0)
    parser.add_argument('--max_accel', type=float, default=0.8)
    parser.add_argument('--drone_radius', type=float, default=0.35)
    parser.add_argument('--obstacle_margin', type=float, default=0.45)
    parser.add_argument('--hard_clearance', type=float, default=0.25)
    parser.add_argument('--emergency_clearance', type=float, default=0.8)
    parser.add_argument('--static_hard_clearance', type=float, default=0.6)
    parser.add_argument('--static_emergency_clearance', type=float, default=1.2)
    parser.add_argument('--local_goal_distance', type=float, default=3.0)
    parser.add_argument('--goal_weight', type=float, default=1.0)
    parser.add_argument('--clearance_weight', type=float, default=8.0)
    parser.add_argument('--ttc_weight', type=float, default=4.0)
    parser.add_argument('--smooth_weight', type=float, default=0.4)
    parser.add_argument('--output_alpha', type=float, default=0.55)
    return parser


def main():
    parser = build_arg_parser()
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = EgoLikePlannerFrameworkNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
