#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 bridge node for the EGO-like planner framework."""

import argparse
import json
import os
import time
from typing import Dict, List

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from planner_framework import PlannerConfig, PlannerState, Vec3
from planner_framework.goal_manager import LocalGoalManager
from planner_framework.perception_interface import PerceptionInterface
from planner_framework.planner_core import EgoLikePlannerCore

from scripts.cmd_debug_logger import CmdDebugLogger

class EgoLikePlannerFrameworkNode(Node):
    def __init__(self, args):
        super().__init__('ego_like_planner_framework')
        self.args = args
        self.num_drones = int(args.num_drones)
        self.state_timeout = float(args.state_timeout)
        self.publish_cmd = bool(int(args.publish_cmd))
        self.cmd_topic_template = str(args.cmd_topic_template or '/xtdrone2/x500_{id}/ego_like_cmd_vel_ned')
        self.manual_waypoints = self._parse_waypoints(args.waypoints, default_z=float(args.target_z))
        self.log_file_path = str(args.log_file or '').strip()
        self.log_fp = None

        if self.log_file_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file_path)), exist_ok=True)
            self.log_fp = open(self.log_file_path, 'a')

        self.debug_logger = CmdDebugLogger()

        self.config = PlannerConfig(
            horizon=float(args.horizon), dt=float(args.dt),
            cruise_speed=float(args.cruise_speed), lateral_speed=float(args.lateral_speed),
            vertical_speed=float(args.vertical_speed), max_speed=float(args.max_speed),
            max_accel=float(args.max_accel), drone_radius=float(args.drone_radius),
            obstacle_margin=float(args.obstacle_margin), hard_clearance=float(args.hard_clearance),
            emergency_clearance=float(args.emergency_clearance),
            static_hard_clearance=float(args.static_hard_clearance),
            static_emergency_clearance=float(args.static_emergency_clearance),
            local_goal_distance=float(args.local_goal_distance), goal_weight=float(args.goal_weight),
            clearance_weight=float(args.clearance_weight), ttc_weight=float(args.ttc_weight),
            smooth_weight=float(args.smooth_weight), output_alpha=float(args.output_alpha),
        )

        self.states = {}
        self.static_tracks = []
        self.dynamic_tracks = []
        self.planners = {
            i: EgoLikePlannerCore(self.config, waypoints=self.manual_waypoints)
            for i in range(1, self.num_drones + 1)
        }

        self.create_subscription(String, args.state_topic, self._state_cb, 10)
        self.create_subscription(String, args.predicted_topic, self._dynamic_cb, 10)
        self.create_subscription(String, args.static_topic, self._static_cb, 10)
        self.report_pub = self.create_publisher(String, args.output_topic, 10)
        self.cmd_pubs = {
            i: self.create_publisher(Twist, self._cmd_topic_for(i), 10)
            for i in range(1, self.num_drones + 1)
        }
        self.timer = self.create_timer(float(args.period), self._run)

    def _cmd_topic_for(self, i):
        return self.cmd_topic_template.format(id=i)

    def _parse_waypoints(self, raw, default_z):
        ws = LocalGoalManager.parse_waypoints(raw or '')
        return [Vec3(w.x, w.y, w.z if abs(w.z) > 1e-9 else float(default_z)) for w in ws]

    def _state_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.states = {s['id']: PlannerState.from_mapping(s) for s in data.get('states', [])}
        except:
            pass

    def _dynamic_cb(self, msg):
        try:
            self.dynamic_tracks = json.loads(msg.data).get('tracks', [])
        except:
            pass

    def _static_cb(self, msg):
        try:
            self.static_tracks = json.loads(msg.data).get('tracks', [])
        except:
            pass

    def _target(self, i):
        off = (i - int(self.args.leader_id)) * float(self.args.target_y_spacing)
        return Vec3(self.args.target_x, self.args.target_y_base + off, self.args.target_z)

    def _run(self):
        now = time.time()
        reports = []

        raw = [{'id':s.drone_id,'x':s.position.x,'y':s.position.y,'z':s.position.z,
                'vx':s.velocity.x,'vy':s.velocity.y,'vz':s.velocity.z,'stamp':s.stamp}
               for s in self.states.values()]

        for i, s in self.states.items():
            if now - s.stamp > self.state_timeout:
                continue

            perception = PerceptionInterface(self.config.drone_radius,0.3,self.config.emergency_clearance)
            perception.update_static_tracks(self.static_tracks, now)
            perception.update_dynamic_tracks(self.dynamic_tracks, now)
            perception.update_swarm_states(raw, i, self.state_timeout)
            pd = perception.build(s.position)

            report = self.planners[i].plan(s, self._target(i), pd)
            reports.append(report)

            # DEBUG LOG (minimal)
            try:
                cmd = report.get('command', {}).get('velocity', {})
                class C: pass
                c = C()
                c.linear = type('L', (), cmd)()

                self.debug_logger.log(
                    primitive=None,
                    safe=None,
                    cmd=c,
                    mode=report.get('command', {}).get('mode'),
                    avoid_source=report.get('command', {}).get('source_trajectory')
                )
            except:
                pass

            if self.publish_cmd:
                self._publish(i, report)

        out = String()
        out.data = json.dumps({'framework':'ego_like','reports':reports}, ensure_ascii=False)
        self.report_pub.publish(out)

    def _publish(self, i, report):
        v = report.get('command', {}).get('velocity', {})
        m = Twist()
        m.linear.x = float(v.get('x',0))
        m.linear.y = float(v.get('y',0))
        m.linear.z = float(v.get('z',0))
        self.cmd_pubs[i].publish(m)

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--num_drones', type=int, default=1)
    p.add_argument('--state_topic', default='/xtdrone2/swarm/state_exchange')
    p.add_argument('--predicted_topic', default='/xtdrone2/swarm/filtered_predicted_dynamic_obstacles')
    p.add_argument('--static_topic', default='/xtdrone2/swarm/tracked_static_obstacles')
    p.add_argument('--output_topic', default='/xtdrone2/swarm/ego_like_planner_framework')
    p.add_argument('--publish_cmd', default='0')
    p.add_argument('--cmd_topic_template', default='/xtdrone2/x500_{id}/ego_like_cmd_vel_ned')
    p.add_argument('--waypoints', default='')
    p.add_argument('--period', type=float, default=0.1)
    p.add_argument('--state_timeout', type=float, default=2.0)
    p.add_argument('--target_x', type=float, default=30)
    p.add_argument('--target_y_base', type=float, default=26)
    p.add_argument('--target_y_spacing', type=float, default=1.8)
    p.add_argument('--target_z', type=float, default=-3)
    p.add_argument('--leader_id', type=int, default=3)
    p.add_argument('--horizon', type=float, default=2.5)
    p.add_argument('--dt', type=float, default=0.5)
    p.add_argument('--cruise_speed', type=float, default=0.7)
    p.add_argument('--lateral_speed', type=float, default=0.45)
    p.add_argument('--vertical_speed', type=float, default=0.35)
    p.add_argument('--max_speed', type=float, default=1.0)
    p.add_argument('--max_accel', type=float, default=0.8)
    p.add_argument('--drone_radius', type=float, default=0.35)
    p.add_argument('--obstacle_margin', type=float, default=0.45)
    p.add_argument('--hard_clearance', type=float, default=0.25)
    p.add_argument('--emergency_clearance', type=float, default=0.8)
    p.add_argument('--static_hard_clearance', type=float, default=0.6)
    p.add_argument('--static_emergency_clearance', type=float, default=1.2)
    p.add_argument('--local_goal_distance', type=float, default=3.0)
    p.add_argument('--goal_weight', type=float, default=1.0)
    p.add_argument('--clearance_weight', type=float, default=8.0)
    p.add_argument('--ttc_weight', type=float, default=4.0)
    p.add_argument('--smooth_weight', type=float, default=0.4)
    p.add_argument('--output_alpha', type=float, default=0.55)
    return p

def main():
    args, ros = build_arg_parser().parse_known_args()
    rclpy.init(args=ros)
    node = EgoLikePlannerFrameworkNode(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
