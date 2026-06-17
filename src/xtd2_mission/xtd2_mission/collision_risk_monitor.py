#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatiotemporal collision risk monitor.

This node is read-only with respect to flight control. It consumes current UAV
states and predicted dynamic-obstacle trajectories, then reports future
collision/warning events without modifying cmd_vel/cmd_pose.
"""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String


class CollisionRiskMonitor(Node):
    def __init__(
        self,
        num_drones=5,
        state_topic='/xtdrone2/swarm/state_exchange',
        predicted_topic='/xtdrone2/swarm/predicted_dynamic_obstacles',
        static_topic='/xtdrone2/swarm/tracked_static_obstacles',
        horizon=3.0,
        drone_radius=0.35,
        safety_margin=0.35,
        warning_margin=1.0,
        state_timeout=2.0,
        max_reports=30,
        rate_hz=10.0,
    ):
        super().__init__('collision_risk_monitor')
        self.num_drones = int(num_drones)
        self.state_topic = str(state_topic)
        self.predicted_topic = str(predicted_topic)
        self.static_topic = str(static_topic)
        self.horizon = float(horizon)
        self.drone_radius = float(drone_radius)
        self.safety_margin = float(safety_margin)
        self.warning_margin = float(warning_margin)
        self.state_timeout = float(state_timeout)
        self.max_reports = int(max_reports)
        self.rate_hz = float(rate_hz)

        self.declare_parameter('horizon', self.horizon)
        self.declare_parameter('drone_radius', self.drone_radius)
        self.declare_parameter('safety_margin', self.safety_margin)
        self.declare_parameter('warning_margin', self.warning_margin)
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.states = {}
        self.predicted_tracks = []
        self.static_tracks = []
        self.predicted_stamp = 0.0
        self.static_stamp = 0.0
        self.last_log_time = 0.0

        self.create_subscription(String, self.state_topic, self._state_cb, 10)
        self.create_subscription(String, self.predicted_topic, self._predicted_cb, 10)
        self.create_subscription(String, self.static_topic, self._static_cb, 10)
        self.risk_pub = self.create_publisher(
            String, '/xtdrone2/swarm/spatiotemporal_collision_risks', 10
        )
        self.timer = self.create_timer(max(0.02, 1.0 / self.rate_hz), self._run)

        self.get_logger().info(
            'collision_risk_monitor started: '
            f'num_drones={self.num_drones}, horizon={self.horizon:.2f}s, '
            f'predicted_topic={self.predicted_topic}, static_topic={self.static_topic}, '
            f'drone_radius={self.drone_radius:.2f}, safety_margin={self.safety_margin:.2f}, '
            f'warning_margin={self.warning_margin:.2f}'
        )

    def _on_params_changed(self, params):
        for p in params:
            if p.name == 'horizon':
                self.horizon = max(0.1, float(p.value))
            elif p.name == 'drone_radius':
                self.drone_radius = max(0.01, float(p.value))
            elif p.name == 'safety_margin':
                self.safety_margin = max(0.0, float(p.value))
            elif p.name == 'warning_margin':
                self.warning_margin = max(0.0, float(p.value))
        self.get_logger().info(
            'collision monitor params updated: '
            f'horizon={self.horizon:.2f}, drone_radius={self.drone_radius:.2f}, '
            f'safety_margin={self.safety_margin:.2f}, warning_margin={self.warning_margin:.2f}'
        )
        return SetParametersResult(successful=True)

    def _state_cb(self, msg):
        try:
            data = json.loads(msg.data)
            states = {}
            for item in data.get('states', []):
                drone_id = int(item.get('id', 0))
                if not (1 <= drone_id <= self.num_drones):
                    continue
                states[drone_id] = {
                    'id': drone_id,
                    'x': float(item.get('world_x', item.get('x', 0.0))),
                    'y': float(item.get('world_y', item.get('y', 0.0))),
                    'z': float(item.get('z', 0.0)),
                    'vx': float(item.get('vx', 0.0)),
                    'vy': float(item.get('vy', 0.0)),
                    'vz': float(item.get('vz', 0.0)),
                    'stamp': float(item.get('stamp', time.time())),
                }
            self.states = states
        except Exception as exc:
            self.get_logger().warn(f'collision monitor state parse failed: {exc}')

    def _predicted_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.predicted_tracks = list(data.get('tracks', []))
            self.predicted_stamp = float(data.get('stamp', time.time()))
        except Exception as exc:
            self.get_logger().warn(f'collision monitor prediction parse failed: {exc}')

    def _static_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.static_tracks = list(data.get('tracks', []))
            self.static_stamp = float(data.get('stamp', time.time()))
        except Exception as exc:
            self.get_logger().warn(f'collision monitor static parse failed: {exc}')

    def _run(self):
        now = time.time()
        fresh_states = {
            drone_id: state
            for drone_id, state in self.states.items()
            if now - float(state.get('stamp', 0.0)) <= self.state_timeout
        }
        risks = []

        tracks = []
        for track in self.predicted_tracks:
            item = dict(track)
            item.setdefault('class', 'dynamic')
            tracks.append(item)
        for track in self.static_tracks:
            item = dict(track)
            item.setdefault('class', 'static')
            item.setdefault('predictions', [])
            tracks.append(item)

        if fresh_states and tracks:
            for drone_id, state in fresh_states.items():
                for track in tracks:
                    risks.extend(self._check_pair(state, track))

        risks.sort(key=lambda item: (item['clearance'], item['time_to_event']))
        collision_events = [r for r in risks if r['level'] == 'collision']
        warning_events = [r for r in risks if r['level'] == 'warning']
        limited = risks[: max(0, self.max_reports)]

        msg = String()
        msg.data = json.dumps(
            {
                'stamp': now,
                'horizon': self.horizon,
                'drone_radius': self.drone_radius,
                'safety_margin': self.safety_margin,
                'warning_margin': self.warning_margin,
                'num_fresh_drones': len(fresh_states),
                'num_predicted_tracks': len(self.predicted_tracks),
                'num_static_tracks': len(self.static_tracks),
                'collision_count': len(collision_events),
                'warning_count': len(warning_events),
                'risks': limited,
            },
            ensure_ascii=False,
        )
        self.risk_pub.publish(msg)

        if now - self.last_log_time >= 1.5:
            self.last_log_time = now
            min_clearance = risks[0]['clearance'] if risks else None
            if min_clearance is None:
                self.get_logger().info(
                    'collision risk: no events, '
                    f'fresh_drones={len(fresh_states)}, predicted_tracks={len(self.predicted_tracks)}, '
                    f'static_tracks={len(self.static_tracks)}'
                )
            else:
                top = risks[0]
                self.get_logger().info(
                    'collision risk: '
                    f'collisions={len(collision_events)}, warnings={len(warning_events)}, '
                    f'min_clearance={min_clearance:.2f}, '
                    f'top=x500_{top["drone_id"]}->track_{top["track_id"]} '
                    f'class={top["track_class"]} name={top["source_name"]} '
                    f't={top["time_to_event"]:.2f}s level={top["level"]}'
                )

    def _check_pair(self, state, track):
        events = []
        track_id = int(track.get('track_id', 0))
        obs_radius = float(track.get('radius', 0.3))
        threshold = self.drone_radius + obs_radius + self.safety_margin
        warning_threshold = threshold + self.warning_margin
        predictions = track.get('predictions') or []

        if not predictions:
            predictions = [
                {
                    't': 0.0,
                    'x': float(track.get('x', 0.0)),
                    'y': float(track.get('y', 0.0)),
                    'z': float(track.get('z', 0.0)),
                    'radius': obs_radius,
                }
            ]

        for pred in predictions:
            t = float(pred.get('t', 0.0))
            if t < 0.0 or t > self.horizon:
                continue
            drone_pos = self._predict_drone_position(state, t)
            obs_pos = (
                float(pred.get('x', track.get('x', 0.0))),
                float(pred.get('y', track.get('y', 0.0))),
                float(pred.get('z', track.get('z', 0.0))),
            )
            dist = self._distance(drone_pos, obs_pos)
            clearance = dist - threshold
            if dist <= threshold:
                level = 'collision'
            elif dist <= warning_threshold:
                level = 'warning'
            else:
                continue

            events.append(
                {
                    'level': level,
                    'drone_id': int(state['id']),
                    'track_id': track_id,
                    'track_class': str(track.get('class', 'dynamic')),
                    'time_to_event': t,
                    'distance': dist,
                    'clearance': clearance,
                    'threshold': threshold,
                    'warning_threshold': warning_threshold,
                    'drone_position': {
                        'x': drone_pos[0],
                        'y': drone_pos[1],
                        'z': drone_pos[2],
                    },
                    'obstacle_position': {
                        'x': obs_pos[0],
                        'y': obs_pos[1],
                        'z': obs_pos[2],
                    },
                    'obstacle_radius': obs_radius,
                    'source_name': str(track.get('source_name', '')),
                }
            )
        return events

    @staticmethod
    def _predict_drone_position(state, t):
        return (
            float(state['x']) + float(state.get('vx', 0.0)) * t,
            float(state['y']) + float(state.get('vy', 0.0)) * t,
            float(state['z']) + float(state.get('vz', 0.0)) * t,
        )

    @staticmethod
    def _distance(a, b):
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )


def main():
    parser = argparse.ArgumentParser(description='Spatiotemporal collision risk monitor')
    parser.add_argument('--num-drones', type=int, default=5)
    parser.add_argument('--state-topic', type=str, default='/xtdrone2/swarm/state_exchange')
    parser.add_argument('--predicted-topic', type=str, default='/xtdrone2/swarm/predicted_dynamic_obstacles')
    parser.add_argument('--static-topic', type=str, default='/xtdrone2/swarm/tracked_static_obstacles')
    parser.add_argument('--horizon', type=float, default=3.0)
    parser.add_argument('--drone-radius', type=float, default=0.35)
    parser.add_argument('--safety-margin', type=float, default=0.35)
    parser.add_argument('--warning-margin', type=float, default=1.0)
    parser.add_argument('--state-timeout', type=float, default=2.0)
    parser.add_argument('--max-reports', type=int, default=30)
    parser.add_argument('--rate', type=float, default=10.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = CollisionRiskMonitor(
        num_drones=args.num_drones,
        state_topic=args.state_topic,
        predicted_topic=args.predicted_topic,
        static_topic=args.static_topic,
        horizon=args.horizon,
        drone_radius=args.drone_radius,
        safety_margin=args.safety_margin,
        warning_margin=args.warning_margin,
        state_timeout=args.state_timeout,
        max_reports=args.max_reports,
        rate_hz=args.rate,
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
