#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter predicted dynamic-obstacle trajectories before collision checking."""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String


class TrajectoryFilter(Node):
    def __init__(
        self,
        input_topic='/xtdrone2/swarm/predicted_dynamic_obstacles',
        output_topic='/xtdrone2/swarm/filtered_predicted_dynamic_obstacles',
        state_topic='/xtdrone2/swarm/state_exchange',
        num_drones=5,
        min_confidence=0.0,
        min_speed=0.1,
        min_points=2,
        max_age=1.0,
        horizon=3.0,
        min_x=-100.0,
        max_x=100.0,
        min_y=-100.0,
        max_y=100.0,
        min_z=-10.0,
        max_z=10.0,
        merge_distance=0.8,
        relevance_radius=8.0,
        state_timeout=2.0,
    ):
        super().__init__('trajectory_filter')
        self.input_topic = str(input_topic)
        self.output_topic = str(output_topic)
        self.state_topic = str(state_topic)
        self.num_drones = int(num_drones)
        self.min_confidence = float(min_confidence)
        self.min_speed = float(min_speed)
        self.min_points = int(min_points)
        self.max_age = float(max_age)
        self.horizon = float(horizon)
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.min_z = float(min_z)
        self.max_z = float(max_z)
        self.merge_distance = float(merge_distance)
        self.relevance_radius = float(relevance_radius)
        self.state_timeout = float(state_timeout)

        self.declare_parameter('min_confidence', self.min_confidence)
        self.declare_parameter('min_speed', self.min_speed)
        self.declare_parameter('merge_distance', self.merge_distance)
        self.declare_parameter('relevance_radius', self.relevance_radius)
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.last_log_time = 0.0
        self.states = {}
        self.create_subscription(String, self.input_topic, self._cb, 10)
        self.create_subscription(String, self.state_topic, self._state_cb, 10)
        self.pub = self.create_publisher(String, self.output_topic, 10)

        self.get_logger().info(
            'trajectory_filter started: '
            f'input={self.input_topic}, output={self.output_topic}, '
            f'min_conf={self.min_confidence:.2f}, min_speed={self.min_speed:.2f}, '
            f'min_points={self.min_points}, max_age={self.max_age:.2f}, '
            f'horizon={self.horizon:.2f}, merge_distance={self.merge_distance:.2f}, '
            f'relevance_radius={self.relevance_radius:.2f}'
        )

    def _on_params_changed(self, params):
        for p in params:
            if p.name == 'min_confidence':
                self.min_confidence = max(0.0, min(1.0, float(p.value)))
            elif p.name == 'min_speed':
                self.min_speed = max(0.0, float(p.value))
            elif p.name == 'merge_distance':
                self.merge_distance = max(0.0, float(p.value))
            elif p.name == 'relevance_radius':
                self.relevance_radius = max(0.0, float(p.value))
        self.get_logger().info(
            'trajectory filter params updated: '
            f'min_conf={self.min_confidence:.2f}, min_speed={self.min_speed:.2f}, '
            f'merge_distance={self.merge_distance:.2f}, relevance_radius={self.relevance_radius:.2f}'
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
            self.get_logger().warn(f'trajectory filter state parse failed: {exc}')

    def _cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f'trajectory filter parse failed: {exc}')
            return

        tracks = list(data.get('tracks', []))
        filtered, stats = self._filter_tracks(tracks)
        now = time.time()
        out = {
            'stamp': now,
            'source_stamp': data.get('stamp', now),
            'input_count': len(tracks),
            'filtered_count': len(filtered),
            'filter_stats': stats,
            'tracks': filtered,
        }
        out_msg = String()
        out_msg.data = json.dumps(out, ensure_ascii=False)
        self.pub.publish(out_msg)

        if now - self.last_log_time >= 2.0:
            self.last_log_time = now
            self.get_logger().info(
                'trajectory filter: '
                f'in={len(tracks)}, out={len(filtered)}, '
                f'low_conf={stats["low_confidence"]}, slow={stats["slow"]}, '
                f'short={stats["short_prediction"]}, stale={stats["stale"]}, '
                f'out_of_bounds={stats["out_of_bounds"]}, irrelevant={stats["irrelevant"]}, '
                f'duplicate={stats["duplicate"]}, fresh_drones={stats["fresh_drones"]}'
            )

    def _filter_tracks(self, tracks):
        stats = {
            'low_confidence': 0,
            'slow': 0,
            'short_prediction': 0,
            'stale': 0,
            'out_of_bounds': 0,
            'irrelevant': 0,
            'duplicate': 0,
            'fresh_drones': 0,
        }
        fresh_states = self._fresh_states()
        stats['fresh_drones'] = len(fresh_states)
        candidates = []
        for track in tracks:
            if float(track.get('confidence', 0.0)) < self.min_confidence:
                stats['low_confidence'] += 1
                continue
            if float(track.get('speed', 0.0)) < self.min_speed:
                stats['slow'] += 1
                continue
            if float(track.get('last_seen_age', 0.0)) > self.max_age:
                stats['stale'] += 1
                continue

            predictions = [
                p for p in list(track.get('predictions') or [])
                if 0.0 < float(p.get('t', 0.0)) <= self.horizon
            ]
            if len(predictions) < self.min_points:
                stats['short_prediction'] += 1
                continue
            if not self._trajectory_in_bounds(predictions):
                stats['out_of_bounds'] += 1
                continue
            if fresh_states and not self._trajectory_relevant(predictions, fresh_states):
                stats['irrelevant'] += 1
                continue

            clean_track = dict(track)
            clean_track['predictions'] = predictions
            clean_track['trajectory_filter'] = {
                'min_confidence': self.min_confidence,
                'min_speed': self.min_speed,
                'horizon': self.horizon,
                'merge_distance': self.merge_distance,
                'relevance_radius': self.relevance_radius,
            }
            candidates.append(clean_track)

        candidates.sort(
            key=lambda t: (
                -float(t.get('confidence', 0.0)),
                -float(t.get('speed', 0.0)),
                int(t.get('track_id', 0)),
            )
        )

        kept = []
        for track in candidates:
            if self._is_duplicate(track, kept):
                stats['duplicate'] += 1
                continue
            kept.append(track)
        kept.sort(key=lambda t: int(t.get('track_id', 0)))
        return kept, stats

    def _fresh_states(self):
        now = time.time()
        return [
            state for state in self.states.values()
            if now - float(state.get('stamp', 0.0)) <= self.state_timeout
        ]

    def _trajectory_relevant(self, predictions, fresh_states):
        if self.relevance_radius <= 0.0:
            return True
        radius_sq = self.relevance_radius ** 2
        for state in fresh_states:
            for pred in predictions:
                t = float(pred.get('t', 0.0))
                dx = (float(state['x']) + float(state.get('vx', 0.0)) * t) - float(pred.get('x', 0.0))
                dy = (float(state['y']) + float(state.get('vy', 0.0)) * t) - float(pred.get('y', 0.0))
                dz = (float(state['z']) + float(state.get('vz', 0.0)) * t) - float(pred.get('z', 0.0))
                if dx * dx + dy * dy + dz * dz <= radius_sq:
                    return True
        return False

    def _trajectory_in_bounds(self, predictions):
        for p in predictions:
            x = float(p.get('x', 0.0))
            y = float(p.get('y', 0.0))
            z = float(p.get('z', 0.0))
            if (
                self.min_x <= x <= self.max_x
                and self.min_y <= y <= self.max_y
                and self.min_z <= z <= self.max_z
            ):
                return True
        return False

    def _is_duplicate(self, track, kept_tracks):
        if self.merge_distance <= 0.0:
            return False
        for kept in kept_tracks:
            dist = self._trajectory_distance(track, kept)
            if dist is not None and dist <= self.merge_distance:
                return True
        return False

    @staticmethod
    def _trajectory_distance(a, b):
        a_points = a.get('predictions') or []
        b_points = b.get('predictions') or []
        n = min(len(a_points), len(b_points), 3)
        if n <= 0:
            return None
        total = 0.0
        for i in range(n):
            ax, ay, az = TrajectoryFilter._point(a_points[i])
            bx, by, bz = TrajectoryFilter._point(b_points[i])
            total += math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)
        return total / n

    @staticmethod
    def _point(p):
        return (
            float(p.get('x', 0.0)),
            float(p.get('y', 0.0)),
            float(p.get('z', 0.0)),
        )


def main():
    parser = argparse.ArgumentParser(description='Predicted trajectory filter')
    parser.add_argument('--input-topic', type=str, default='/xtdrone2/swarm/predicted_dynamic_obstacles')
    parser.add_argument('--output-topic', type=str, default='/xtdrone2/swarm/filtered_predicted_dynamic_obstacles')
    parser.add_argument('--state-topic', type=str, default='/xtdrone2/swarm/state_exchange')
    parser.add_argument('--num-drones', type=int, default=5)
    parser.add_argument('--min-confidence', type=float, default=0.0)
    parser.add_argument('--min-speed', type=float, default=0.1)
    parser.add_argument('--min-points', type=int, default=2)
    parser.add_argument('--max-age', type=float, default=1.0)
    parser.add_argument('--horizon', type=float, default=3.0)
    parser.add_argument('--min-x', type=float, default=-100.0)
    parser.add_argument('--max-x', type=float, default=100.0)
    parser.add_argument('--min-y', type=float, default=-100.0)
    parser.add_argument('--max-y', type=float, default=100.0)
    parser.add_argument('--min-z', type=float, default=-10.0)
    parser.add_argument('--max-z', type=float, default=10.0)
    parser.add_argument('--merge-distance', type=float, default=0.8)
    parser.add_argument('--relevance-radius', type=float, default=8.0)
    parser.add_argument('--state-timeout', type=float, default=2.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = TrajectoryFilter(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        state_topic=args.state_topic,
        num_drones=args.num_drones,
        min_confidence=args.min_confidence,
        min_speed=args.min_speed,
        min_points=args.min_points,
        max_age=args.max_age,
        horizon=args.horizon,
        min_x=args.min_x,
        max_x=args.max_x,
        min_y=args.min_y,
        max_y=args.max_y,
        min_z=args.min_z,
        max_z=args.max_z,
        merge_distance=args.merge_distance,
        relevance_radius=args.relevance_radius,
        state_timeout=args.state_timeout,
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
