#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tracking-based static/dynamic obstacle separation.

This node intentionally does not use simulator labels such as ``sdf_dynamic``
for classification. It treats /xtdrone2/swarm/dynamic_obstacles as raw
observations, associates them across frames by position, estimates velocity,
and classifies tracks as static, dynamic, or unknown from temporal motion.
"""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String


class ObstacleTracker(Node):
    def __init__(
        self,
        input_topic='/xtdrone2/swarm/dynamic_obstacles',
        rate_hz=10.0,
        max_match_distance=1.8,
        dynamic_speed_threshold=0.15,
        static_speed_threshold=0.05,
        dynamic_confirm_frames=3,
        static_confirm_frames=6,
        max_missed_frames=8,
        velocity_alpha=0.45,
        prediction_horizon=3.0,
        prediction_dt=0.5,
        prediction_max_speed=3.0,
    ):
        super().__init__('obstacle_tracker')
        self.input_topic = str(input_topic)
        self.rate_hz = float(rate_hz)
        self.max_match_distance = float(max_match_distance)
        self.dynamic_speed_threshold = float(dynamic_speed_threshold)
        self.static_speed_threshold = float(static_speed_threshold)
        self.dynamic_confirm_frames = int(dynamic_confirm_frames)
        self.static_confirm_frames = int(static_confirm_frames)
        self.max_missed_frames = int(max_missed_frames)
        self.velocity_alpha = float(velocity_alpha)
        self.prediction_horizon = float(prediction_horizon)
        self.prediction_dt = float(prediction_dt)
        self.prediction_max_speed = float(prediction_max_speed)

        self.declare_parameter('max_match_distance', self.max_match_distance)
        self.declare_parameter('dynamic_speed_threshold', self.dynamic_speed_threshold)
        self.declare_parameter('static_speed_threshold', self.static_speed_threshold)
        self.declare_parameter('dynamic_confirm_frames', self.dynamic_confirm_frames)
        self.declare_parameter('static_confirm_frames', self.static_confirm_frames)
        self.declare_parameter('prediction_horizon', self.prediction_horizon)
        self.declare_parameter('prediction_dt', self.prediction_dt)
        self.declare_parameter('prediction_max_speed', self.prediction_max_speed)
        self.add_on_set_parameters_callback(self._on_params_changed)

        self.tracks = {}
        self.next_track_id = 1
        self.latest_observations = []
        self.latest_stamp = 0.0
        self.last_update_time = 0.0
        self.last_log_time = 0.0

        self.create_subscription(String, self.input_topic, self._obs_cb, 10)
        self.tracked_pub = self.create_publisher(String, '/xtdrone2/swarm/tracked_obstacles', 10)
        self.dynamic_pub = self.create_publisher(String, '/xtdrone2/swarm/tracked_dynamic_obstacles', 10)
        self.static_pub = self.create_publisher(String, '/xtdrone2/swarm/tracked_static_obstacles', 10)
        self.unknown_pub = self.create_publisher(String, '/xtdrone2/swarm/tracked_unknown_obstacles', 10)
        self.predicted_dynamic_pub = self.create_publisher(
            String, '/xtdrone2/swarm/predicted_dynamic_obstacles', 10
        )
        self.timer = self.create_timer(max(0.02, 1.0 / self.rate_hz), self._publish)

        self.get_logger().info(
            'obstacle_tracker started: '
            f'input={self.input_topic}, max_match={self.max_match_distance:.2f}, '
            f'dyn_thr={self.dynamic_speed_threshold:.2f}, static_thr={self.static_speed_threshold:.2f}, '
            f'dyn_frames={self.dynamic_confirm_frames}, static_frames={self.static_confirm_frames}, '
            f'prediction_horizon={self.prediction_horizon:.2f}s, prediction_dt={self.prediction_dt:.2f}s'
        )

    def _on_params_changed(self, params):
        for p in params:
            if p.name == 'max_match_distance':
                self.max_match_distance = max(0.1, float(p.value))
            elif p.name == 'dynamic_speed_threshold':
                self.dynamic_speed_threshold = max(0.0, float(p.value))
            elif p.name == 'static_speed_threshold':
                self.static_speed_threshold = max(0.0, float(p.value))
            elif p.name == 'dynamic_confirm_frames':
                self.dynamic_confirm_frames = max(1, int(p.value))
            elif p.name == 'static_confirm_frames':
                self.static_confirm_frames = max(1, int(p.value))
            elif p.name == 'prediction_horizon':
                self.prediction_horizon = max(0.0, float(p.value))
            elif p.name == 'prediction_dt':
                self.prediction_dt = max(0.05, float(p.value))
            elif p.name == 'prediction_max_speed':
                self.prediction_max_speed = max(0.0, float(p.value))
        self.get_logger().info(
            'tracker params updated: '
            f'max_match={self.max_match_distance:.2f}, '
            f'dyn_thr={self.dynamic_speed_threshold:.2f}, static_thr={self.static_speed_threshold:.2f}, '
            f'dyn_frames={self.dynamic_confirm_frames}, static_frames={self.static_confirm_frames}, '
            f'prediction_horizon={self.prediction_horizon:.2f}, prediction_dt={self.prediction_dt:.2f}'
        )
        return SetParametersResult(successful=True)

    def _obs_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f'obstacle_tracker parse failed: {exc}')
            return

        observations = []
        for idx, raw in enumerate(data.get('obstacles', []), start=1):
            try:
                observations.append(
                    {
                        'obs_index': idx,
                        'x': float(raw.get('x', 0.0)),
                        'y': float(raw.get('y', 0.0)),
                        'z': float(raw.get('z', 0.0)),
                        'radius': float(raw.get('radius', 0.3)),
                        # Kept for debugging/evaluation only; not used to classify.
                        'source_name': str(raw.get('name', '')),
                        'source_obs_id': int(raw.get('obs_id', idx)),
                    }
                )
            except Exception:
                continue

        now = time.time()
        dt = now - self.last_update_time if self.last_update_time > 0.0 else 0.0
        self.latest_observations = observations
        self.latest_stamp = float(data.get('stamp', now))
        self._update_tracks(observations, now, dt)
        self.last_update_time = now

    def _update_tracks(self, observations, now, dt):
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_obs = set(range(len(observations)))
        matches = []

        candidates = []
        for track_id, track in self.tracks.items():
            predicted = self._predicted_xyz(track, dt)
            for obs_idx, obs in enumerate(observations):
                dist = self._distance(predicted, (obs['x'], obs['y'], obs['z']))
                if dist <= self.max_match_distance:
                    candidates.append((dist, track_id, obs_idx))

        for _, track_id, obs_idx in sorted(candidates, key=lambda item: item[0]):
            if track_id not in unmatched_track_ids or obs_idx not in unmatched_obs:
                continue
            matches.append((track_id, obs_idx))
            unmatched_track_ids.remove(track_id)
            unmatched_obs.remove(obs_idx)

        for track_id, obs_idx in matches:
            self._update_track(track_id, observations[obs_idx], now)

        for obs_idx in sorted(unmatched_obs):
            self._create_track(observations[obs_idx], now)

        for track_id in list(unmatched_track_ids):
            track = self.tracks.get(track_id)
            if track is None:
                continue
            track['misses'] += 1
            track['last_update_age'] = now - track['last_seen']
            if track['misses'] > self.max_missed_frames:
                del self.tracks[track_id]

    def _create_track(self, obs, now):
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = {
            'track_id': track_id,
            'x': obs['x'],
            'y': obs['y'],
            'z': obs['z'],
            'vx': 0.0,
            'vy': 0.0,
            'vz': 0.0,
            'speed': 0.0,
            'radius': obs['radius'],
            'age': 1,
            'hits': 1,
            'misses': 0,
            'dynamic_hits': 0,
            'static_hits': 0,
            'class': 'unknown',
            'confidence': 0.0,
            'last_seen': now,
            'source_name': obs.get('source_name', ''),
            'source_obs_id': obs.get('source_obs_id', 0),
        }

    def _update_track(self, track_id, obs, now):
        track = self.tracks[track_id]
        dt = max(1e-3, now - float(track['last_seen']))
        raw_vx = (obs['x'] - track['x']) / dt
        raw_vy = (obs['y'] - track['y']) / dt
        raw_vz = (obs['z'] - track['z']) / dt
        alpha = min(max(self.velocity_alpha, 0.0), 1.0)
        track['vx'] = alpha * track['vx'] + (1.0 - alpha) * raw_vx
        track['vy'] = alpha * track['vy'] + (1.0 - alpha) * raw_vy
        track['vz'] = alpha * track['vz'] + (1.0 - alpha) * raw_vz
        track['speed'] = math.sqrt(track['vx'] ** 2 + track['vy'] ** 2 + track['vz'] ** 2)
        track['x'] = obs['x']
        track['y'] = obs['y']
        track['z'] = obs['z']
        track['radius'] = obs['radius']
        track['age'] += 1
        track['hits'] += 1
        track['misses'] = 0
        track['last_seen'] = now
        track['source_name'] = obs.get('source_name', '')
        track['source_obs_id'] = obs.get('source_obs_id', 0)

        if track['speed'] >= self.dynamic_speed_threshold:
            track['dynamic_hits'] += 1
            track['static_hits'] = 0
        elif track['speed'] <= self.static_speed_threshold:
            track['static_hits'] += 1
            track['dynamic_hits'] = 0
        else:
            track['dynamic_hits'] = max(0, track['dynamic_hits'] - 1)
            track['static_hits'] = max(0, track['static_hits'] - 1)

        if track['dynamic_hits'] >= self.dynamic_confirm_frames:
            track['class'] = 'dynamic'
            track['confidence'] = min(1.0, track['dynamic_hits'] / max(1.0, 2.0 * self.dynamic_confirm_frames))
        elif track['static_hits'] >= self.static_confirm_frames:
            track['class'] = 'static'
            track['confidence'] = min(1.0, track['static_hits'] / max(1.0, 2.0 * self.static_confirm_frames))
        else:
            track['class'] = 'unknown'
            track['confidence'] = 0.0

    def _publish(self):
        now = time.time()
        active_tracks = [
            self._track_msg(track, now)
            for track in self.tracks.values()
            if now - float(track['last_seen']) <= max(1.0, self.max_missed_frames / max(self.rate_hz, 1e-3))
        ]
        dynamic_tracks = [t for t in active_tracks if t['class'] == 'dynamic']
        static_tracks = [t for t in active_tracks if t['class'] == 'static']
        unknown_tracks = [t for t in active_tracks if t['class'] == 'unknown']

        self._publish_payload(self.tracked_pub, active_tracks, now)
        self._publish_payload(self.dynamic_pub, dynamic_tracks, now)
        self._publish_payload(self.static_pub, static_tracks, now)
        self._publish_payload(self.unknown_pub, unknown_tracks, now)
        self._publish_payload(self.predicted_dynamic_pub, dynamic_tracks, now, include_prediction=True)

        if now - self.last_log_time >= 2.0:
            self.last_log_time = now
            sample = dynamic_tracks[0] if dynamic_tracks else (unknown_tracks[0] if unknown_tracks else None)
            sample_text = ''
            if sample is not None:
                sample_text = (
                    f', sample_track={sample["track_id"]}:{sample["class"]} '
                    f'pos=({sample["x"]:.2f},{sample["y"]:.2f},{sample["z"]:.2f}) '
                    f'speed={sample["speed"]:.2f}'
                )
            self.get_logger().info(
                f'tracker summary: total={len(active_tracks)}, dynamic={len(dynamic_tracks)}, '
                f'static={len(static_tracks)}, unknown={len(unknown_tracks)}{sample_text}'
            )

    def _publish_payload(self, publisher, tracks, now, include_prediction=False):
        payload_tracks = tracks
        if include_prediction:
            payload_tracks = []
            for track in tracks:
                predicted = dict(track)
                predicted['prediction_model'] = 'constant_velocity'
                predicted['prediction_horizon'] = float(self.prediction_horizon)
                predicted['prediction_dt'] = float(self.prediction_dt)
                predicted['predictions'] = self._predict_trajectory(track)
                payload_tracks.append(predicted)

        msg = String()
        msg.data = json.dumps(
            {
                'stamp': now,
                'source_stamp': self.latest_stamp,
                'tracks': payload_tracks,
            },
            ensure_ascii=False,
        )
        publisher.publish(msg)

    def _track_msg(self, track, now):
        return {
            'track_id': int(track['track_id']),
            'class': str(track['class']),
            'confidence': float(track['confidence']),
            'x': float(track['x']),
            'y': float(track['y']),
            'z': float(track['z']),
            'vx': float(track['vx']),
            'vy': float(track['vy']),
            'vz': float(track['vz']),
            'speed': float(track['speed']),
            'radius': float(track['radius']),
            'age': int(track['age']),
            'hits': int(track['hits']),
            'misses': int(track['misses']),
            'last_seen_age': float(now - track['last_seen']),
            'source_name': str(track.get('source_name', '')),
            'source_obs_id': int(track.get('source_obs_id', 0)),
        }

    def _predict_trajectory(self, track):
        if self.prediction_horizon <= 0.0:
            return []

        speed = float(track.get('speed', 0.0))
        vx = float(track.get('vx', 0.0))
        vy = float(track.get('vy', 0.0))
        vz = float(track.get('vz', 0.0))
        if self.prediction_max_speed > 0.0 and speed > self.prediction_max_speed:
            scale = self.prediction_max_speed / max(speed, 1e-6)
            vx *= scale
            vy *= scale
            vz *= scale

        points = []
        step = max(0.05, float(self.prediction_dt))
        horizon = max(0.0, float(self.prediction_horizon))
        n_steps = int(math.floor(horizon / step + 1e-6))
        for k in range(1, n_steps + 1):
            t = k * step
            points.append(
                {
                    't': float(t),
                    'x': float(track['x'] + vx * t),
                    'y': float(track['y'] + vy * t),
                    'z': float(track['z'] + vz * t),
                    'radius': float(track['radius']),
                }
            )
        return points

    def _predicted_xyz(self, track, dt):
        dt = max(0.0, min(float(dt), 1.0))
        return (
            float(track['x']) + float(track['vx']) * dt,
            float(track['y']) + float(track['vy']) * dt,
            float(track['z']) + float(track['vz']) * dt,
        )

    @staticmethod
    def _distance(a, b):
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )


def main():
    parser = argparse.ArgumentParser(description='Tracking-based static/dynamic obstacle separator')
    parser.add_argument('--input-topic', type=str, default='/xtdrone2/swarm/dynamic_obstacles')
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--max-match-distance', type=float, default=1.8)
    parser.add_argument('--dynamic-speed-threshold', type=float, default=0.15)
    parser.add_argument('--static-speed-threshold', type=float, default=0.05)
    parser.add_argument('--dynamic-confirm-frames', type=int, default=3)
    parser.add_argument('--static-confirm-frames', type=int, default=6)
    parser.add_argument('--max-missed-frames', type=int, default=8)
    parser.add_argument('--velocity-alpha', type=float, default=0.45)
    parser.add_argument('--prediction-horizon', type=float, default=3.0)
    parser.add_argument('--prediction-dt', type=float, default=0.5)
    parser.add_argument('--prediction-max-speed', type=float, default=3.0)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = ObstacleTracker(
        input_topic=args.input_topic,
        rate_hz=args.rate,
        max_match_distance=args.max_match_distance,
        dynamic_speed_threshold=args.dynamic_speed_threshold,
        static_speed_threshold=args.static_speed_threshold,
        dynamic_confirm_frames=args.dynamic_confirm_frames,
        static_confirm_frames=args.static_confirm_frames,
        max_missed_frames=args.max_missed_frames,
        velocity_alpha=args.velocity_alpha,
        prediction_horizon=args.prediction_horizon,
        prediction_dt=args.prediction_dt,
        prediction_max_speed=args.prediction_max_speed,
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
