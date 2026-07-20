#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select lightweight UAV motion primitives using predicted obstacle risk.

This node is intentionally advisory: it publishes recommended primitives but
does not command PX4 or override mission control.
"""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

from xtd2_mission.cost_function import risk_aware_altitude_cost, transition_cost
from xtd2_mission.motion_primitive_library import MotionPrimitiveLibrary, norm3


class MotionPrimitiveSelector(Node):
    def __init__(
        self,
        num_drones=5,
        state_topic='/xtdrone2/swarm/state_exchange',
        predicted_topic='/xtdrone2/swarm/filtered_predicted_dynamic_obstacles',
        static_topic='/xtdrone2/swarm/tracked_static_obstacles',
        output_topic='/xtdrone2/swarm/selected_motion_primitives',
        target_x=30.0,
        target_y_base=26.0,
        target_y_spacing=1.8,
        target_z=-3.0,
        leader_id=3,
        horizon=2.5,
        dt=0.5,
        cruise_speed=0.7,
        lateral_speed=0.45,
        vertical_speed=0.35,
        max_speed=1.0,
        drone_radius=0.35,
        safety_margin=0.45,
        risk_radius=5.0,
        state_timeout=2.0,
        target_weight=1.0,
        risk_weight=8.0,
        smooth_weight=0.3,
        transition_weight=1.2,
        altitude_weight=1.4,
        altitude_clearance=0.7,
        publish_shadow_cmd=True,
        include_static=True,
        include_other_drones=True,
        emergency_clearance=0.8,
        projection_alpha=1.2,
        projection_iterations=2,
        projection_max_constraints=16,
        escape_clearance=0.4,
        hard_clearance=0.2,
        static_hard_clearance=0.6,
        static_emergency_clearance=1.2,
    ):
        super().__init__('motion_primitive_selector')
        self.num_drones = int(num_drones)
        self.state_topic = str(state_topic)
        self.predicted_topic = str(predicted_topic)
        self.static_topic = str(static_topic)
        self.output_topic = str(output_topic)
        self.target_x = float(target_x)
        self.target_y_base = float(target_y_base)
        self.target_y_spacing = float(target_y_spacing)
        self.target_z = float(target_z)
        self.leader_id = int(leader_id)
        self.drone_radius = float(drone_radius)
        self.safety_margin = float(safety_margin)
        self.risk_radius = float(risk_radius)
        self.state_timeout = float(state_timeout)
        self.target_weight = float(target_weight)
        self.risk_weight = float(risk_weight)
        self.smooth_weight = float(smooth_weight)
        self.transition_weight = float(transition_weight)
        self.altitude_weight = float(altitude_weight)
        self.altitude_clearance = float(altitude_clearance)
        self.publish_shadow_cmd = bool(publish_shadow_cmd)
        self.include_static = bool(include_static)
        self.include_other_drones = bool(include_other_drones)
        self.emergency_clearance = float(emergency_clearance)
        self.projection_alpha = float(projection_alpha)
        self.projection_iterations = max(1, int(projection_iterations))
        self.projection_max_constraints = max(1, int(projection_max_constraints))
        self.escape_clearance = float(escape_clearance)
        self.hard_clearance = float(hard_clearance)
        self.static_hard_clearance = float(static_hard_clearance)
        self.static_emergency_clearance = float(static_emergency_clearance)
        self.library = MotionPrimitiveLibrary(
            horizon=horizon,
            dt=dt,
            cruise_speed=cruise_speed,
            lateral_speed=lateral_speed,
            vertical_speed=vertical_speed,
            max_speed=max_speed,
        )

        self.states = {}
        self.predicted_tracks = []
        self.static_tracks = []
        self.static_stamp = 0.0
        self.last_log_time = 0.0
        self.last_primitives = {}

        self.create_subscription(String, self.state_topic, self._state_cb, 10)
        self.create_subscription(String, self.predicted_topic, self._predicted_cb, 10)
        self.create_subscription(String, self.static_topic, self._static_cb, 10)
        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.cmd_pubs = {
            i: self.create_publisher(Twist, f'/xtdrone2/x500_{i}/primitive_cmd_vel_ned', 10)
            for i in range(1, self.num_drones + 1)
        }
        self.timer = self.create_timer(0.1, self._run)

        self.get_logger().info(
            'motion_primitive_selector started: '
            f'num={self.num_drones}, predicted_topic={self.predicted_topic}, '
            f'static_topic={self.static_topic}, '
            f'horizon={horizon:.2f}, dt={dt:.2f}, risk_radius={self.risk_radius:.2f}, '
            f'advisory_only=1, publish_shadow_cmd={int(self.publish_shadow_cmd)}, '
            f'include_static={int(self.include_static)}, include_other_drones={int(self.include_other_drones)}, '
            f'emergency_clearance={self.emergency_clearance:.2f}, '
            f'projection_alpha={self.projection_alpha:.2f}, '
            f'escape_clearance={self.escape_clearance:.2f}, hard_clearance={self.hard_clearance:.2f}, '
            f'static_hard_clearance={self.static_hard_clearance:.2f}, '
            f'static_emergency_clearance={self.static_emergency_clearance:.2f}, '
            f'transition_weight={self.transition_weight:.2f}, '
            f'altitude_weight={self.altitude_weight:.2f}, altitude_clearance={self.altitude_clearance:.2f}'
        )

    def _state_cb(self, msg):
        try:
            data = json.loads(msg.data)
            states = {}
            for raw in data.get('states', []):
                drone_id = int(raw.get('id', 0))
                if not (1 <= drone_id <= self.num_drones):
                    continue
                states[drone_id] = {
                    'id': drone_id,
                    'x': float(raw.get('world_x', raw.get('x', 0.0))),
                    'y': float(raw.get('world_y', raw.get('y', 0.0))),
                    'z': float(raw.get('z', 0.0)),
                    'vx': float(raw.get('vx', 0.0)),
                    'vy': float(raw.get('vy', 0.0)),
                    'vz': float(raw.get('vz', 0.0)),
                    'stamp': float(raw.get('stamp', time.time())),
                }
            self.states = states
        except Exception as exc:
            self.get_logger().warn(f'primitive selector state parse failed: {exc}')

    def _predicted_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.predicted_tracks = list(data.get('tracks', []))
        except Exception as exc:
            self.get_logger().warn(f'primitive selector prediction parse failed: {exc}')

    def _static_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.static_stamp = float(data.get('stamp', time.time()))
            self.static_tracks = list(data.get('tracks', []))
        except Exception as exc:
            self.get_logger().warn(f'primitive selector static parse failed: {exc}')

    def _run(self):
        now = time.time()
        fresh_states = [
            state for state in self.states.values()
            if now - float(state.get('stamp', 0.0)) <= self.state_timeout
        ]
        selections = []
        fresh_ids = set()
        for state in sorted(fresh_states, key=lambda item: int(item['id'])):
            drone_id = int(state['id'])
            fresh_ids.add(drone_id)
            target = self._target_for_drone(drone_id)
            selection = self._select_for_drone(state, target)
            selections.append(selection)
            self.last_primitives[drone_id] = {
                'name': selection['primitive'],
                'vx': selection['recommended_velocity_ned']['vx'],
                'vy': selection['recommended_velocity_ned']['vy'],
                'vz': selection['recommended_velocity_ned']['vz'],
            }
        for stale_id in list(self.last_primitives.keys()):
            if stale_id not in fresh_ids:
                self.last_primitives.pop(stale_id, None)

        msg = String()
        msg.data = json.dumps(
            {
                'stamp': now,
                'advisory_only': True,
                'num_fresh_drones': len(fresh_states),
                'num_predicted_tracks': len(self.predicted_tracks),
                'selections': selections,
            },
            ensure_ascii=False,
        )
        self.pub.publish(msg)
        if self.publish_shadow_cmd:
            self._publish_shadow_cmds(selections)

        if now - self.last_log_time >= 1.5:
            self.last_log_time = now
            if selections:
                best = min(selections, key=lambda item: float(item.get('score', 0.0)))
                most_risky = max(
                    selections,
                    key=lambda item: (
                        float(item.get('risk_cost', 0.0)),
                        -float(item.get('min_clearance', 999.0)),
                    ),
                )
                self.get_logger().info(
                    'primitive selector: '
                    f'fresh_drones={len(fresh_states)}, predicted_tracks={len(self.predicted_tracks)}, '
                    f'static_tracks={len(self.static_tracks)}, '
                    f'best=x500_{best["drone_id"]}:{best["primitive"]} '
                    f'score={best["score"]:.2f}, risk={best["risk_cost"]:.2f}, '
                    f'alt={best.get("altitude_cost", 0.0):.2f}, trans={best.get("transition_cost", 0.0):.2f}, '
                    f'clearance={best["min_clearance"]:.2f}, '
                    f'source={best.get("min_clearance_source", "unknown")}, '
                    f'static_clearance={best.get("min_static_clearance", 999.0):.2f}, '
                    f'feasible={int(best["feasible"])}/{best["feasible_count"]}, '
                    f'safe={best.get("safe_count", 0)}, emergency={int(best.get("emergency_mode", False))}, '
                    f'projected={int(best.get("projected", False))}, '
                    f'escape={int(best.get("escape_mode", False))}, '
                    f'active_constraints={best.get("active_constraints", 0)}; '
                    f'most_risky=x500_{most_risky["drone_id"]}:{most_risky["primitive"]} '
                    f'score={most_risky["score"]:.2f}, risk={most_risky["risk_cost"]:.2f}, '
                    f'alt={most_risky.get("altitude_cost", 0.0):.2f}, trans={most_risky.get("transition_cost", 0.0):.2f}, '
                    f'clearance={most_risky["min_clearance"]:.2f}, '
                    f'source={most_risky.get("min_clearance_source", "unknown")}, '
                    f'static_clearance={most_risky.get("min_static_clearance", 999.0):.2f}, '
                    f'feasible={int(most_risky["feasible"])}/{most_risky["feasible_count"]}'
                )
            else:
                self.get_logger().info(
                    'primitive selector: no fresh states, '
                    f'predicted_tracks={len(self.predicted_tracks)}, static_tracks={len(self.static_tracks)}'
                )

    def _publish_shadow_cmds(self, selections):
        for selection in selections:
            drone_id = int(selection.get('drone_id', 0))
            pub = self.cmd_pubs.get(drone_id)
            if pub is None:
                continue
            vel = selection.get('recommended_velocity_ned') or {}
            msg = Twist()
            msg.linear.x = float(vel.get('vx', 0.0))
            msg.linear.y = float(vel.get('vy', 0.0))
            msg.linear.z = float(vel.get('vz', 0.0))
            pub.publish(msg)

    def _select_for_drone(self, state, target):
        previous = self.last_primitives.get(int(state['id']))
        primitives = self.library.generate(state, target)
        scored = [self._score_primitive(state, target, primitive, previous) for primitive in primitives]
        feasible = [item for item in scored if item['feasible']]
        safe = [
            item for item in feasible
            if item['min_clearance'] >= self.emergency_clearance
            and item['min_static_clearance'] >= self.static_emergency_clearance
        ]
        if safe:
            pool = sorted(safe, key=lambda item: item['score'])
            emergency_mode = False
        elif feasible:
            pool = sorted(feasible, key=lambda item: (-item['min_clearance'], item['score']))
            emergency_mode = True
        else:
            pool = sorted(scored, key=lambda item: (-item['min_clearance'], item['score']))
            emergency_mode = True
        scored.sort(key=lambda item: (not item['feasible'], item['score']))
        best = pool[0]
        projected = False
        escape_mode = False
        active_constraints = 0
        projected_clearance = best['min_clearance']
        if emergency_mode:
            pvx, pvy, pvz, active_constraints = self._project_velocity(
                state,
                (best['vx'], best['vy'], best['vz']),
            )
            primitive = self._constant_velocity_primitive(
                f'projected_{best["name"]}',
                state,
                pvx,
                pvy,
                pvz,
            )
            projected_eval = self._score_primitive(state, target, primitive, previous)
            best = projected_eval
            projected = True
            projected_clearance = projected_eval['min_clearance']
            if (
                not projected_eval['feasible']
                or projected_clearance < max(self.escape_clearance, self.emergency_clearance)
                or projected_eval['min_static_clearance'] < self.static_emergency_clearance
            ):
                escape = self._select_escape_primitive(scored)
                if escape is not None and escape['min_clearance'] > projected_clearance:
                    best = dict(escape)
                    best['name'] = f'escape_{best["name"]}'
                    escape_mode = True
            if not best['feasible']:
                brake = self._score_primitive(
                    state,
                    target,
                    self._constant_velocity_primitive('emergency_brake', state, 0.0, 0.0, 0.0),
                    previous,
                )
                if brake['min_clearance'] >= best['min_clearance']:
                    best = brake
                    escape_mode = True
        return {
            'drone_id': int(state['id']),
            'target': target,
            'primitive': best['name'],
            'feasible': best['feasible'],
            'feasible_count': len(feasible),
            'safe_count': len(safe),
            'emergency_mode': emergency_mode,
            'projected': projected,
            'escape_mode': escape_mode,
            'active_constraints': active_constraints,
            'projected_clearance': projected_clearance,
            'score': best['score'],
            'target_cost': best['target_cost'],
            'risk_cost': best['risk_cost'],
            'smooth_cost': best['smooth_cost'],
            'transition_cost': best['transition_cost'],
            'altitude_cost': best['altitude_cost'],
            'min_clearance': best['min_clearance'],
            'min_clearance_source': best.get('min_clearance_source', 'unknown'),
            'min_static_clearance': best.get('min_static_clearance', 999.0),
            'recommended_velocity_ned': {
                'vx': best['vx'],
                'vy': best['vy'],
                'vz': best['vz'],
            },
            'candidate_count': len(scored),
            'top_candidates': scored[: min(3, len(scored))],
        }

    def _select_escape_primitive(self, scored):
        escape_names = {
            'wait',
            'back',
            'strafe_left',
            'strafe_right',
            'climb',
            'descend',
            'back_climb',
            'back_descend',
            'left_climb',
            'right_climb',
            'left_descend',
            'right_descend',
        }
        escape_candidates = [item for item in scored if item['name'] in escape_names]
        if not escape_candidates:
            return None
        return max(
            escape_candidates,
            key=lambda item: (
                float(item.get('min_clearance', -999.0)),
                -float(item.get('risk_cost', 999.0)),
                -float(item.get('smooth_cost', 999.0)),
            ),
        )

    def _constant_velocity_primitive(self, name, state, vx, vy, vz):
        x = float(state['x'])
        y = float(state['y'])
        z = float(state['z'])
        points = []
        for raw_point in self.library._rollout(x, y, z, vx, vy, vz):
            points.append(raw_point)
        return {
            'name': name,
            'vx': float(vx),
            'vy': float(vy),
            'vz': float(vz),
            'points': points,
        }

    def _score_primitive(self, state, target, primitive, previous_primitive=None):
        final = primitive['points'][-1]
        target_cost = self._distance(
            (final['x'], final['y'], final['z']),
            (target['x'], target['y'], target['z']),
        )
        risk_cost, min_clearance, min_source, min_static_clearance = self._risk_cost(primitive, state)
        feasible = (
            min_clearance >= self.hard_clearance
            and min_static_clearance >= self.static_hard_clearance
        )
        smooth_cost = norm3(
            (
                float(primitive['vx']) - float(state.get('vx', 0.0)),
                float(primitive['vy']) - float(state.get('vy', 0.0)),
                float(primitive['vz']) - float(state.get('vz', 0.0)),
            )
        )
        trans_cost = transition_cost(
            previous_primitive,
            primitive,
            previous_velocity=(state.get('vx', 0.0), state.get('vy', 0.0), state.get('vz', 0.0)),
        )
        altitude_cost = risk_aware_altitude_cost(
            primitive,
            target['z'],
            self._obstacle_samples_for_altitude(primitive, state),
            self.risk_radius,
            self.drone_radius,
            self.safety_margin,
            self.altitude_clearance,
        )
        score = (
            self.target_weight * target_cost
            + self.risk_weight * risk_cost
            + self.smooth_weight * smooth_cost
            + self.transition_weight * trans_cost
            + self.altitude_weight * altitude_cost
        )
        if not feasible:
            score += 1000.0 + 100.0 * abs(self.hard_clearance - min_clearance)
        return {
            'name': primitive['name'],
            'feasible': bool(feasible),
            'score': float(score),
            'target_cost': float(target_cost),
            'risk_cost': float(risk_cost),
            'smooth_cost': float(smooth_cost),
            'transition_cost': float(trans_cost),
            'altitude_cost': float(altitude_cost),
            'min_clearance': float(min_clearance),
            'min_clearance_source': str(min_source),
            'min_static_clearance': float(min_static_clearance),
            'vx': float(primitive['vx']),
            'vy': float(primitive['vy']),
            'vz': float(primitive['vz']),
        }

    def _obstacle_samples_for_altitude(self, primitive, state):
        samples = []
        for track in self.predicted_tracks:
            obs_radius = float(track.get('radius', 0.3))
            predictions = list(track.get('predictions') or [])
            for point in primitive['points']:
                pred = self._nearest_prediction(predictions, float(point['t']))
                if pred is None:
                    continue
                samples.append(
                    {
                        'x': float(pred.get('x', 0.0)),
                        'y': float(pred.get('y', 0.0)),
                        'z': float(pred.get('z', 0.0)),
                        'radius': obs_radius,
                    }
                )
        if self.include_static:
            for track in self.static_tracks:
                samples.append(
                    {
                        'x': float(track.get('x', 0.0)),
                        'y': float(track.get('y', 0.0)),
                        'z': float(track.get('z', 0.0)),
                        'radius': float(track.get('radius', 0.3)),
                    }
                )
        if self.include_other_drones:
            own_id = int(state.get('id', 0))
            now = time.time()
            for other_id, other in self.states.items():
                if int(other_id) == own_id:
                    continue
                if now - float(other.get('stamp', 0.0)) > self.state_timeout:
                    continue
                for point in primitive['points']:
                    t = float(point['t'])
                    samples.append(
                        {
                            'x': float(other.get('x', 0.0)) + float(other.get('vx', 0.0)) * t,
                            'y': float(other.get('y', 0.0)) + float(other.get('vy', 0.0)) * t,
                            'z': float(other.get('z', 0.0)) + float(other.get('vz', 0.0)) * t,
                            'radius': self.drone_radius,
                        }
                    )
        return samples

    def _risk_cost(self, primitive, state):
        max_cost = 0.0
        min_clearance = 999.0
        min_source = 'none'
        min_static_clearance = 999.0

        def apply_clearance(point, obs_x, obs_y, obs_z, obs_radius, source):
            nonlocal max_cost, min_clearance, min_source, min_static_clearance
            threshold = self.drone_radius + float(obs_radius) + self.safety_margin
            dist = self._distance(
                (point['x'], point['y'], point['z']),
                (obs_x, obs_y, obs_z),
            )
            clearance = dist - threshold
            if clearance < min_clearance:
                min_clearance = clearance
                min_source = source
            if source == 'static':
                min_static_clearance = min(min_static_clearance, clearance)
            if clearance <= 0.0:
                cost = 1.0
            elif clearance < self.risk_radius:
                cost = (self.risk_radius - clearance) / max(self.risk_radius, 1e-6)
            else:
                cost = 0.0
            time_weight = 1.0 / max(float(point['t']), 0.2)
            max_cost = max(max_cost, cost * time_weight)

        for track in self.predicted_tracks:
            obs_radius = float(track.get('radius', 0.3))
            predictions = list(track.get('predictions') or [])
            for point in primitive['points']:
                pred = self._nearest_prediction(predictions, float(point['t']))
                if pred is None:
                    continue
                apply_clearance(
                    point,
                    float(pred.get('x', 0.0)),
                    float(pred.get('y', 0.0)),
                    float(pred.get('z', 0.0)),
                    obs_radius,
                    'dynamic',
                )

        if self.include_static:
            for track in self.static_tracks:
                obs_radius = float(track.get('radius', 0.3))
                obs_x = float(track.get('x', 0.0))
                obs_y = float(track.get('y', 0.0))
                obs_z = float(track.get('z', 0.0))
                for point in primitive['points']:
                    apply_clearance(point, obs_x, obs_y, obs_z, obs_radius, 'static')

        if self.include_other_drones:
            own_id = int(state.get('id', 0))
            now = time.time()
            for other_id, other in self.states.items():
                if int(other_id) == own_id:
                    continue
                if now - float(other.get('stamp', 0.0)) > self.state_timeout:
                    continue
                for point in primitive['points']:
                    t = float(point['t'])
                    apply_clearance(
                        point,
                        float(other.get('x', 0.0)) + float(other.get('vx', 0.0)) * t,
                        float(other.get('y', 0.0)) + float(other.get('vy', 0.0)) * t,
                        float(other.get('z', 0.0)) + float(other.get('vz', 0.0)) * t,
                        self.drone_radius,
                        'drone',
                    )
        return max_cost, min_clearance, min_source, min_static_clearance

    def _project_velocity(self, state, desired_velocity):
        px = float(state.get('x', 0.0))
        py = float(state.get('y', 0.0))
        pz = float(state.get('z', 0.0))
        vx, vy, vz = (
            float(desired_velocity[0]),
            float(desired_velocity[1]),
            float(desired_velocity[2]),
        )
        constraints = self._projection_constraints_for_state(state, (vx, vy, vz))
        if not constraints:
            return self.library._limit_speed(vx, vy, vz) + (0,)

        constraints.sort(key=lambda item: float(item['clearance']))
        active = constraints[: self.projection_max_constraints]

        for _ in range(self.projection_iterations):
            for item in active:
                ox = float(item['x'])
                oy = float(item['y'])
                oz = float(item['z'])
                ovx = float(item.get('vx', 0.0))
                ovy = float(item.get('vy', 0.0))
                ovz = float(item.get('vz', 0.0))
                threshold = float(item['threshold'])
                ux = float(item.get('ux', px))
                uy = float(item.get('uy', py))
                uz = float(item.get('uz', pz))
                dx = ux - ox
                dy = uy - oy
                dz = uz - oz
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < 1e-6:
                    dx = vx - ovx
                    dy = vy - ovy
                    dz = vz - ovz
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist < 1e-6:
                        dx, dy, dz, dist = 0.0, 0.0, 1.0, 1.0
                nx = dx / dist
                ny = dy / dist
                nz = dz / dist
                h = dist - threshold
                lower_bound = (nx * ovx + ny * ovy + nz * ovz) - self.projection_alpha * h
                current = nx * vx + ny * vy + nz * vz
                if current < lower_bound:
                    delta = lower_bound - current
                    vx += delta * nx
                    vy += delta * ny
                    vz += delta * nz
                    vx, vy, vz = self.library._limit_speed(vx, vy, vz)

        vx, vy, vz = self.library._limit_speed(vx, vy, vz)
        return vx, vy, vz, len(active)

    def _projection_constraints_for_state(self, state, desired_velocity):
        constraints = []
        px = float(state.get('x', 0.0))
        py = float(state.get('y', 0.0))
        pz = float(state.get('z', 0.0))
        dvx = float(desired_velocity[0])
        dvy = float(desired_velocity[1])
        dvz = float(desired_velocity[2])

        def append_constraint(x, y, z, radius, vx=0.0, vy=0.0, vz=0.0, source='obstacle', t=0.0):
            t = max(0.0, float(t))
            ux = px + dvx * t
            uy = py + dvy * t
            uz = pz + dvz * t
            threshold = self.drone_radius + float(radius) + self.safety_margin
            dist = self._distance((ux, uy, uz), (x, y, z))
            clearance = dist - threshold
            if clearance <= self.risk_radius:
                constraints.append(
                    {
                        'ux': float(ux),
                        'uy': float(uy),
                        'uz': float(uz),
                        'x': float(x),
                        'y': float(y),
                        'z': float(z),
                        'vx': float(vx),
                        'vy': float(vy),
                        'vz': float(vz),
                        'threshold': float(threshold),
                        'clearance': float(clearance),
                        'source': source,
                        't': float(t),
                    }
                )

        for track in self.predicted_tracks:
            predictions = list(track.get('predictions') or [])
            if predictions:
                for pred in predictions:
                    t = float(pred.get('t', 0.0))
                    if t < 0.0 or t > self.library.horizon:
                        continue
                    append_constraint(
                        float(pred.get('x', track.get('x', 0.0))),
                        float(pred.get('y', track.get('y', 0.0))),
                        float(pred.get('z', track.get('z', 0.0))),
                        float(track.get('radius', 0.3)),
                        float(track.get('vx', 0.0)),
                        float(track.get('vy', 0.0)),
                        float(track.get('vz', 0.0)),
                        'dynamic_future',
                        t,
                    )
            else:
                append_constraint(
                    float(track.get('x', 0.0)),
                    float(track.get('y', 0.0)),
                    float(track.get('z', 0.0)),
                    float(track.get('radius', 0.3)),
                    float(track.get('vx', 0.0)),
                    float(track.get('vy', 0.0)),
                    float(track.get('vz', 0.0)),
                    'dynamic',
                    0.0,
                )

        if self.include_static:
            for track in self.static_tracks:
                append_constraint(
                    float(track.get('x', 0.0)),
                    float(track.get('y', 0.0)),
                    float(track.get('z', 0.0)),
                    float(track.get('radius', 0.3)),
                    0.0,
                    0.0,
                    0.0,
                    'static',
                    0.0,
                )

        if self.include_other_drones:
            own_id = int(state.get('id', 0))
            now = time.time()
            time_samples = [
                max(0.0, min(self.library.horizon, self.library.dt * k))
                for k in range(1, int(math.floor(self.library.horizon / max(self.library.dt, 1e-6))) + 1)
            ]
            for other_id, other in self.states.items():
                if int(other_id) == own_id:
                    continue
                if now - float(other.get('stamp', 0.0)) > self.state_timeout:
                    continue
                for t in time_samples:
                    append_constraint(
                        float(other.get('x', 0.0)) + float(other.get('vx', 0.0)) * t,
                        float(other.get('y', 0.0)) + float(other.get('vy', 0.0)) * t,
                        float(other.get('z', 0.0)) + float(other.get('vz', 0.0)) * t,
                        self.drone_radius,
                        float(other.get('vx', 0.0)),
                        float(other.get('vy', 0.0)),
                        float(other.get('vz', 0.0)),
                        'drone_future',
                        t,
                    )

        return constraints

    def _target_for_drone(self, drone_id):
        offset = (int(drone_id) - self.leader_id) * self.target_y_spacing
        return {
            'x': self.target_x,
            'y': self.target_y_base + offset,
            'z': self.target_z,
        }

    @staticmethod
    def _nearest_prediction(predictions, t):
        if not predictions:
            return None
        return min(predictions, key=lambda item: abs(float(item.get('t', 0.0)) - t))

    @staticmethod
    def _distance(a, b):
        return math.sqrt(
            (float(a[0]) - float(b[0])) ** 2
            + (float(a[1]) - float(b[1])) ** 2
            + (float(a[2]) - float(b[2])) ** 2
        )


def main():
    parser = argparse.ArgumentParser(description='Lightweight UAV motion primitive selector')
    parser.add_argument('--num-drones', type=int, default=5)
    parser.add_argument('--state-topic', type=str, default='/xtdrone2/swarm/state_exchange')
    parser.add_argument('--predicted-topic', type=str, default='/xtdrone2/swarm/filtered_predicted_dynamic_obstacles')
    parser.add_argument('--static-topic', type=str, default='/xtdrone2/swarm/tracked_static_obstacles')
    parser.add_argument('--output-topic', type=str, default='/xtdrone2/swarm/selected_motion_primitives')
    parser.add_argument('--target-x', type=float, default=30.0)
    parser.add_argument('--target-y-base', type=float, default=26.0)
    parser.add_argument('--target-y-spacing', type=float, default=1.8)
    parser.add_argument('--target-z', type=float, default=-3.0)
    parser.add_argument('--leader-id', type=int, default=3)
    parser.add_argument('--horizon', type=float, default=2.5)
    parser.add_argument('--dt', type=float, default=0.5)
    parser.add_argument('--cruise-speed', type=float, default=0.7)
    parser.add_argument('--lateral-speed', type=float, default=0.45)
    parser.add_argument('--vertical-speed', type=float, default=0.35)
    parser.add_argument('--max-speed', type=float, default=1.0)
    parser.add_argument('--drone-radius', type=float, default=0.35)
    parser.add_argument('--safety-margin', type=float, default=0.45)
    parser.add_argument('--risk-radius', type=float, default=5.0)
    parser.add_argument('--state-timeout', type=float, default=2.0)
    parser.add_argument('--target-weight', type=float, default=1.0)
    parser.add_argument('--risk-weight', type=float, default=8.0)
    parser.add_argument('--smooth-weight', type=float, default=0.3)
    parser.add_argument('--transition-weight', type=float, default=1.2)
    parser.add_argument('--altitude-weight', type=float, default=1.4)
    parser.add_argument('--altitude-clearance', type=float, default=0.7)
    parser.add_argument('--publish-shadow-cmd', type=str, default='true')
    parser.add_argument('--include-static', type=str, default='true')
    parser.add_argument('--include-other-drones', type=str, default='true')
    parser.add_argument('--emergency-clearance', type=float, default=0.8)
    parser.add_argument('--projection-alpha', type=float, default=1.2)
    parser.add_argument('--projection-iterations', type=int, default=2)
    parser.add_argument('--projection-max-constraints', type=int, default=16)
    parser.add_argument('--escape-clearance', type=float, default=0.4)
    parser.add_argument('--hard-clearance', type=float, default=0.2)
    parser.add_argument('--static-hard-clearance', type=float, default=0.6)
    parser.add_argument('--static-emergency-clearance', type=float, default=1.2)
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = MotionPrimitiveSelector(
        num_drones=args.num_drones,
        state_topic=args.state_topic,
        predicted_topic=args.predicted_topic,
        static_topic=args.static_topic,
        output_topic=args.output_topic,
        target_x=args.target_x,
        target_y_base=args.target_y_base,
        target_y_spacing=args.target_y_spacing,
        target_z=args.target_z,
        leader_id=args.leader_id,
        horizon=args.horizon,
        dt=args.dt,
        cruise_speed=args.cruise_speed,
        lateral_speed=args.lateral_speed,
        vertical_speed=args.vertical_speed,
        max_speed=args.max_speed,
        drone_radius=args.drone_radius,
        safety_margin=args.safety_margin,
        risk_radius=args.risk_radius,
        state_timeout=args.state_timeout,
        target_weight=args.target_weight,
        risk_weight=args.risk_weight,
        smooth_weight=args.smooth_weight,
        transition_weight=args.transition_weight,
        altitude_weight=args.altitude_weight,
        altitude_clearance=args.altitude_clearance,
        publish_shadow_cmd=str(args.publish_shadow_cmd).strip().lower() in ('1', 'true', 'yes', 'on'),
        include_static=str(args.include_static).strip().lower() in ('1', 'true', 'yes', 'on'),
        include_other_drones=str(args.include_other_drones).strip().lower() in ('1', 'true', 'yes', 'on'),
        emergency_clearance=args.emergency_clearance,
        projection_alpha=args.projection_alpha,
        projection_iterations=args.projection_iterations,
        projection_max_constraints=args.projection_max_constraints,
        escape_clearance=args.escape_clearance,
        hard_clearance=args.hard_clearance,
        static_hard_clearance=args.static_hard_clearance,
        static_emergency_clearance=args.static_emergency_clearance,
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
