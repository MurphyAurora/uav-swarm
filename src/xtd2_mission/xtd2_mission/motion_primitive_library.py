#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight 3D motion primitive generation for UAV local planning."""

import math


def clamp(value, limit):
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def norm3(vec):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])


def unit_xy(dx, dy, fallback=(1.0, 0.0)):
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return fallback
    return dx / length, dy / length


class MotionPrimitiveLibrary:
    """Generate short constant-velocity UAV trajectory primitives.

    Coordinates are expected to use the same world/NED-like frame as
    /xtdrone2/swarm/state_exchange: x/y horizontal, z negative upward.
    """

    def __init__(
        self,
        horizon=2.5,
        dt=0.5,
        cruise_speed=0.7,
        lateral_speed=0.45,
        vertical_speed=0.35,
        max_speed=1.0,
    ):
        self.horizon = float(horizon)
        self.dt = float(dt)
        self.cruise_speed = float(cruise_speed)
        self.lateral_speed = float(lateral_speed)
        self.vertical_speed = float(vertical_speed)
        self.max_speed = float(max_speed)

    def generate(self, state, target):
        x = float(state['x'])
        y = float(state['y'])
        z = float(state['z'])
        tx = float(target['x'])
        ty = float(target['y'])
        tz = float(target['z'])

        fx, fy = unit_xy(tx - x, ty - y)
        lx, ly = -fy, fx
        z_error = clamp(tz - z, self.vertical_speed)

        specs = [
            ('straight', self.cruise_speed * fx, self.cruise_speed * fy, z_error),
            ('slow_forward', 0.45 * self.cruise_speed * fx, 0.45 * self.cruise_speed * fy, z_error),
            ('wait', 0.0, 0.0, 0.0),
            ('back', -0.45 * self.cruise_speed * fx, -0.45 * self.cruise_speed * fy, 0.0),
            ('strafe_left', self.lateral_speed * lx, self.lateral_speed * ly, 0.0),
            ('strafe_right', -self.lateral_speed * lx, -self.lateral_speed * ly, 0.0),
            ('left', self.cruise_speed * fx + self.lateral_speed * lx, self.cruise_speed * fy + self.lateral_speed * ly, z_error),
            ('right', self.cruise_speed * fx - self.lateral_speed * lx, self.cruise_speed * fy - self.lateral_speed * ly, z_error),
            ('climb', 0.65 * self.cruise_speed * fx, 0.65 * self.cruise_speed * fy, -self.vertical_speed),
            ('descend', 0.65 * self.cruise_speed * fx, 0.65 * self.cruise_speed * fy, self.vertical_speed),
            ('back_climb', -0.35 * self.cruise_speed * fx, -0.35 * self.cruise_speed * fy, -self.vertical_speed),
            ('back_descend', -0.35 * self.cruise_speed * fx, -0.35 * self.cruise_speed * fy, self.vertical_speed),
            ('left_climb', self.cruise_speed * fx + self.lateral_speed * lx, self.cruise_speed * fy + self.lateral_speed * ly, -self.vertical_speed),
            ('right_climb', self.cruise_speed * fx - self.lateral_speed * lx, self.cruise_speed * fy - self.lateral_speed * ly, -self.vertical_speed),
            ('left_descend', self.cruise_speed * fx + self.lateral_speed * lx, self.cruise_speed * fy + self.lateral_speed * ly, self.vertical_speed),
            ('right_descend', self.cruise_speed * fx - self.lateral_speed * lx, self.cruise_speed * fy - self.lateral_speed * ly, self.vertical_speed),
        ]

        primitives = []
        for name, vx, vy, vz in specs:
            vx, vy, vz = self._limit_speed(vx, vy, vz)
            primitives.append(
                {
                    'name': name,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'points': self._rollout(x, y, z, vx, vy, vz),
                }
            )
        return primitives

    def _rollout(self, x, y, z, vx, vy, vz):
        step = max(0.05, self.dt)
        horizon = max(step, self.horizon)
        n_steps = int(math.floor(horizon / step + 1e-6))
        points = []
        for k in range(1, n_steps + 1):
            t = k * step
            points.append(
                {
                    't': t,
                    'x': x + vx * t,
                    'y': y + vy * t,
                    'z': z + vz * t,
                }
            )
        return points

    def _limit_speed(self, vx, vy, vz):
        speed = norm3((vx, vy, vz))
        if self.max_speed > 0.0 and speed > self.max_speed:
            scale = self.max_speed / max(speed, 1e-6)
            return vx * scale, vy * scale, vz * scale
        return vx, vy, vz
