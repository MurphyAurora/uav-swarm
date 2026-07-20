#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cost helpers for lightweight 3D motion primitive selection.

The selector owns ROS state and obstacle sampling. This module keeps the
primitive scoring terms small, deterministic, and easy to tune.
"""

import math


def _norm3(x, y, z):
    return math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))


def transition_cost(previous_primitive, primitive, previous_velocity=None):
    """Penalize abrupt primitive switching.

    The existing smooth term compares the chosen velocity to current vehicle
    velocity. This term compares the new primitive with the previously selected
    primitive, which damps left-right / climb-descend flip-flopping.
    """
    if not previous_primitive:
        return 0.0

    pvx = float(previous_primitive.get('vx', 0.0))
    pvy = float(previous_primitive.get('vy', 0.0))
    pvz = float(previous_primitive.get('vz', 0.0))
    vx = float(primitive.get('vx', 0.0))
    vy = float(primitive.get('vy', 0.0))
    vz = float(primitive.get('vz', 0.0))

    velocity_delta = _norm3(vx - pvx, vy - pvy, vz - pvz)
    prev_speed = _norm3(pvx, pvy, pvz)
    new_speed = _norm3(vx, vy, vz)

    heading_flip = 0.0
    if prev_speed > 1e-6 and new_speed > 1e-6:
        dot = (pvx * vx + pvy * vy + pvz * vz) / max(prev_speed * new_speed, 1e-6)
        heading_flip = max(0.0, -max(-1.0, min(1.0, dot)))

    vertical_flip = 0.0
    if abs(pvz) > 1e-4 and abs(vz) > 1e-4 and pvz * vz < 0.0:
        vertical_flip = 1.0

    name_change = 0.0
    if str(previous_primitive.get('name', '')) != str(primitive.get('name', '')):
        name_change = 0.15

    if previous_velocity is not None:
        cvx, cvy, cvz = previous_velocity
        command_jump = _norm3(vx - float(cvx), vy - float(cvy), vz - float(cvz))
    else:
        command_jump = 0.0

    return velocity_delta + 0.8 * heading_flip + 0.6 * vertical_flip + name_change + 0.25 * command_jump


def risk_aware_altitude_cost(
    primitive,
    target_z,
    obstacle_samples,
    risk_radius,
    drone_radius,
    safety_margin,
    altitude_clearance=0.7,
):
    """Prefer cruise altitude normally, but climb over nearby high obstacles.

    Coordinates follow the project convention: z is NED-like, so more negative
    z means higher altitude. When a rollout point is horizontally near an
    obstacle whose top intrudes near cruise altitude, the desired z is moved
    upward just enough to clear it. Once the rollout is beyond risk, the desired
    altitude naturally returns to target_z.
    """
    target_z = float(target_z)
    risk_radius = max(float(risk_radius), 1e-6)
    drone_radius = max(float(drone_radius), 0.0)
    safety_margin = max(float(safety_margin), 0.0)
    altitude_clearance = max(float(altitude_clearance), 0.0)
    samples = list(obstacle_samples or [])
    points = list(primitive.get('points') or [])
    if not points:
        return 0.0

    total = 0.0
    total_weight = 0.0
    cruise_weight = 0.2

    for point in points:
        px = float(point.get('x', 0.0))
        py = float(point.get('y', 0.0))
        pz = float(point.get('z', target_z))
        desired_z = target_z
        risk_weight = 0.0

        for obs in samples:
            ox = float(obs.get('x', 0.0))
            oy = float(obs.get('y', 0.0))
            oz = float(obs.get('z', 0.0))
            radius = max(0.0, float(obs.get('radius', 0.0)))
            horizontal_clearance = math.hypot(px - ox, py - oy) - (radius + drone_radius + safety_margin)
            if horizontal_clearance >= risk_radius:
                continue
            closeness = 1.0 - max(0.0, horizontal_clearance) / risk_radius
            obstacle_top_z = oz - radius
            clear_z = obstacle_top_z - drone_radius - safety_margin - altitude_clearance
            if clear_z < desired_z:
                desired_z = clear_z
            risk_weight = max(risk_weight, closeness)

        weight = cruise_weight + risk_weight
        total += weight * abs(pz - desired_z)
        total_weight += weight

    if total_weight <= 1e-9:
        return 0.0
    return total / total_weight
