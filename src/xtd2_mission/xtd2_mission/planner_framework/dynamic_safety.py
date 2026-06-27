#!/usr/bin/env python3
"""Dynamic-obstacle TTC and escape helpers.

These helpers are intentionally lightweight and deterministic so they can be
used both by the offline simulator and by the ROS planner nodes.  They do not
replace the hard SafetyChecker; they add a dynamic layer that distinguishes
"hover is safe" in static scenes from "hover is still in the path of an
approaching obstacle" in dynamic scenes.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from .common import PerceptionData, PlannerConfig, PlannerState, TrajectoryPoint, Vec3


_MIN_DYNAMIC_SPEED = 0.03


def obstacle_velocity(obs, sample_dt: float = 0.35) -> Vec3:
    """Return a robust velocity estimate for an obstacle.

    Prefer the explicit obstacle velocity, but fall back to the first prediction
    difference.  This keeps dynamic safety usable when the simulator provides
    predictions but the velocity field is zero or missing.
    """
    vel = getattr(obs, "velocity", Vec3()) or Vec3()
    if vel.norm() > _MIN_DYNAMIC_SPEED:
        return vel
    try:
        p0 = obs.position_at(0.0)
        p1 = obs.position_at(sample_dt)
        return (p1 - p0) * (1.0 / max(sample_dt, 1.0e-6))
    except Exception:
        return Vec3()


def is_dynamic_obstacle(obs) -> bool:
    if getattr(obs, "source", "") == "boundary":
        return False
    return obstacle_velocity(obs).norm() > _MIN_DYNAMIC_SPEED


def dynamic_ttc_snapshot(
    state: PlannerState,
    perception: Optional[PerceptionData],
    config: PlannerConfig,
) -> Tuple[float, float, float, Optional[object]]:
    """Return min TTC, max closing speed, min dynamic clearance, and obstacle."""
    if perception is None:
        return 999.0, 0.0, 999.0, None
    min_ttc = 999.0
    max_closing = 0.0
    min_clearance = 999.0
    worst = None
    ego_vel = Vec3(state.velocity.x, state.velocity.y, 0.0)
    radius_base = float(getattr(config, "drone_radius", 0.35))
    for obs in perception.obstacles:
        if not is_dynamic_obstacle(obs):
            continue
        obs_pos = obs.position_at(0.0)
        rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)
        dist = rel.norm()
        if dist <= 1.0e-6:
            continue
        clearance = dist - (radius_base + float(getattr(obs, "radius", 0.0)))
        rel_vel = obstacle_velocity(obs) - ego_vel
        closing = -((rel.x * rel_vel.x + rel.y * rel_vel.y) / dist)
        if closing <= 0.0:
            ttc = 999.0
        else:
            ttc = max(0.0, clearance) / closing
        if ttc < min_ttc or clearance < min_clearance:
            worst = obs
        min_ttc = min(min_ttc, ttc)
        max_closing = max(max_closing, closing)
        min_clearance = min(min_clearance, clearance)
    return float(min_ttc), float(max_closing), float(min_clearance), worst


def dynamic_escape_command(
    state: PlannerState,
    perception: Optional[PerceptionData],
    config: PlannerConfig,
) -> Optional[Vec3]:
    """Compute an emergency escape velocity for approaching dynamic obstacles.

    In dynamic head-on or crossing cases, pure hover can be unsafe because the
    obstacle keeps moving.  This function chooses a lateral/back-lateral command
    that increases short-horizon clearance against the nearest approaching
    dynamic obstacle.
    """
    if perception is None:
        return None
    ttc_trigger = float(getattr(config, "dynamic_escape_ttc", 1.8))
    clearance_trigger = float(getattr(config, "dynamic_escape_clearance", 1.15))
    best_obs = None
    best_key = (999.0, 999.0)
    ego_vel = Vec3(state.velocity.x, state.velocity.y, 0.0)
    radius_base = float(getattr(config, "drone_radius", 0.35))
    for obs in perception.obstacles:
        if not is_dynamic_obstacle(obs):
            continue
        obs_pos = obs.position_at(0.0)
        rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)
        dist = rel.norm()
        if dist <= 1.0e-6:
            continue
        clearance = dist - (radius_base + float(getattr(obs, "radius", 0.0)))
        rel_vel = obstacle_velocity(obs) - ego_vel
        closing = -((rel.x * rel_vel.x + rel.y * rel_vel.y) / dist)
        if closing <= 0.02 and clearance > clearance_trigger:
            continue
        ttc = max(0.0, clearance) / max(closing, 1.0e-6) if closing > 0.0 else 999.0
        if ttc > ttc_trigger and clearance > clearance_trigger:
            continue
        key = (ttc, clearance)
        if key < best_key:
            best_key = key
            best_obs = obs
    if best_obs is None:
        return None

    obs_pos = best_obs.position_at(0.0)
    rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)
    rel_vel = obstacle_velocity(best_obs) - ego_vel
    # Escape mostly perpendicular to the line of approach.  Pick the side that
    # moves away from the obstacle's current lateral side in the body frame.
    heading = float(getattr(state, "heading", 0.0))
    forward = Vec3(math.cos(heading), math.sin(heading), 0.0).normalized(Vec3(1.0, 0.0, 0.0))
    left = Vec3(-forward.y, forward.x, 0.0)
    rel_lateral = rel.x * left.x + rel.y * left.y
    side = -1.0 if rel_lateral > 0.0 else 1.0
    if abs(rel_lateral) < 0.10:
        # Head-on: use the side that is most perpendicular to the relative motion.
        cross = forward.x * rel_vel.y - forward.y * rel_vel.x
        side = -1.0 if cross > 0.0 else 1.0
    lateral = left * side
    back = forward * -0.35
    speed = min(float(getattr(config, "max_speed", 0.75)), max(0.22, float(getattr(config, "dynamic_escape_speed", 0.32))))
    return (lateral + back).normalized(lateral) * speed


def dynamic_trajectory_risk(
    points: Sequence[TrajectoryPoint],
    controls: Sequence[Vec3],
    perception: Optional[PerceptionData],
    config: PlannerConfig,
) -> Tuple[float, float, float, float]:
    """Soft dynamic risk for MPC candidate scoring.

    Returns: cost, min_ttc, max_closing_speed, min_dynamic_clearance.
    This is a soft cost, not a hard filter.  SafetyChecker still owns hard
    feasibility; this only makes MPC prefer trajectories that increase TTC and
    leave the collision cone earlier.
    """
    if perception is None or not perception.obstacles or not points:
        return 0.0, 999.0, 0.0, 999.0
    radius_base = float(getattr(config, "drone_radius", 0.35))
    ttc_soft = float(getattr(config, "dynamic_ttc_soft", 2.2))
    clearance_soft = float(getattr(config, "dynamic_clearance_soft", 0.80))
    cost = 0.0
    samples = 0
    min_ttc = 999.0
    max_closing = 0.0
    min_clearance = 999.0
    for idx, point in enumerate(points[1:], start=1):
        ego_vel = controls[min(idx - 1, len(controls) - 1)] if controls else Vec3()
        for obs in perception.obstacles:
            if not is_dynamic_obstacle(obs):
                continue
            obs_pos = obs.position_at(point.t)
            rel = Vec3(obs_pos.x - point.position.x, obs_pos.y - point.position.y, 0.0)
            dist = rel.norm()
            if dist <= 1.0e-6:
                continue
            clearance = dist - (radius_base + float(getattr(obs, "radius", 0.0)))
            obs_vel = obstacle_velocity(obs)
            rel_vel = obs_vel - ego_vel
            closing = -((rel.x * rel_vel.x + rel.y * rel_vel.y) / dist)
            ttc = max(0.0, clearance) / max(closing, 1.0e-6) if closing > 0.0 else 999.0
            min_ttc = min(min_ttc, ttc)
            max_closing = max(max_closing, closing)
            min_clearance = min(min_clearance, clearance)
            if closing > 0.0 and ttc < ttc_soft:
                gap = ttc_soft - ttc
                cost += gap * gap * (1.0 + 0.8 * closing)
            if clearance < clearance_soft:
                gap = clearance_soft - clearance
                cost += 0.75 * gap * gap
            samples += 1
    return cost / max(1, samples), float(min_ttc), float(max_closing), float(min_clearance)
