#!/usr/bin/env python3
"""Trajectory metrics for moving-obstacle MPC candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .common import PerceptionData, PlannerConfig, TrajectoryPoint, Vec3
from .dynamic_safety import is_dynamic_obstacle, obstacle_velocity


@dataclass
class DynamicMPCMetrics:
    total_cost: float = 0.0
    risk_flag: bool = False
    corridor_flag: bool = False
    min_ttc: float = 999.0
    max_closing_speed: float = 0.0
    min_clearance: float = 999.0
    min_lateral_separation: float = 999.0
    required_lateral_separation: float = 0.0
    lateral_progress: float = 0.0
    samples: int = 0
    worst_obstacle_id: str = ""

    def to_dict(self) -> dict:
        return {
            "dynamic_total_cost": float(self.total_cost),
            "dynamic_risk_flag": bool(self.risk_flag),
            "dynamic_corridor_flag": bool(self.corridor_flag),
            "dynamic_min_ttc": float(self.min_ttc),
            "dynamic_closing_speed": float(self.max_closing_speed),
            "dynamic_min_clearance": float(self.min_clearance),
            "dynamic_min_lateral_separation": float(self.min_lateral_separation),
            "dynamic_required_lateral_separation": float(self.required_lateral_separation),
            "dynamic_lateral_progress": float(self.lateral_progress),
            "dynamic_metric_samples": int(self.samples),
            "dynamic_worst_obstacle_id": self.worst_obstacle_id,
        }


def _perp(v: Vec3) -> Vec3:
    return Vec3(-v.y, v.x, 0.0)


def _radius(config: PlannerConfig, obs) -> float:
    return float(getattr(config, "drone_radius", 0.35)) + float(getattr(obs, "radius", 0.0))


def _required_lateral_sep(config: PlannerConfig, obs) -> float:
    return max(
        _radius(config, obs) + float(getattr(config, "dynamic_lateral_margin", 0.35)),
        float(getattr(config, "dynamic_lateral_separation", 0.85)),
    )


def evaluate_dynamic_mpc_metrics(
    points: Sequence[TrajectoryPoint],
    controls: Sequence[Vec3],
    perception: Optional[PerceptionData],
    config: PlannerConfig,
) -> DynamicMPCMetrics:
    metrics = DynamicMPCMetrics()
    if perception is None or not perception.obstacles or not points:
        return metrics

    ttc_soft = float(getattr(config, "dynamic_ttc_soft", 2.4))
    ttc_risk = float(getattr(config, "dynamic_ttc_risk", 0.85))
    corridor_ttc = float(getattr(config, "dynamic_corridor_ttc", 3.2))
    clearance_risk = float(getattr(config, "dynamic_clearance_risk", 0.16))
    clearance_soft = float(getattr(config, "dynamic_clearance_soft", 0.85))

    total_cost = 0.0
    lateral_first = None
    lateral_last = None

    for idx, point in enumerate(points[1:], start=1):
        ego_vel = controls[min(idx - 1, len(controls) - 1)] if controls else Vec3()
        for obs in perception.obstacles:
            if not is_dynamic_obstacle(obs):
                continue
            obs_pos = obs.position_at(point.t)
            obs_vel = obstacle_velocity(obs)
            rel = Vec3(obs_pos.x - point.position.x, obs_pos.y - point.position.y, 0.0)
            dist = rel.norm()
            if dist <= 1.0e-6:
                continue

            radius = _radius(config, obs)
            clearance = dist - radius
            rel_vel = obs_vel - ego_vel
            closing = -((rel.x * rel_vel.x + rel.y * rel_vel.y) / dist)
            ttc = max(0.0, clearance) / max(closing, 1.0e-6) if closing > 0.0 else 999.0

            axis = obs_vel.normalized(rel.normalized(Vec3(1.0, 0.0, 0.0)))
            lateral_axis = _perp(axis)
            lateral_sep = abs(rel.x * lateral_axis.x + rel.y * lateral_axis.y)
            required_sep = _required_lateral_sep(config, obs)

            metrics.samples += 1
            metrics.min_ttc = min(metrics.min_ttc, ttc)
            metrics.max_closing_speed = max(metrics.max_closing_speed, closing)
            if clearance < metrics.min_clearance:
                metrics.min_clearance = clearance
                metrics.worst_obstacle_id = str(getattr(obs, "obstacle_id", ""))
            metrics.min_lateral_separation = min(metrics.min_lateral_separation, lateral_sep)
            metrics.required_lateral_separation = max(metrics.required_lateral_separation, required_sep)
            if lateral_first is None:
                lateral_first = lateral_sep
            lateral_last = lateral_sep

            if closing > 0.0 and ttc < ttc_soft:
                gap = ttc_soft - ttc
                total_cost += gap * gap * (1.0 + 1.2 * closing)
            if clearance < clearance_soft:
                gap = clearance_soft - clearance
                total_cost += 1.2 * gap * gap

            corridor_active = closing > 0.03 and ttc < corridor_ttc
            if corridor_active and lateral_sep < required_sep:
                gap = required_sep - lateral_sep
                total_cost += 18.0 * gap * gap + 6.0 * gap
                metrics.corridor_flag = True

            if clearance < clearance_risk or (closing > 0.0 and ttc < ttc_risk) or (
                corridor_active and lateral_sep < required_sep * 0.55
            ):
                metrics.risk_flag = True
                total_cost += 160.0

    metrics.total_cost = total_cost / max(1, metrics.samples)
    if lateral_first is not None and lateral_last is not None:
        metrics.lateral_progress = float(lateral_last - lateral_first)
        if metrics.corridor_flag and metrics.lateral_progress < 0.05:
            metrics.total_cost += 35.0 + 25.0 * max(0.0, 0.05 - metrics.lateral_progress)
    return metrics
