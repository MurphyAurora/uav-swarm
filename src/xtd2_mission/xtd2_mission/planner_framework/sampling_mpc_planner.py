#!/usr/bin/env python3
"""Sampling-MPC frontend candidate generator.

This module intentionally only proposes candidate rollouts. Collision, hard
clearance, and final command arbitration remain owned by SafetyChecker,
BackendSelector, and PlannerCore's FSM.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from .common import CandidateTrajectory, PerceptionData, PlannerConfig, PlannerState, TrajectoryPoint, Vec3
from .reference_generator import LocalAStarReferenceGenerator, ReferenceGenerator


class SamplingMPCPlanner:
    """Argmin sampling MPC frontend with reference tracking and soft risk costs."""

    def __init__(self, config: Optional[PlannerConfig] = None, reference_generator: Optional[ReferenceGenerator] = None):
        self.config = config or PlannerConfig()
        self.reference_generator = reference_generator or LocalAStarReferenceGenerator(self.config)
        self.diagnostics = {}
        self._committed_side = 0.0
        self._committed_until = 0.0
        self._last_min_soft_clearance = 999.0

    def generate(self, state: PlannerState, local_goal: Vec3, perception: Optional[PerceptionData] = None) -> List[CandidateTrajectory]:
        num_samples = max(1, int(getattr(self.config, "sampling_mpc_num_samples", 200)))
        horizon_steps = max(1, int(getattr(self.config, "sampling_mpc_horizon_steps", 12)))
        dt = max(1.0e-3, float(getattr(self.config, "sampling_mpc_dt", 0.2)))
        max_speed = max(1.0e-6, float(getattr(self.config, "max_speed", 0.75)))
        max_accel = max(0.0, float(getattr(self.config, "sampling_mpc_max_accel", 0.8)))

        reference = self.reference_generator.generate(state, local_goal, perception)
        to_goal = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0)
        distance = to_goal.norm()
        fallback_dir = to_goal.normalized(Vec3(1.0, 0.0, 0.0))
        ref_dir = reference.direction.normalized(fallback_dir)
        self._update_side_commitment(state, ref_dir, reference.side_hint, perception)
        left = Vec3(-ref_dir.y, ref_dir.x, 0.0)
        current = Vec3(state.velocity.x, state.velocity.y, 0.0).limit_norm(max_speed)
        cruise = min(float(getattr(self.config, "cruise_speed", max_speed)), max_speed)
        reference_speed = max(0.0, min(max_speed, float(reference.speed)))
        distance_speed = min(cruise, max(0.08, distance / max(horizon_steps * dt, 1.0e-6)))
        if distance > 1.5:
            nominal_floor = min(cruise, 0.38)
        elif distance > 0.8:
            nominal_floor = min(cruise, 0.24)
        else:
            nominal_floor = 0.08
        nominal_speed = min(max_speed, max(reference_speed, distance_speed, nominal_floor))
        nominal = ref_dir * nominal_speed

        rng = random.Random(self._seed_from_state(state, local_goal))
        specs = self._sample_specs(num_samples, rng, reference.side_hint, self._committed_side)
        ref_progress_now = self._path_progress(reference.points, state.position, local_goal)
        candidates: List[CandidateTrajectory] = []
        best_cost = 999999.0
        best_name = ""
        best_breakdown = {}
        stop_count = 0
        moving_count = 0

        for idx, spec in enumerate(specs[:num_samples]):
            controls = self._controls_from_spec(spec, current, nominal, ref_dir, left, horizon_steps, dt, max_speed, max_accel)
            candidate = self._rollout(f"sampling_mpc:{idx:03d}:{spec['kind']}", state.position, controls, dt)
            cost, breakdown = self._sequence_cost(
                candidate.points,
                controls,
                state.position,
                local_goal,
                reference.points,
                nominal,
                dt,
                perception,
                ref_progress_now,
                left,
            )
            kind = str(spec["kind"])
            is_stop = kind == "stop"
            if is_stop:
                stop_count += 1
            elif candidate.velocity.norm() > 0.05:
                moving_count += 1
            candidate.metadata.update(
                {
                    "planner": "sampling_mpc",
                    "controls_count": len(controls),
                    "sample_kind": kind,
                    "sample_cost": float(cost),
                    "sample_breakdown": dict(breakdown),
                    "horizon_steps": int(horizon_steps),
                    "dt": float(dt),
                    "reference_source": reference.source,
                    "reference_confidence": float(reference.confidence),
                    "reference_reason": reference.reason,
                    "reference_progress": float(breakdown.get("reference_progress", 0.0)),
                    "soft_clearance": float(breakdown.get("min_soft_clearance", 999.0)),
                    "committed_side": float(self._committed_side),
                    "is_stop": bool(is_stop),
                    "emergency_only": bool(is_stop),
                    "reference_tracking_target": breakdown.get("tracking_target", {}),
                }
            )
            if cost < best_cost:
                best_cost = cost
                best_name = candidate.name
                best_breakdown = breakdown
            candidates.append(candidate)

        candidates.sort(key=lambda item: float(item.metadata.get("sample_cost", 999999.0)))
        self.diagnostics = {
            "mode": "sampling_mpc",
            "samples": len(candidates),
            "horizon_steps": int(horizon_steps),
            "dt": float(dt),
            "max_speed": float(max_speed),
            "max_accel": float(max_accel),
            "best_sample": best_name,
            "best_sample_cost": float(best_cost),
            "best_cost_breakdown": dict(best_breakdown),
            "goal_distance": float(distance),
            "perception_obstacles": len(perception.obstacles) if perception is not None else 0,
            "committed_side": float(self._committed_side),
            "committed_until": float(self._committed_until),
            "moving_samples": int(moving_count),
            "stop_samples": int(stop_count),
            "reference": {
                "source": reference.source,
                "confidence": float(reference.confidence),
                "reason": reference.reason,
                "direction": reference.direction.to_dict(),
                "speed": float(reference.speed),
                "side_hint": float(reference.side_hint),
                "points": len(reference.points),
            },
        }
        return candidates

    def _sample_specs(self, num_samples: int, rng: random.Random, side_hint: float = 0.0, committed_side: float = 0.0) -> List[dict]:
        side_hint = self._sign(side_hint)
        committed_side = self._sign(committed_side)
        specs: List[dict] = [
            {"kind": "nominal", "forward": 0.0, "lateral": 0.0, "speed_scale": 1.0, "turn_rate": 0.0},
            {"kind": "slow", "forward": -0.06, "lateral": 0.0, "speed_scale": 0.72, "turn_rate": 0.0},
            {"kind": "brake", "forward": -0.18, "lateral": 0.0, "speed_scale": 0.42, "turn_rate": 0.0},
            {"kind": "stop", "forward": -1.0, "lateral": 0.0, "speed_scale": 0.0, "turn_rate": 0.0},
        ]
        for source, side in (("committed", committed_side), ("reference", side_hint)):
            if abs(side) > 0.5:
                specs.extend(
                    [
                        {"kind": f"{source}_side", "forward": 0.02, "lateral": side * 0.46, "speed_scale": 0.95, "turn_rate": side * 0.10},
                        {"kind": f"{source}_side_slow", "forward": -0.06, "lateral": side * 0.64, "speed_scale": 0.68, "turn_rate": side * 0.12},
                        {"kind": f"{source}_wide", "forward": 0.00, "lateral": side * 0.92, "speed_scale": 0.82, "turn_rate": side * 0.16},
                    ]
                )
        for side, label in ((1.0, "left"), (-1.0, "right")):
            for gain in (0.28, 0.48, 0.70, 0.95):
                specs.append({"kind": f"bias_{label}", "forward": 0.00, "lateral": side * gain, "speed_scale": 0.86, "turn_rate": side * gain * 0.18})
            specs.append({"kind": f"slow_{label}", "forward": -0.08, "lateral": side * 0.55, "speed_scale": 0.55, "turn_rate": side * 0.10})

        while len(specs) < num_samples:
            lateral_mean = committed_side * 0.34 if abs(committed_side) > 0.5 else side_hint * 0.18
            lateral = rng.gauss(lateral_mean, 0.50)
            forward = rng.gauss(0.04, 0.18)
            speed_scale = max(0.18, min(1.15, rng.gauss(0.90, 0.18)))
            turn_rate = rng.gauss(committed_side * 0.04, 0.16)
            specs.append({"kind": "sample", "forward": forward, "lateral": lateral, "speed_scale": speed_scale, "turn_rate": turn_rate})
        return specs

    def _controls_from_spec(
        self,
        spec: dict,
        current: Vec3,
        nominal: Vec3,
        ref_dir: Vec3,
        left: Vec3,
        horizon_steps: int,
        dt: float,
        max_speed: float,
        max_accel: float,
    ) -> List[Vec3]:
        controls: List[Vec3] = []
        prev = current
        accel_step = max_accel * dt
        for k in range(horizon_steps):
            alpha = k / max(horizon_steps - 1, 1)
            wobble = math.sin((alpha + 0.15) * math.pi)
            forward_noise = float(spec["forward"]) * (1.0 - 0.25 * alpha)
            lateral_noise = float(spec["lateral"]) * wobble
            turn = float(spec["turn_rate"]) * (alpha - 0.35)
            direction = (ref_dir * (1.0 + forward_noise + turn) + left * lateral_noise).normalized(ref_dir)
            speed_scale = float(spec["speed_scale"])
            nominal_speed = nominal.norm()
            target_speed = min(max_speed, max(0.0, nominal_speed * speed_scale))
            if spec["kind"] == "stop":
                target_speed = 0.0
            target = (direction * target_speed).limit_norm(max_speed)
            control = self._accel_limited(prev, target, accel_step).limit_norm(max_speed)
            controls.append(control)
            prev = control
        return controls

    def _rollout(self, name: str, start: Vec3, controls: Sequence[Vec3], dt: float) -> CandidateTrajectory:
        points = [TrajectoryPoint(t=0.0, position=start)]
        pos = start
        for idx, control in enumerate(controls, start=1):
            pos = pos + control * dt
            points.append(TrajectoryPoint(t=idx * dt, position=pos))
        first = controls[0] if controls else Vec3()
        return CandidateTrajectory(name=name, velocity=first, points=points)

    def _sequence_cost(
        self,
        points: Sequence[TrajectoryPoint],
        controls: Sequence[Vec3],
        start: Vec3,
        local_goal: Vec3,
        reference_points: Sequence[Vec3],
        nominal: Vec3,
        dt: float,
        perception: Optional[PerceptionData],
        ref_progress_now: float,
        left: Vec3,
    ) -> Tuple[float, dict]:
        final = points[-1].position if points else start
        tracking_target = self._tracking_target(reference_points, local_goal, ref_progress_now, nominal.norm(), dt, len(controls))
        goal_cost = final.distance_to(local_goal)
        reference_cost = final.distance_to(tracking_target)
        goal_progress = max(0.0, start.distance_to(local_goal) - final.distance_to(local_goal))
        reference_progress = max(0.0, self._path_progress(reference_points, final, local_goal) - ref_progress_now)
        obstacle_cost, min_soft_clearance = self._soft_obstacle_cost(points, perception)
        side_cost = self._side_commitment_cost(points, start, left)
        smooth = 0.0
        effort = 0.0
        prev = Vec3()
        for idx, control in enumerate(controls):
            effort += control.norm() * dt
            if idx > 0:
                smooth += (control - prev).norm()
            prev = control
        first_error = (controls[0] - nominal).norm() if controls else 0.0
        reverse_or_stall = 0.0 if reference_progress > 0.28 else (0.28 - reference_progress) * 8.0
        if controls and controls[0].norm() < 0.04:
            reverse_or_stall += 3.0
        cost = (
            goal_cost * 1.6
            + reference_cost * 1.8
            + obstacle_cost * 1.4
            + side_cost * 0.45
            + smooth * 0.35
            + effort * 0.06
            + first_error * 0.20
            + reverse_or_stall
            - goal_progress * 1.8
            - reference_progress * 7.0
        )
        return cost, {
            "goal_cost": float(goal_cost),
            "reference_cost": float(reference_cost),
            "goal_progress": float(goal_progress),
            "reference_progress": float(reference_progress),
            "obstacle_cost": float(obstacle_cost),
            "side_cost": float(side_cost),
            "smooth": float(smooth),
            "effort": float(effort),
            "first_error": float(first_error),
            "reverse_or_stall": float(reverse_or_stall),
            "min_soft_clearance": float(min_soft_clearance),
            "tracking_target": tracking_target.to_dict(),
        }

    def _soft_obstacle_cost(self, points: Sequence[TrajectoryPoint], perception: Optional[PerceptionData]) -> Tuple[float, float]:
        if perception is None or not perception.obstacles:
            return 0.0, 999.0
        soft_clearance = float(getattr(self.config, "sampling_mpc_soft_clearance", 0.35))
        cost = 0.0
        min_clearance = 999.0
        for point in points[1:]:
            p = point.position
            for obs in perception.obstacles:
                rel = obs.position_at(point.t) - p
                clearance = math.hypot(rel.x, rel.y) - self._soft_radius(obs)
                min_clearance = min(min_clearance, clearance)
                if clearance < soft_clearance:
                    gap = soft_clearance - clearance
                    cost += gap * gap
                if clearance < 0.0:
                    cost += 25.0 + 50.0 * abs(clearance)
        self._last_min_soft_clearance = float(min_clearance)
        return cost / max(1, len(points) - 1), float(min_clearance)

    def _side_commitment_cost(self, points: Sequence[TrajectoryPoint], start: Vec3, left: Vec3) -> float:
        if abs(self._committed_side) < 0.5 or len(points) < 2:
            return 0.0
        final = points[-1].position
        displacement = Vec3(final.x - start.x, final.y - start.y, 0.0)
        lateral = displacement.x * left.x + displacement.y * left.y
        if lateral * self._committed_side >= 0.04:
            return 0.0
        return abs(lateral - self._committed_side * 0.04)

    def _update_side_commitment(self, state: PlannerState, ref_dir: Vec3, reference_side: float, perception: Optional[PerceptionData]) -> None:
        now = float(getattr(state, "stamp", 0.0))
        if now < self._committed_until and abs(self._committed_side) > 0.5:
            return
        side = self._sign(reference_side)
        nearest = self._nearest_front_obstacle(state, ref_dir, perception)
        if nearest is not None:
            obs_forward, obs_lateral, clearance = nearest
            if 0.0 <= obs_forward <= 4.2 and clearance < float(getattr(self.config, "sampling_mpc_commit_clearance", 1.05)):
                side = -1.0 if obs_lateral > 0.0 else 1.0
        if abs(side) > 0.5:
            hold = float(getattr(self.config, "sampling_mpc_side_hold_sec", 1.6))
            self._committed_side = side
            self._committed_until = now + max(0.2, hold)
        elif now >= self._committed_until:
            self._committed_side = 0.0

    def _nearest_front_obstacle(self, state: PlannerState, ref_dir: Vec3, perception: Optional[PerceptionData]):
        if perception is None:
            return None
        left = Vec3(-ref_dir.y, ref_dir.x, 0.0)
        best = None
        best_dist = 999.0
        for obs in perception.obstacles:
            obs_pos = obs.position_at(0.0)
            rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)
            forward = rel.x * ref_dir.x + rel.y * ref_dir.y
            lateral = rel.x * left.x + rel.y * left.y
            if forward < -0.15 or abs(lateral) > 2.0:
                continue
            dist = rel.norm()
            if dist >= best_dist:
                continue
            clearance = dist - self._soft_radius(obs)
            best = (float(forward), float(lateral), float(clearance))
            best_dist = dist
        return best

    def _soft_radius(self, obs) -> float:
        margin = min(float(getattr(self.config, "obstacle_margin", 0.45)), 0.05)
        if getattr(obs, "source", "") == "boundary":
            margin = 0.03
        elif getattr(obs, "source", "") == "lidar_near_field":
            margin = 0.15
        return float(getattr(self.config, "drone_radius", 0.35)) + float(obs.radius) + margin

    def _path_progress(self, reference_points: Sequence[Vec3], pos: Vec3, local_goal: Vec3) -> float:
        if len(reference_points) < 2:
            start = reference_points[0] if reference_points else pos
            goal_vec = Vec3(local_goal.x - start.x, local_goal.y - start.y, 0.0)
            unit = goal_vec.normalized(Vec3(1.0, 0.0, 0.0))
            rel = Vec3(pos.x - start.x, pos.y - start.y, 0.0)
            return rel.x * unit.x + rel.y * unit.y
        best_s = 0.0
        best_dist = 999999.0
        cumulative = 0.0
        for a, b in zip(reference_points[:-1], reference_points[1:]):
            ab = Vec3(b.x - a.x, b.y - a.y, 0.0)
            length = ab.norm()
            if length <= 1.0e-6:
                continue
            ap = Vec3(pos.x - a.x, pos.y - a.y, 0.0)
            u = max(0.0, min(1.0, (ap.x * ab.x + ap.y * ab.y) / (length * length)))
            proj = Vec3(a.x + ab.x * u, a.y + ab.y * u, a.z + (b.z - a.z) * u)
            dist = pos.distance_to(proj)
            if dist < best_dist:
                best_dist = dist
                best_s = cumulative + u * length
            cumulative += length
        return float(best_s)

    def _tracking_target(
        self,
        reference_points: Sequence[Vec3],
        local_goal: Vec3,
        ref_progress_now: float,
        nominal_speed: float,
        dt: float,
        horizon_steps: int,
    ) -> Vec3:
        if not reference_points:
            return local_goal
        if len(reference_points) < 2:
            return reference_points[0]
        lookahead = max(0.65, min(1.35, max(0.1, nominal_speed) * dt * max(horizon_steps, 1) * 0.65))
        return self._point_at_reference_progress(reference_points, ref_progress_now + lookahead)

    @staticmethod
    def _point_at_reference_progress(reference_points: Sequence[Vec3], target_s: float) -> Vec3:
        if not reference_points:
            return Vec3()
        if len(reference_points) == 1:
            return reference_points[0]
        cumulative = 0.0
        for a, b in zip(reference_points[:-1], reference_points[1:]):
            segment = Vec3(b.x - a.x, b.y - a.y, b.z - a.z)
            length = segment.norm()
            if length <= 1.0e-6:
                continue
            if cumulative + length >= target_s:
                u = max(0.0, min(1.0, (target_s - cumulative) / length))
                return Vec3(a.x + segment.x * u, a.y + segment.y * u, a.z + segment.z * u)
            cumulative += length
        return reference_points[-1]

    @staticmethod
    def _accel_limited(prev: Vec3, target: Vec3, max_delta: float) -> Vec3:
        if max_delta <= 0.0:
            return target
        delta = target - prev
        mag = delta.norm()
        if mag <= max_delta or mag <= 1.0e-9:
            return target
        return prev + delta * (max_delta / mag)

    @staticmethod
    def _sign(value: float) -> float:
        if value > 0.5:
            return 1.0
        if value < -0.5:
            return -1.0
        return 0.0

    @staticmethod
    def _seed_from_state(state: PlannerState, local_goal: Vec3) -> int:
        stamp = int(round(float(state.stamp) * 100.0))
        px = int(round(float(state.position.x) * 50.0))
        py = int(round(float(state.position.y) * 50.0))
        gx = int(round(float(local_goal.x) * 10.0))
        gy = int(round(float(local_goal.y) * 10.0))
        return (stamp * 73856093) ^ (px * 19349663) ^ (py * 83492791) ^ (gx * 2654435761) ^ gy
