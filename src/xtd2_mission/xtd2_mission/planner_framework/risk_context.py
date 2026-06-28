#!/usr/bin/env python3
"""Risk-context helpers for mode-aware local trajectory selection.

A mature unified MPC does not use one global stop/escape preference for every
case.  Static obstacles, moving head-on obstacles, moving crossing obstacles,
and mixed scenes all share the same candidate rollout pipeline, but their
selection priorities differ.

This module keeps that context inference separate from candidate generation and
final command arbitration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from .common import CandidateEvaluation, PerceptionData, PlannerConfig, PlannerState, Vec3
from .dynamic_safety import dynamic_ttc_snapshot, is_dynamic_obstacle, obstacle_velocity


class RiskKind(str, Enum):
    LOW_RISK = "LOW_RISK"
    STATIC_ONLY = "STATIC_ONLY"
    DYNAMIC_HEAD_ON = "DYNAMIC_HEAD_ON"
    DYNAMIC_CROSSING = "DYNAMIC_CROSSING"
    MIXED_STATIC_DYNAMIC = "MIXED_STATIC_DYNAMIC"


@dataclass
class RiskContext:
    kind: RiskKind
    static_count: int = 0
    dynamic_count: int = 0
    dynamic_ttc: float = 999.0
    dynamic_closing: float = 0.0
    dynamic_clearance: float = 999.0
    head_on_score: float = 0.0
    crossing_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "risk_kind": self.kind.value,
            "static_count": int(self.static_count),
            "dynamic_count": int(self.dynamic_count),
            "dynamic_ttc": float(self.dynamic_ttc),
            "dynamic_closing": float(self.dynamic_closing),
            "dynamic_clearance": float(self.dynamic_clearance),
            "head_on_score": float(self.head_on_score),
            "crossing_score": float(self.crossing_score),
        }


def classify_perception_context(
    state: PlannerState,
    local_goal: Vec3,
    perception: Optional[PerceptionData],
    config: PlannerConfig,
) -> RiskContext:
    """Classify current obstacle context from perception.

    The result is intentionally simple and robust.  It does not decide the final
    command; it only gives the selector a context so static scenes are not forced
    to use dynamic stop/yield behavior and dynamic head-on scenes are not allowed
    to treat hover as safe.
    """
    if perception is None or not perception.obstacles:
        return RiskContext(RiskKind.LOW_RISK)

    static_count = 0
    dynamic_count = 0
    head_on_score = 0.0
    crossing_score = 0.0
    goal_dir = Vec3(local_goal.x - state.position.x, local_goal.y - state.position.y, 0.0).normalized(Vec3(1.0, 0.0, 0.0))
    ego_vel = Vec3(state.velocity.x, state.velocity.y, 0.0)

    for obs in perception.obstacles:
        if getattr(obs, "source", "") == "boundary":
            continue
        if not is_dynamic_obstacle(obs):
            static_count += 1
            continue
        dynamic_count += 1
        obs_pos = obs.position_at(0.0)
        rel = Vec3(obs_pos.x - state.position.x, obs_pos.y - state.position.y, 0.0)
        dist = max(rel.norm(), 1.0e-6)
        rel_unit = rel * (1.0 / dist)
        obs_vel = obstacle_velocity(obs)
        rel_vel = obs_vel - ego_vel
        closing = -((rel.x * rel_vel.x + rel.y * rel_vel.y) / dist)
        if closing <= 0.0:
            continue
        obs_motion = obs_vel.normalized(Vec3())
        # Head-on: obstacle motion is mostly opposite the local goal/forward axis
        # and is closing. Crossing: obstacle motion is mostly lateral to the goal.
        along_goal = obs_motion.x * goal_dir.x + obs_motion.y * goal_dir.y
        lateral_motion = abs(obs_motion.x * (-goal_dir.y) + obs_motion.y * goal_dir.x)
        head_on_score = max(head_on_score, closing * max(0.0, -along_goal))
        crossing_score = max(crossing_score, closing * lateral_motion)

    dyn_ttc, dyn_closing, dyn_clearance, _ = dynamic_ttc_snapshot(state, perception, config)
    ctx = RiskContext(
        RiskKind.LOW_RISK,
        static_count=static_count,
        dynamic_count=dynamic_count,
        dynamic_ttc=dyn_ttc,
        dynamic_closing=dyn_closing,
        dynamic_clearance=dyn_clearance,
        head_on_score=head_on_score,
        crossing_score=crossing_score,
    )

    dynamic_active = dynamic_count > 0 and (dyn_closing > 0.02 or dyn_ttc < float(getattr(config, "dynamic_avoid_ttc", 1.20)) or dyn_clearance < float(getattr(config, "dynamic_avoid_clearance", 0.75)))
    if static_count > 0 and dynamic_active:
        ctx.kind = RiskKind.MIXED_STATIC_DYNAMIC
    elif dynamic_active and head_on_score >= max(0.03, crossing_score * 0.85):
        ctx.kind = RiskKind.DYNAMIC_HEAD_ON
    elif dynamic_active:
        ctx.kind = RiskKind.DYNAMIC_CROSSING
    elif static_count > 0:
        ctx.kind = RiskKind.STATIC_ONLY
    else:
        ctx.kind = RiskKind.LOW_RISK
    return ctx


def infer_evaluation_context(evaluations: Iterable[CandidateEvaluation]) -> RiskContext:
    """Infer context when only evaluated candidates are available.

    BackendSelector does not receive raw perception.  SamplingMPC stores dynamic
    metadata on candidates, so we use that metadata to avoid applying dynamic
    stop/escape priorities to static-only scenes.
    """
    items = list(evaluations)
    dynamic_items = [item for item in items if (item.trajectory.metadata or {}).get("dynamic_side_hint", 0.0)]
    dynamic_risk = [item for item in items if bool((item.trajectory.metadata or {}).get("dynamic_risk_flag", False)) or bool((item.trajectory.metadata or {}).get("dynamic_corridor_flag", False))]
    dynamic_candidates = [item for item in items if str((item.trajectory.metadata or {}).get("sample_kind", "")).lower().startswith("dynamic_")]
    if dynamic_items or dynamic_risk or dynamic_candidates:
        # Candidate-only inference cannot robustly distinguish crossing vs head-on.
        # The selector only needs to know whether dynamic-specific priorities are
        # active; detailed command-mode logic still lives in PlannerCore/fallback.
        min_ttc = min((float((item.trajectory.metadata or {}).get("dynamic_min_ttc", 999.0)) for item in items), default=999.0)
        max_closing = max((float((item.trajectory.metadata or {}).get("dynamic_closing_speed", 0.0)) for item in items), default=0.0)
        min_clearance = min((float((item.trajectory.metadata or {}).get("dynamic_min_clearance", 999.0)) for item in items), default=999.0)
        return RiskContext(
            RiskKind.DYNAMIC_CROSSING,
            dynamic_count=len(dynamic_candidates) or len(dynamic_items),
            dynamic_ttc=min_ttc,
            dynamic_closing=max_closing,
            dynamic_clearance=min_clearance,
        )
    if items:
        return RiskContext(RiskKind.STATIC_ONLY, static_count=1)
    return RiskContext(RiskKind.LOW_RISK)


def is_dynamic_context(ctx: RiskContext) -> bool:
    return ctx.kind in {RiskKind.DYNAMIC_HEAD_ON, RiskKind.DYNAMIC_CROSSING, RiskKind.MIXED_STATIC_DYNAMIC}


def is_static_only_context(ctx: RiskContext) -> bool:
    return ctx.kind == RiskKind.STATIC_ONLY
