"""EGO-like lightweight planner framework.

This package is an incremental refactor target for the existing
fix/lidar-risk-astar-safety planner logic.  The first version keeps the
interfaces separated so the old runnable logic can be moved in module by
module without changing behavior all at once.
"""

from .common import (
    CandidateEvaluation,
    CandidateTrajectory,
    Obstacle,
    PerceptionData,
    PlannerCommand,
    PlannerConfig,
    PlannerState,
    SafetyReport,
    TrajectoryPoint,
    Vec3,
)
from .planner_core import EgoLikePlannerCore

__all__ = [
    "CandidateEvaluation",
    "CandidateTrajectory",
    "EgoLikePlannerCore",
    "Obstacle",
    "PerceptionData",
    "PlannerCommand",
    "PlannerConfig",
    "PlannerState",
    "SafetyReport",
    "TrajectoryPoint",
    "Vec3",
]
