"""EGO-like lightweight planner framework.

This package is an incremental refactor target for the existing
fix/lidar-risk-astar-safety planner logic.  The first version keeps the
interfaces separated so the old runnable logic can be moved in module by
module without changing behavior all at once.
"""

from .common import (
    PlannerState,
    Vec3,
    Obstacle,
    CandidateTrajectory,
    PlannerCommand,
    PerceptionData,
)

__all__ = [
    "PlannerState",
    "Vec3",
    "Obstacle",
    "CandidateTrajectory",
    "PlannerCommand",
    "PerceptionData",
]
