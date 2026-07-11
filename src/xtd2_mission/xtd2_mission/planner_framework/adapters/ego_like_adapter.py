"""Benchmark adapter for the native EGO-like planner.

This file intentionally keeps the benchmark interface separate from ROS2.
Future planners (EGO-Swarm, SOGM, ORCA, etc.) can implement the same style
of adapter without changing the benchmark runner.
"""

from ..planner_api import EgoLikePlannerAdapter


class EgoLikeBenchmarkAdapter(EgoLikePlannerAdapter):
    """Adapter entry point for EGO-like planner evaluation."""

    name = "ego_like"

    def planner_name(self) -> str:
        return self.name
