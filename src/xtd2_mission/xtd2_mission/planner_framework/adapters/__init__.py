"""Planner adapters for benchmark comparison.

Adapters isolate individual planning algorithms from the benchmark runner.
The runtime platform and ROS2 interfaces remain unchanged.
"""

from .ego_like_adapter import EgoLikeBenchmarkAdapter

__all__ = [
    "EgoLikeBenchmarkAdapter",
]
