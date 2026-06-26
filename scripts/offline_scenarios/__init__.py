"""Offline planner scenario library.

The modules in this package keep map/scenario definitions separate from
scripts/offline_planner_sim.py so the runner only executes simulations.
"""

from .registry import CASE_REGISTRY, build_cases, get_case

__all__ = ["CASE_REGISTRY", "build_cases", "get_case"]
