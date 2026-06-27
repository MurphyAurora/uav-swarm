from __future__ import annotations

from typing import Callable, Dict

from .base import SimCase
from .constraint_cases import boundary_pressure_gap, corridor_bottleneck, narrow_corridor_offset
from .dynamic_cases import (
    dynamic_crossing_fast,
    dynamic_crossing_slow,
    head_on_fast,
    head_on_slow,
    mixed_static_dynamic_crossing,
)
from .generated_cases import random_forest_dense, random_forest_medium, random_forest_sparse
from .static_cases import (
    forest_gap,
    narrow_corridor,
    narrow_corridor_bias,
    narrow_corridor_easy,
    narrow_corridor_hard,
    narrow_corridor_medium,
    rejoin_shell,
    s_curve_easy,
    s_curve_medium,
    single_pillar,
    single_pillar_left_bias,
    single_pillar_right_bias,
    two_corridors,
    u_trap_shallow,
)

CASE_REGISTRY: Dict[str, Callable[[], SimCase]] = {
    # Original smoke-test cases kept for backward compatibility.
    "single_pillar": single_pillar,
    "forest_gap": forest_gap,
    "rejoin_shell": rejoin_shell,
    "narrow_corridor_easy": narrow_corridor_easy,
    "narrow_corridor_medium": narrow_corridor_medium,
    "narrow_corridor_hard": narrow_corridor_hard,
    "narrow_corridor": narrow_corridor,
    "narrow_corridor_bias": narrow_corridor_bias,
    # Extra static cases for diagnosing planner behavior beyond corridors.
    "boundary_pressure_gap": boundary_pressure_gap,
    "corridor_bottleneck": corridor_bottleneck,
    "narrow_corridor_offset": narrow_corridor_offset,
    "single_pillar_left_bias": single_pillar_left_bias,
    "single_pillar_right_bias": single_pillar_right_bias,
    "s_curve_easy": s_curve_easy,
    "s_curve_medium": s_curve_medium,
    "two_corridors": two_corridors,
    "u_trap_shallow": u_trap_shallow,
    # Dynamic obstacle cases.
    "dynamic_crossing_slow": dynamic_crossing_slow,
    "dynamic_crossing_fast": dynamic_crossing_fast,
    "head_on_slow": head_on_slow,
    "head_on_fast": head_on_fast,
    "mixed_static_dynamic_crossing": mixed_static_dynamic_crossing,
    # Reproducible generated cases.
    "random_forest_sparse": random_forest_sparse,
    "random_forest_medium": random_forest_medium,
    "random_forest_dense": random_forest_dense,
}


def get_case(name: str) -> SimCase:
    try:
        return CASE_REGISTRY[name]()
    except KeyError as exc:
        choices = ", ".join(sorted(CASE_REGISTRY))
        raise KeyError(f"unknown offline case {name!r}; available: {choices}") from exc


def build_cases() -> Dict[str, SimCase]:
    return {name: factory() for name, factory in CASE_REGISTRY.items()}
