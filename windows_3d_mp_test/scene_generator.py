from pathlib import Path
import sys

import numpy as np

from obstacle import DynamicObstacle


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_HEIGHTS = {
    "low": 2.0,
    "medium": 5.0,
    "high": 8.0,
}

DEFAULT_SINGLE_PILLAR_RADIUS = 1.6

# Mirrored from scripts/offline_scenarios/static_cases.py::forest_gap.
# This fallback keeps windows_3d_mp_test runnable without importing the full
# xtd2_mission planner package used by the offline benchmark package.
FALLBACK_CASES = {
    "no_obstacle": {
        "start": [0.0, 0.0],
        "goal": [12.0, 0.0],
        "obstacles": [],
    },
    "forest_gap": {
        "start": [0.0, 0.0],
        "goal": [16.0, 0.0],
        "obstacles": [
            (3.0, -1.25, 0.36, "p1"),
            (4.6, 1.35, 0.36, "p2"),
            (6.3, 0.30, 0.40, "p3"),
            (8.3, -1.35, 0.38, "p4"),
            (10.0, 1.25, 0.38, "p5"),
            (11.8, 0.15, 0.36, "p6"),
        ],
    },
    "single_pillar": {
        "start": [0.0, 0.0],
        "goal": [12.0, 0.0],
        "obstacles": [(4.0, 0.0, DEFAULT_SINGLE_PILLAR_RADIUS, "pillar_1")],
    },
    "high_block": {
        "start": [0.0, 0.0],
        "goal": [15.0, 0.0],
        "obstacles": [(7.5, 0.0, 2.0, "high_block_1")],
        "height": DEFAULT_HEIGHTS["high"],
    },
    "test_side_bypass": {
        "start": [0.0, 0.0],
        "goal": [12.0, 0.0],
        "obstacles": [(5.0, 0.0, 1.0, "side_bypass_pillar")],
        "height": 3.0,
    },
    "test_over_top": {
        "start": [-6.0, 0.0],
        "goal": [15.0, 0.0],
        "obstacles": [(7.5, 0.0, 2.0, "over_top_pillar")],
        "height": 8.0,
        "bounds": (-6.5, 15.5, -1.2, 1.2),
    },
}


def _component(vec, name, index, default=0.0):
    if hasattr(vec, name):
        return float(getattr(vec, name))
    try:
        return float(vec[index])
    except (TypeError, IndexError):
        return float(default)


def _vec3_to_array(vec, fallback_z=3.0):
    return np.array(
        [
            _component(vec, "x", 0),
            _component(vec, "y", 1),
            float(fallback_z),
        ],
        dtype=float,
    )


def _fallback_case(case_name, pillar_height, flight_height, pillar_radius=None):
    try:
        data = FALLBACK_CASES[case_name]
    except KeyError as exc:
        choices = ", ".join(sorted(FALLBACK_CASES))
        raise KeyError(
            f"fallback case {case_name!r} is not available; available: {choices}"
        ) from exc

    scenario_height = float(data.get("height", pillar_height))
    obstacles = []
    for x, y, radius, obstacle_id in data["obstacles"]:
        if case_name == "single_pillar" and pillar_radius is not None:
            radius = float(pillar_radius)
        obstacles.append(
            DynamicObstacle.static_pillar(
                x=x,
                y=y,
                radius=radius,
                height=scenario_height,
                obstacle_id=obstacle_id,
            )
        )

    return {
        "name": case_name,
        "start": np.array([data["start"][0], data["start"][1], flight_height], dtype=float),
        "goal": np.array([data["goal"][0], data["goal"][1], flight_height], dtype=float),
        "obstacles": obstacles,
        "bounds": data.get("bounds"),
    }


def load_offline_case(case_name="forest_gap", pillar_height=5.0, flight_height=3.0, pillar_radius=None):
    """Lift a 2D offline scenario into a 3D cylinder scene.

    The XY obstacle distribution is preserved exactly. Only z=0 and obstacle
    height are added, which keeps the existing 2D validation geometry reusable.
    """
    if case_name in FALLBACK_CASES:
        return _fallback_case(case_name, pillar_height, flight_height, pillar_radius=pillar_radius)

    try:
        from scripts.offline_scenarios.registry import get_case
        case = get_case(case_name)
    except (ModuleNotFoundError, KeyError):
        return _fallback_case(case_name, pillar_height, flight_height, pillar_radius=pillar_radius)

    start = _vec3_to_array(case.start, fallback_z=flight_height)
    goal = _vec3_to_array(case.goal, fallback_z=flight_height)

    obstacles = [
        DynamicObstacle.from_2d_circle(item, height=pillar_height)
        for item in case.obstacles
    ]

    for item in getattr(case, "dynamic_obstacles", ()):
        x, y = item.position_at(0.0)
        obstacles.append(
            DynamicObstacle.ground_vehicle(
                position=[x, y, 0.0],
                velocity=[item.vx, item.vy, 0.0],
                radius=item.radius,
                height=1.8,
                obstacle_id=getattr(item, "obstacle_id", ""),
            )
        )

    return {
        "name": case.name,
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "bounds": getattr(case, "bounds", None),
    }


def make_scene(name="forest_gap", pillar_height=5.0, flight_height=3.0, pillar_radius=None):
    if name in {
        "no_obstacle",
        "forest_gap",
        "single_pillar",
        "high_block",
        "test_side_bypass",
        "test_over_top",
        "narrow_corridor_easy",
        "narrow_corridor_medium",
        "narrow_corridor_hard",
    }:
        return load_offline_case(
            name,
            pillar_height=pillar_height,
            flight_height=flight_height,
            pillar_radius=pillar_radius,
        )

    if name == "forest_gap_high":
        return load_offline_case("forest_gap", pillar_height=DEFAULT_HEIGHTS["high"], flight_height=flight_height)

    if name == "forest_gap_dynamic":
        scene = load_offline_case("forest_gap", pillar_height=pillar_height, flight_height=flight_height)
        scene["name"] = name
        scene["obstacles"].append(
            DynamicObstacle.ground_vehicle(
                position=[7.0, -2.2, 0.0],
                velocity=[0.0, 0.35, 0.0],
                radius=0.45,
                height=1.6,
                obstacle_id="crossing_vehicle",
            )
        )
        return scene

    if name == "forest_gap_flying":
        scene = load_offline_case("forest_gap", pillar_height=pillar_height, flight_height=flight_height)
        scene["name"] = name
        scene["obstacles"].append(
            DynamicObstacle.flying_obstacle(
                position=[7.0, -2.0, flight_height],
                velocity=[0.2, 0.25, 0.05],
                radius=0.45,
                obstacle_id="crossing_uav",
            )
        )
        return scene

    return load_offline_case(
        name,
        pillar_height=pillar_height,
        flight_height=flight_height,
        pillar_radius=pillar_radius,
    )
