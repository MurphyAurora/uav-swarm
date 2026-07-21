from pathlib import Path
import random
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


def _heighted_obstacles(obstacles, heights):
    return [tuple(obstacle) + (float(height),) for obstacle, height in zip(obstacles, heights)]


def _random_forest_case(
    seed,
    n,
    length=16.0,
    width=6.0,
    min_start_clear=1.8,
    min_goal_clear=1.2,
    height_range=None,
    center_corridor_half_width=0.65,
    center_corridor_max_radius=0.36,
    max_center_corridor_obstacles=4,
):
    rng = random.Random(seed)
    obstacles = []
    center_corridor_count = 0
    attempts = 0
    while len(obstacles) < n and attempts < n * 160:
        attempts += 1
        x = rng.uniform(2.0, length - 2.0)
        y = rng.uniform(-width * 0.5, width * 0.5)
        radius = rng.uniform(0.25, 0.55)

        if x < min_start_clear and abs(y) < 1.4:
            continue
        if x > length - min_goal_clear and abs(y) < 1.4:
            continue

        in_center_corridor = abs(y) < center_corridor_half_width
        if in_center_corridor:
            if center_corridor_count >= max_center_corridor_obstacles:
                continue
            radius = min(radius, center_corridor_max_radius)
            if rng.random() < 0.45:
                continue

        ok = True
        for obs in obstacles:
            obs_x, obs_y, obs_radius = obs[0], obs[1], obs[2]
            min_sep = radius + obs_radius + 0.45
            if (x - obs_x) ** 2 + (y - obs_y) ** 2 < min_sep ** 2:
                ok = False
                break
        if not ok:
            continue

        obstacle = (x, y, radius, f"tree_{len(obstacles):02d}")
        if height_range is not None:
            obstacle = obstacle + (rng.uniform(height_range[0], height_range[1]),)
        obstacles.append(obstacle)
        if in_center_corridor:
            center_corridor_count += 1

    return {
        "start": [0.0, 0.0],
        "goal": [length, 0.0],
        "obstacles": obstacles,
        "bounds": (-0.6, length + 0.5, -width * 0.5, width * 0.5),
    }


S_CURVE_MEDIUM_OBSTACLES = [
    (2.8, -0.50, 0.58, "s1"),
    (4.9, 0.55, 0.58, "s2"),
    (7.0, -0.55, 0.58, "s3"),
    (9.1, 0.55, 0.58, "s4"),
    (11.2, -0.50, 0.58, "s5"),
]

FOREST_GAP_OBSTACLES = [
    (3.0, -1.25, 0.36, "p1"),
    (4.6, 1.35, 0.36, "p2"),
    (6.3, 0.30, 0.40, "p3"),
    (8.3, -1.35, 0.38, "p4"),
    (10.0, 1.25, 0.38, "p5"),
    (11.8, 0.15, 0.36, "p6"),
]


# Mirrored from scripts/offline_scenarios/static_cases.py and generated_cases.py.
# This fallback keeps windows_3d_mp_test runnable without importing xtd2_mission.
FALLBACK_CASES = {
    "no_obstacle": {
        "start": [0.0, 0.0],
        "goal": [12.0, 0.0],
        "obstacles": [],
    },
    "forest_gap": {
        "start": [0.0, 0.0],
        "goal": [16.0, 0.0],
        "obstacles": FOREST_GAP_OBSTACLES,
        "bounds": (-0.6, 16.5, -3.4, 3.4),
    },
    "forest_gap_mixed_height": {
        "start": [0.0, 0.0],
        "goal": [16.0, 0.0],
        "obstacles": _heighted_obstacles(FOREST_GAP_OBSTACLES, [2.4, 6.2, 3.2, 7.5, 4.0, 6.8]),
        "bounds": (-0.6, 16.5, -3.4, 3.4),
    },
    "s_curve_easy": {
        "start": [0.0, 0.0],
        "goal": [13.0, 0.0],
        "obstacles": [
            (3.0, -0.90, 0.50, "s1"),
            (5.2, 0.90, 0.50, "s2"),
            (7.4, -0.90, 0.50, "s3"),
            (9.6, 0.90, 0.50, "s4"),
        ],
        "bounds": (-0.6, 13.5, -2.2, 2.2),
    },
    "s_curve_medium": {
        "start": [0.0, 0.0],
        "goal": [14.0, 0.0],
        "obstacles": S_CURVE_MEDIUM_OBSTACLES,
        "bounds": (-0.6, 14.5, -2.1, 2.1),
    },
    "s_curve_mixed_height": {
        "start": [0.0, 0.0],
        "goal": [14.0, 0.0],
        "obstacles": _heighted_obstacles(S_CURVE_MEDIUM_OBSTACLES, [2.2, 7.2, 3.8, 6.4, 2.8]),
        "bounds": (-0.6, 14.5, -2.1, 2.1),
    },
    "random_forest_sparse": _random_forest_case(seed=0, n=12, width=6.5),
    "random_forest_medium": _random_forest_case(seed=1, n=20, width=6.0),
    "random_forest_dense": _random_forest_case(seed=2, n=30, width=6.0),
    "random_forest_mixed_sparse": _random_forest_case(seed=10, n=12, width=6.5, height_range=(2.0, 8.0)),
    "random_forest_mixed_medium": _random_forest_case(seed=11, n=20, width=6.0, height_range=(2.0, 8.0)),
    "random_forest_mixed_dense": _random_forest_case(seed=12, n=30, width=6.0, height_range=(2.0, 8.0)),
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


def _fallback_case(case_name, pillar_height, flight_height, pillar_radius=None, height_range=None, height_seed=0):
    try:
        data = FALLBACK_CASES[case_name]
    except KeyError as exc:
        choices = ", ".join(sorted(FALLBACK_CASES))
        raise KeyError(
            f"fallback case {case_name!r} is not available; available: {choices}"
        ) from exc

    scenario_height = float(data.get("height", pillar_height))
    height_rng = random.Random(height_seed)
    obstacles = []
    for obstacle in data["obstacles"]:
        x, y, radius, obstacle_id = obstacle[:4]
        obstacle_height = float(obstacle[4]) if len(obstacle) >= 5 else scenario_height
        if height_range is not None:
            obstacle_height = height_rng.uniform(height_range[0], height_range[1])
        if case_name == "single_pillar" and pillar_radius is not None:
            radius = float(pillar_radius)
        obstacles.append(
            DynamicObstacle.static_pillar(
                x=x,
                y=y,
                radius=radius,
                height=obstacle_height,
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


def load_offline_case(
    case_name="forest_gap",
    pillar_height=5.0,
    flight_height=3.0,
    pillar_radius=None,
    height_range=None,
    height_seed=0,
):
    """Lift a 2D offline scenario into a 3D cylinder scene.

    The XY obstacle distribution is preserved exactly. Only z=0 and obstacle
    height are added, which keeps the existing 2D validation geometry reusable.
    """
    if case_name in FALLBACK_CASES:
        return _fallback_case(
            case_name,
            pillar_height,
            flight_height,
            pillar_radius=pillar_radius,
            height_range=height_range,
            height_seed=height_seed,
        )

    try:
        from scripts.offline_scenarios.registry import get_case
        case = get_case(case_name)
    except (ModuleNotFoundError, KeyError):
        return _fallback_case(
            case_name,
            pillar_height,
            flight_height,
            pillar_radius=pillar_radius,
            height_range=height_range,
            height_seed=height_seed,
        )

    start = _vec3_to_array(case.start, fallback_z=flight_height)
    goal = _vec3_to_array(case.goal, fallback_z=flight_height)

    height_rng = random.Random(height_seed)
    obstacles = []
    for item in case.obstacles:
        obstacle_height = height_rng.uniform(height_range[0], height_range[1]) if height_range is not None else pillar_height
        obstacles.append(DynamicObstacle.from_2d_circle(item, height=obstacle_height))

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


def make_scene(
    name="forest_gap",
    pillar_height=5.0,
    flight_height=3.0,
    pillar_radius=None,
    height_range=None,
    height_seed=0,
):
    if name in {
        "no_obstacle",
        "forest_gap",
        "forest_gap_mixed_height",
        "s_curve_easy",
        "s_curve_medium",
        "s_curve_mixed_height",
        "random_forest_sparse",
        "random_forest_medium",
        "random_forest_dense",
        "random_forest_mixed_sparse",
        "random_forest_mixed_medium",
        "random_forest_mixed_dense",
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
            height_range=height_range,
            height_seed=height_seed,
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
        height_range=height_range,
        height_seed=height_seed,
    )
