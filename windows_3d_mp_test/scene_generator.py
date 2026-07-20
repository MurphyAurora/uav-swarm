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


def _vec3_to_array(vec, fallback_z=3.0):
    return np.array(
        [
            float(getattr(vec, "x", vec[0] if len(vec) > 0 else 0.0)),
            float(getattr(vec, "y", vec[1] if len(vec) > 1 else 0.0)),
            fallback_z,
        ],
        dtype=float,
    )


def load_offline_case(case_name="forest_gap", pillar_height=5.0, flight_height=3.0):
    """Lift a 2D offline scenario into a 3D cylinder scene.

    The XY obstacle distribution is preserved exactly. Only z=0 and obstacle
    height are added, which keeps the existing 2D validation geometry reusable.
    """
    from scripts.offline_scenarios.registry import get_case

    case = get_case(case_name)
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


def make_scene(name="forest_gap", pillar_height=5.0, flight_height=3.0):
    if name in {"forest_gap", "single_pillar", "narrow_corridor_easy", "narrow_corridor_medium", "narrow_corridor_hard"}:
        return load_offline_case(name, pillar_height=pillar_height, flight_height=flight_height)

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

    return load_offline_case(name, pillar_height=pillar_height, flight_height=flight_height)
