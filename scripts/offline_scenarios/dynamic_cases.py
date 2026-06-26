from __future__ import annotations

from .base import CircleObstacle, MovingCircleObstacle, SimCase, v3


def dynamic_crossing_slow() -> SimCase:
    return SimCase(
        name="dynamic_crossing_slow",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        dynamic_obstacles=(
            MovingCircleObstacle(6.0, -3.0, 0.0, 0.45, 0.40, obstacle_id="cross_slow"),
        ),
        steps=650,
    )


def dynamic_crossing_fast() -> SimCase:
    return SimCase(
        name="dynamic_crossing_fast",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        dynamic_obstacles=(
            MovingCircleObstacle(6.0, -3.0, 0.0, 0.85, 0.40, obstacle_id="cross_fast"),
        ),
        steps=650,
    )


def head_on_slow() -> SimCase:
    return SimCase(
        name="head_on_slow",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        dynamic_obstacles=(
            MovingCircleObstacle(10.0, 0.0, -0.45, 0.0, 0.42, obstacle_id="head_on_slow"),
        ),
        steps=700,
    )


def head_on_fast() -> SimCase:
    return SimCase(
        name="head_on_fast",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        dynamic_obstacles=(
            MovingCircleObstacle(10.0, 0.0, -0.80, 0.0, 0.42, obstacle_id="head_on_fast"),
        ),
        steps=700,
    )


def mixed_static_dynamic_crossing() -> SimCase:
    return SimCase(
        name="mixed_static_dynamic_crossing",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            CircleObstacle(4.0, 1.0, 0.45, obstacle_id="static_top"),
            CircleObstacle(8.0, -1.0, 0.45, obstacle_id="static_bottom"),
        ),
        dynamic_obstacles=(
            MovingCircleObstacle(6.5, -3.0, 0.0, 0.55, 0.38, obstacle_id="cross_1"),
            MovingCircleObstacle(10.5, 3.0, 0.0, -0.45, 0.38, obstacle_id="cross_2"),
        ),
        steps=800,
    )
