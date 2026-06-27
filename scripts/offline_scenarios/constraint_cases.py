from __future__ import annotations

from .base import CircleObstacle, SimCase, default_bounds, v3


def boundary_pressure_gap() -> SimCase:
    return SimCase(
        name="boundary_pressure_gap",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            CircleObstacle(3.1, -0.95, 0.46, obstacle_id="lower_1"),
            CircleObstacle(4.8, -0.75, 0.46, obstacle_id="lower_2"),
            CircleObstacle(6.4, -0.95, 0.46, obstacle_id="lower_3"),
            CircleObstacle(8.2, -0.70, 0.44, obstacle_id="lower_4"),
            CircleObstacle(10.0, -0.95, 0.44, obstacle_id="lower_5"),
            CircleObstacle(6.1, 1.55, 0.34, obstacle_id="upper_guard"),
        ),
        bounds=default_bounds(14.5, half_width=2.25),
        steps=700,
    )


def corridor_bottleneck() -> SimCase:
    return SimCase(
        name="corridor_bottleneck",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            CircleObstacle(3.0, 1.65, 0.38, obstacle_id="top_entry"),
            CircleObstacle(3.0, -1.65, 0.38, obstacle_id="bottom_entry"),
            CircleObstacle(5.5, 1.35, 0.42, obstacle_id="top_neck_1"),
            CircleObstacle(5.5, -1.35, 0.42, obstacle_id="bottom_neck_1"),
            CircleObstacle(7.2, 1.18, 0.42, obstacle_id="top_neck_2"),
            CircleObstacle(7.2, -1.18, 0.42, obstacle_id="bottom_neck_2"),
            CircleObstacle(9.0, 1.45, 0.40, obstacle_id="top_exit"),
            CircleObstacle(9.0, -1.45, 0.40, obstacle_id="bottom_exit"),
            CircleObstacle(11.0, 1.75, 0.36, obstacle_id="top_far"),
            CircleObstacle(11.0, -1.75, 0.36, obstacle_id="bottom_far"),
        ),
        bounds=default_bounds(14.5, half_width=2.75),
        steps=800,
    )


def narrow_corridor_offset() -> SimCase:
    return SimCase(
        name="narrow_corridor_offset",
        start=v3(0.0, 0.0),
        goal=v3(13.0, 0.0),
        obstacles=(
            CircleObstacle(2.8, 1.30, 0.40, obstacle_id="top_1"),
            CircleObstacle(4.8, 1.48, 0.40, obstacle_id="top_2"),
            CircleObstacle(6.8, 1.26, 0.40, obstacle_id="top_3"),
            CircleObstacle(8.8, 1.48, 0.40, obstacle_id="top_4"),
            CircleObstacle(10.8, 1.30, 0.40, obstacle_id="top_5"),
            CircleObstacle(2.8, -1.48, 0.40, obstacle_id="bottom_1"),
            CircleObstacle(4.8, -1.26, 0.40, obstacle_id="bottom_2"),
            CircleObstacle(6.8, -1.48, 0.40, obstacle_id="bottom_3"),
            CircleObstacle(8.8, -1.26, 0.40, obstacle_id="bottom_4"),
            CircleObstacle(10.8, -1.48, 0.40, obstacle_id="bottom_5"),
        ),
        bounds=default_bounds(13.5, half_width=2.65),
        steps=750,
    )
