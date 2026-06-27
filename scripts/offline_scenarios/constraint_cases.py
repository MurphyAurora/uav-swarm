from __future__ import annotations

from .base import CircleObstacle, SimCase, default_bounds, v3


def boundary_pressure_gap() -> SimCase:
    return SimCase(
        name="boundary_pressure_gap",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            # Lower row pushes the local planner upward, while the upper boundary
            # limits how far it can escape. The case is deliberately passable:
            # it should test boundary pressure, not create a near-impossible wall.
            CircleObstacle(3.0, -0.90, 0.44, obstacle_id="lower_1"),
            CircleObstacle(4.7, -0.72, 0.46, obstacle_id="lower_2"),
            CircleObstacle(6.4, -0.82, 0.46, obstacle_id="lower_3"),
            CircleObstacle(8.2, -0.70, 0.44, obstacle_id="lower_4"),
            CircleObstacle(10.0, -0.88, 0.44, obstacle_id="lower_5"),
            CircleObstacle(6.1, 1.30, 0.30, obstacle_id="upper_guard_1"),
            CircleObstacle(8.6, 1.28, 0.28, obstacle_id="upper_guard_2"),
        ),
        bounds=default_bounds(14.5, half_width=2.00),
        steps=760,
    )


def corridor_bottleneck() -> SimCase:
    return SimCase(
        name="corridor_bottleneck",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            # Wide entry, a local throat, then a wider exit. The throat should
            # trigger cautious motion, but it remains clearly traversable with the
            # offline safety profile.
            CircleObstacle(2.8, 1.65, 0.38, obstacle_id="top_entry"),
            CircleObstacle(2.8, -1.65, 0.38, obstacle_id="bottom_entry"),
            CircleObstacle(5.1, 1.35, 0.40, obstacle_id="top_neck_1"),
            CircleObstacle(5.1, -1.35, 0.40, obstacle_id="bottom_neck_1"),
            CircleObstacle(6.8, 1.20, 0.40, obstacle_id="top_throat"),
            CircleObstacle(6.8, -1.20, 0.40, obstacle_id="bottom_throat"),
            CircleObstacle(8.4, 1.38, 0.40, obstacle_id="top_neck_2"),
            CircleObstacle(8.4, -1.38, 0.40, obstacle_id="bottom_neck_2"),
            CircleObstacle(10.6, 1.70, 0.36, obstacle_id="top_exit"),
            CircleObstacle(10.6, -1.70, 0.36, obstacle_id="bottom_exit"),
        ),
        bounds=default_bounds(14.5, half_width=2.65),
        steps=820,
    )


def narrow_corridor_offset() -> SimCase:
    return SimCase(
        name="narrow_corridor_offset",
        start=v3(0.0, 0.0),
        goal=v3(13.0, 0.0),
        obstacles=(
            # Offset corridor: the feasible centerline shifts slightly along x.
            # It should require smooth small corrections without becoming a hard
            # geometric squeeze.
            CircleObstacle(2.8, 1.34, 0.40, obstacle_id="top_1"),
            CircleObstacle(4.8, 1.52, 0.40, obstacle_id="top_2"),
            CircleObstacle(6.8, 1.32, 0.40, obstacle_id="top_3"),
            CircleObstacle(8.8, 1.52, 0.40, obstacle_id="top_4"),
            CircleObstacle(10.8, 1.34, 0.40, obstacle_id="top_5"),
            CircleObstacle(2.8, -1.52, 0.40, obstacle_id="bottom_1"),
            CircleObstacle(4.8, -1.32, 0.40, obstacle_id="bottom_2"),
            CircleObstacle(6.8, -1.52, 0.40, obstacle_id="bottom_3"),
            CircleObstacle(8.8, -1.32, 0.40, obstacle_id="bottom_4"),
            CircleObstacle(10.8, -1.52, 0.40, obstacle_id="bottom_5"),
        ),
        bounds=default_bounds(13.5, half_width=2.60),
        steps=760,
    )
