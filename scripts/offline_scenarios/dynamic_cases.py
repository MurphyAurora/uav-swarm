from __future__ import annotations

from .base import CircleObstacle, MovingCircleObstacle, SimCase, default_bounds, v3


# The crossing cases below are intentionally time-aligned conflict tests.
# With a nominal UAV speed around 0.50~0.60 m/s, the UAV reaches the crossing
# line x=6.0 at about 10~12 s.  The moving obstacles are initialized so they
# also reach y=0 around that time.  A straight-line, no-reaction policy should
# therefore have near-collision/collision risk; passing these cases requires the
# planner to predict the crossing and either slow down, yield, or laterally
# separate from the moving obstacle corridor.


def dynamic_crossing_slow() -> SimCase:
    return SimCase(
        name="dynamic_crossing_slow",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        dynamic_obstacles=(
            # Crosses y=0 near t=10.8 s at x=6.0.
            MovingCircleObstacle(6.0, -5.2, 0.0, 0.48, 0.42, obstacle_id="cross_conflict_slow"),
        ),
        bounds=default_bounds(12.5, half_width=3.2),
        steps=650,
    )


def dynamic_crossing_fast() -> SimCase:
    return SimCase(
        name="dynamic_crossing_fast",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        dynamic_obstacles=(
            # Crosses y=0 near t=10.5 s at x=6.0.  This is a pressure test, not
            # a low-pressure timing miss: a nominal straight line should meet the
            # obstacle near the crossing point.
            MovingCircleObstacle(6.0, -8.2, 0.0, 0.78, 0.42, obstacle_id="cross_conflict_fast"),
        ),
        bounds=default_bounds(12.5, half_width=3.2),
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
        bounds=default_bounds(12.5, half_width=3.0),
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
        bounds=default_bounds(12.5, half_width=3.0),
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
            # Two timed crossings around the two static-obstacle gaps.
            MovingCircleObstacle(6.5, -5.4, 0.0, 0.50, 0.38, obstacle_id="cross_conflict_1"),
            MovingCircleObstacle(10.5, 6.0, 0.0, -0.46, 0.38, obstacle_id="cross_conflict_2"),
        ),
        bounds=default_bounds(14.5, half_width=3.2),
        steps=800,
    )
