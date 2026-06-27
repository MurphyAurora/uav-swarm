from __future__ import annotations

from .base import CircleObstacle, SimCase, default_bounds, v3


def boundary_pressure_gap() -> SimCase:
    return SimCase(
        name="boundary_pressure_gap",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            # Lower row pushes the local planner upward, while the upper boundary
            # limits how far it can escape.  The upper guard creates a local pinch
            # so this is a true obstacle-boundary pressure case, not just a single
            # row of obstacles with free space above.
            CircleObstacle(3.0, -0.85, 0.46, obstacle_id="lower_1"),
            CircleObstacle(4.7, -0.68, 0.48, obstacle_id="lower_2"),
            CircleObstacle(6.4, -0.78, 0.48, obstacle_id="lower_3"),
            CircleObstacle(8.2, -0.62, 0.46, obstacle_id="lower_4"),
            CircleObstacle(10.0, -0.82, 0.46, obstacle_id="lower_5"),
            CircleObstacle(6.1, 1.22, 0.34, obstacle_id="upper_guard_1"),
            CircleObstacle(8.6, 1.18, 0.30, obstacle_id="upper_guard_2"),
        ),
        bounds=default_bounds(14.5, half_width=1.85),
        steps=760,
    )


def corridor_bottleneck() -> SimCase:
    return SimCase(
        name="corridor_bottleneck",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            # Wide entry, a short near-limit throat, then a wider exit.  The neck
            # is intentionally local: the planner should slow down through it and
            # then recover speed instead of staying in recovery forever.
            CircleObstacle(2.8, 1.65, 0.38, obstacle_id="top_entry"),
            CircleObstacle(2.8, -1.65, 0.38, obstacle_id="bottom_entry"),
            CircleObstacle(5.1, 1.28, 0.42, obstacle_id="top_neck_1"),
            CircleObstacle(5.1, -1.28, 0.42, obstacle_id="bottom_neck_1"),
            CircleObstacle(6.8, 1.08, 0.43, obstacle_id="top_throat"),
            CircleObstacle(6.8, -1.08, 0.43, obstacle_id="bottom_throat"),
            CircleObstacle(8.4, 1.30, 0.42, obstacle_id="top_neck_2"),
            CircleObstacle(8.4, -1.30, 0.42, obstacle_id="bottom_neck_2"),
            CircleObstacle(10.6, 1.70, 0.36, obstacle_id="top_exit"),
            CircleObstacle(10.6, -1.70, 0.36, obstacle_id="bottom_exit"),
        ),
        bounds=default_bounds(14.5, half_width=2.55),
        steps=850,
    )


def narrow_corridor_offset() -> SimCase:
    return SimCase(
        name="narrow_corridor_offset",
        start=v3(0.0, 0.0),
        goal=v3(13.0, 0.0),
        obstacles=(
            # Offset corridor: the feasible centerline shifts by a small amount
            # along x.  This should cause smooth corrections, not a full S-curve
            # detour and not boundary hugging.
            CircleObstacle(2.8, 1.24, 0.41, obstacle_id="top_1"),
            CircleObstacle(4.8, 1.48, 0.41, obstacle_id="top_2"),
            CircleObstacle(6.8, 1.22, 0.41, obstacle_id="top_3"),
            CircleObstacle(8.8, 1.50, 0.41, obstacle_id="top_4"),
            CircleObstacle(10.8, 1.26, 0.41, obstacle_id="top_5"),
            CircleObstacle(2.8, -1.50, 0.41, obstacle_id="bottom_1"),
            CircleObstacle(4.8, -1.22, 0.41, obstacle_id="bottom_2"),
            CircleObstacle(6.8, -1.52, 0.41, obstacle_id="bottom_3"),
            CircleObstacle(8.8, -1.24, 0.41, obstacle_id="bottom_4"),
            CircleObstacle(10.8, -1.50, 0.41, obstacle_id="bottom_5"),
        ),
        bounds=default_bounds(13.5, half_width=2.50),
        steps=800,
    )
