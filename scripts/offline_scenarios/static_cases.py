from __future__ import annotations

from .base import CircleObstacle, SimCase, default_bounds, v3


def single_pillar() -> SimCase:
    return SimCase(
        name="single_pillar",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        obstacles=(CircleObstacle(4.0, 0.0, 0.55, obstacle_id="pillar_1"),),
        bounds=default_bounds(12.5, half_width=3.0),
    )


def single_pillar_left_bias() -> SimCase:
    return SimCase(
        name="single_pillar_left_bias",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        obstacles=(CircleObstacle(4.0, 0.65, 0.55, obstacle_id="pillar_left_bias"),),
        bounds=default_bounds(12.5, half_width=3.0),
    )


def single_pillar_right_bias() -> SimCase:
    return SimCase(
        name="single_pillar_right_bias",
        start=v3(0.0, 0.0),
        goal=v3(12.0, 0.0),
        obstacles=(CircleObstacle(4.0, -0.65, 0.55, obstacle_id="pillar_right_bias"),),
        bounds=default_bounds(12.5, half_width=3.0),
    )


def forest_gap() -> SimCase:
    return SimCase(
        name="forest_gap",
        start=v3(0.0, 0.0),
        goal=v3(16.0, 0.0),
        obstacles=(
            # Sparse forest: scattered trunks with several valid gaps.  This case
            # should test multi-obstacle gap selection and rejoin-to-goal behavior,
            # not force a single S-curve or a boundary-hugging solution.
            CircleObstacle(3.0, -1.25, 0.36, obstacle_id="p1"),
            CircleObstacle(4.6, 1.35, 0.36, obstacle_id="p2"),
            CircleObstacle(6.3, 0.30, 0.40, obstacle_id="p3"),
            CircleObstacle(8.3, -1.35, 0.38, obstacle_id="p4"),
            CircleObstacle(10.0, 1.25, 0.38, obstacle_id="p5"),
            CircleObstacle(11.8, 0.15, 0.36, obstacle_id="p6"),
        ),
        bounds=default_bounds(16.5, half_width=3.4),
        steps=700,
    )


def rejoin_shell() -> SimCase:
    return SimCase(
        name="rejoin_shell",
        start=v3(0.0, 0.0),
        goal=v3(8.0, 0.0),
        obstacles=(
            CircleObstacle(1.1, -0.8, 0.50, obstacle_id="near_side"),
            CircleObstacle(3.8, 0.8, 0.45, obstacle_id="front_side"),
        ),
        bounds=default_bounds(8.5, half_width=2.8),
        steps=450,
    )


def s_curve_easy() -> SimCase:
    return SimCase(
        name="s_curve_easy",
        start=v3(0.0, 0.0),
        goal=v3(13.0, 0.0),
        obstacles=(
            # Easy S-curve should test weaving, not create a hard-inflated deadlock.
            # Keep the alternating pattern, but leave enough room for drone_radius
            # and hard/emergency margins used by the offline smoke planner.
            CircleObstacle(3.0, -0.90, 0.50, obstacle_id="s1"),
            CircleObstacle(5.2, 0.90, 0.50, obstacle_id="s2"),
            CircleObstacle(7.4, -0.90, 0.50, obstacle_id="s3"),
            CircleObstacle(9.6, 0.90, 0.50, obstacle_id="s4"),
        ),
        bounds=default_bounds(13.5, half_width=2.20),
        steps=700,
    )


def s_curve_medium() -> SimCase:
    return SimCase(
        name="s_curve_medium",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            CircleObstacle(2.8, -0.50, 0.58, obstacle_id="s1"),
            CircleObstacle(4.9, 0.55, 0.58, obstacle_id="s2"),
            CircleObstacle(7.0, -0.55, 0.58, obstacle_id="s3"),
            CircleObstacle(9.1, 0.55, 0.58, obstacle_id="s4"),
            CircleObstacle(11.2, -0.50, 0.58, obstacle_id="s5"),
        ),
        bounds=default_bounds(14.5, half_width=2.10),
        steps=750,
    )


def two_corridors() -> SimCase:
    return SimCase(
        name="two_corridors",
        start=v3(0.0, 0.0),
        goal=v3(14.0, 0.0),
        obstacles=(
            CircleObstacle(4.0, 0.0, 0.65, obstacle_id="center_1"),
            CircleObstacle(5.8, 0.0, 0.65, obstacle_id="center_2"),
            CircleObstacle(7.6, 0.0, 0.65, obstacle_id="center_3"),
            CircleObstacle(9.4, 0.0, 0.65, obstacle_id="center_4"),
            CircleObstacle(6.0, 2.0, 0.45, obstacle_id="top_pressure"),
            CircleObstacle(8.0, -2.0, 0.45, obstacle_id="bottom_pressure"),
        ),
        bounds=default_bounds(14.5, half_width=3.0),
        steps=750,
    )


def u_trap_shallow() -> SimCase:
    return SimCase(
        name="u_trap_shallow",
        start=v3(0.0, 0.0),
        goal=v3(10.0, 0.0),
        obstacles=(
            CircleObstacle(4.0, 1.0, 0.45, obstacle_id="u_top_1"),
            CircleObstacle(5.0, 1.0, 0.45, obstacle_id="u_top_2"),
            CircleObstacle(6.0, 1.0, 0.45, obstacle_id="u_top_3"),
            CircleObstacle(4.0, -1.0, 0.45, obstacle_id="u_bottom_1"),
            CircleObstacle(5.0, -1.0, 0.45, obstacle_id="u_bottom_2"),
            CircleObstacle(6.0, -1.0, 0.45, obstacle_id="u_bottom_3"),
            CircleObstacle(6.4, 0.0, 0.45, obstacle_id="u_end"),
        ),
        bounds=default_bounds(10.5, half_width=2.8),
        steps=650,
    )


def narrow_corridor_easy() -> SimCase:
    return SimCase(
        name="narrow_corridor_easy",
        start=v3(0.0, 0.0),
        goal=v3(13.0, 0.0),
        obstacles=(
            CircleObstacle(2.8, 1.75, 0.40, obstacle_id="top_1"),
            CircleObstacle(4.8, 1.75, 0.40, obstacle_id="top_2"),
            CircleObstacle(6.8, 1.75, 0.40, obstacle_id="top_3"),
            CircleObstacle(8.8, 1.75, 0.40, obstacle_id="top_4"),
            CircleObstacle(10.8, 1.75, 0.40, obstacle_id="top_5"),
            CircleObstacle(2.8, -1.75, 0.40, obstacle_id="bottom_1"),
            CircleObstacle(4.8, -1.75, 0.40, obstacle_id="bottom_2"),
            CircleObstacle(6.8, -1.75, 0.40, obstacle_id="bottom_3"),
            CircleObstacle(8.8, -1.75, 0.40, obstacle_id="bottom_4"),
            CircleObstacle(10.8, -1.75, 0.40, obstacle_id="bottom_5"),
        ),
        bounds=default_bounds(13.5, half_width=2.9),
        steps=650,
    )


def narrow_corridor() -> SimCase:
    return SimCase(
        name="narrow_corridor",
        start=v3(0.0, 0.0),
        goal=v3(13.0, 0.0),
        obstacles=(
            CircleObstacle(2.8, 1.60, 0.42, obstacle_id="top_1"),
            CircleObstacle(4.7, 1.55, 0.42, obstacle_id="top_2"),
            CircleObstacle(6.6, 1.60, 0.42, obstacle_id="top_3"),
            CircleObstacle(8.5, 1.55, 0.42, obstacle_id="top_4"),
            CircleObstacle(10.4, 1.60, 0.42, obstacle_id="top_5"),
            CircleObstacle(3.7, -1.60, 0.42, obstacle_id="bottom_1"),
            CircleObstacle(5.6, -1.55, 0.42, obstacle_id="bottom_2"),
            CircleObstacle(7.5, -1.60, 0.42, obstacle_id="bottom_3"),
            CircleObstacle(9.4, -1.55, 0.42, obstacle_id="bottom_4"),
            CircleObstacle(11.3, -1.60, 0.42, obstacle_id="bottom_5"),
        ),
        bounds=default_bounds(13.5, half_width=2.7),
        steps=650,
    )


def narrow_corridor_bias() -> SimCase:
    return SimCase(
        name="narrow_corridor_bias",
        start=v3(0.0, 0.0),
        goal=v3(13.0, 0.0),
        obstacles=(
            CircleObstacle(2.8, 1.55, 0.42, obstacle_id="top_1"),
            CircleObstacle(4.7, 1.55, 0.42, obstacle_id="top_2"),
            CircleObstacle(6.6, 1.55, 0.42, obstacle_id="top_3"),
            CircleObstacle(8.5, 1.55, 0.42, obstacle_id="top_4"),
            CircleObstacle(10.4, 1.55, 0.42, obstacle_id="top_5"),
            CircleObstacle(3.7, -1.55, 0.42, obstacle_id="bottom_1"),
            CircleObstacle(5.6, -1.55, 0.42, obstacle_id="bottom_2"),
            CircleObstacle(7.5, -1.55, 0.42, obstacle_id="bottom_3"),
            CircleObstacle(9.4, -1.55, 0.42, obstacle_id="bottom_4"),
            CircleObstacle(11.3, -1.55, 0.42, obstacle_id="bottom_5"),
            CircleObstacle(6.4, 0.45, 0.20, obstacle_id="center_bias"),
        ),
        bounds=default_bounds(13.5, half_width=2.6),
        steps=700,
    )
