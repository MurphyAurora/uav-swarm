#!/usr/bin/env python3
"""Smoke tests for the ROS-independent planner/benchmark contracts."""

from xtd2_mission.planner_framework import (
    BenchmarkStep,
    EgoLikePlannerAdapter,
    MotionLimits,
    NativePlannerSystemAdapter,
    Obstacle,
    PerceptionData,
    PlannerConfig,
    PlannerObservation,
    PlannerRequest,
    PlannerState,
    ScenarioContract,
    Trajectory,
    Vec3,
    VehicleState,
    trajectory_from_points,
)


def _request(stamp: float = 1.0) -> PlannerRequest:
    perception = PerceptionData(
        obstacles=[Obstacle(position=Vec3(2.0, 1.0, 1.5), radius=0.3, source="static")],
        stamp=stamp,
    )
    return PlannerRequest(
        state=PlannerState(
            drone_id=1,
            position=Vec3(0.0, 0.0, 1.5),
            velocity=Vec3(),
            stamp=stamp,
            heading=0.0,
        ),
        final_goal=Vec3(3.0, 0.0, 1.5),
        perception=perception,
        limits=MotionLimits(max_velocity=0.75, max_acceleration=0.6),
        scenario_id="smoke_open_space",
        seed=7,
    )


def test_request_builds_portable_extensions_without_ros():
    request = _request()

    assert isinstance(request.vehicle_state, VehicleState)
    assert isinstance(request.observation, PlannerObservation)
    assert request.vehicle_state.position == request.state.position
    assert len(request.observation.obstacles) == 1
    assert request.observation.obstacles[0].source == "static"


def test_portable_trajectory_reports_duration_and_length():
    trajectory = trajectory_from_points(
        [Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(1.0, 1.0, 0.0)],
        dt=0.5,
        source="smoke",
    )

    assert isinstance(trajectory, Trajectory)
    assert len(trajectory) == 3
    assert trajectory.duration == 1.0
    assert abs(trajectory.path_length - 2.0) < 1.0e-9
    assert trajectory.to_dict()["source"] == "smoke"


def test_ego_like_adapter_returns_portable_result():
    planner = EgoLikePlannerAdapter(PlannerConfig(frontend_mode="local_astar"))
    result = planner.plan(_request())

    assert result.planner_name == "ego_like"
    assert result.success
    assert result.planning_time_ms >= 0.0
    assert isinstance(result.command, Vec3)
    assert isinstance(result.trajectory, Trajectory)
    assert len(result.trajectory) >= 2
    assert "command" in result.raw_report
    assert result.diagnostics["scenario_id"] == "smoke_open_space"


def test_native_system_adapter_resets_and_steps():
    system = NativePlannerSystemAdapter(EgoLikePlannerAdapter())
    scenario = ScenarioContract(scenario_id="smoke_open_space", seed=7)
    system.reset(scenario)

    output = system.step(BenchmarkStep(timestamp=1.0, request=_request()))

    assert output.planner_result is not None
    assert output.planner_result.trajectory is not None
    assert output.diagnostics["scenario_id"] == "smoke_open_space"
    assert output.diagnostics["planner_name"] == "ego_like"
