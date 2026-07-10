#!/usr/bin/env python3
"""Smoke tests for the ROS-independent planner/benchmark contracts."""

from xtd2_mission.planner_framework import (
    BenchmarkStep,
    EgoLikePlannerAdapter,
    NativePlannerSystemAdapter,
    PerceptionData,
    PlannerConfig,
    PlannerRequest,
    PlannerState,
    ScenarioContract,
    Vec3,
)


def _request(stamp: float = 1.0) -> PlannerRequest:
    return PlannerRequest(
        state=PlannerState(
            drone_id=1,
            position=Vec3(0.0, 0.0, 1.5),
            velocity=Vec3(),
            stamp=stamp,
            heading=0.0,
        ),
        final_goal=Vec3(3.0, 0.0, 1.5),
        perception=PerceptionData(stamp=stamp),
        scenario_id="smoke_open_space",
        seed=7,
    )


def test_ego_like_adapter_returns_portable_result():
    planner = EgoLikePlannerAdapter(PlannerConfig(frontend_mode="local_astar"))
    result = planner.plan(_request())

    assert result.planner_name == "ego_like"
    assert result.success
    assert result.planning_time_ms >= 0.0
    assert isinstance(result.command, Vec3)
    assert "command" in result.raw_report


def test_native_system_adapter_resets_and_steps():
    system = NativePlannerSystemAdapter(EgoLikePlannerAdapter())
    scenario = ScenarioContract(scenario_id="smoke_open_space", seed=7)
    system.reset(scenario)

    output = system.step(BenchmarkStep(timestamp=1.0, request=_request()))

    assert output.planner_result is not None
    assert output.diagnostics["scenario_id"] == "smoke_open_space"
    assert output.diagnostics["planner_name"] == "ego_like"
