#!/usr/bin/env python3
"""Smoke tests for the ROS-independent offline benchmark."""

from xtd2_mission.offline_benchmark import OfflineBenchmarkRunner, make_scenario
from xtd2_mission.planner_framework import EgoLikePlannerAdapter, PlannerConfig


def test_open_space_reaches_goal_without_collision():
    planner = EgoLikePlannerAdapter(PlannerConfig(frontend_mode="local_astar"))
    result = OfflineBenchmarkRunner(planner).run(make_scenario("open_space"))

    assert result.metrics.success
    assert not result.metrics.collision
    assert result.metrics.planning_steps > 0
    assert result.metrics.final_distance_to_goal <= 0.50


def test_dynamic_scenario_produces_finite_metrics():
    planner = EgoLikePlannerAdapter(PlannerConfig(frontend_mode="local_astar"))
    result = OfflineBenchmarkRunner(planner).run(make_scenario("dynamic_crossing"))

    assert result.metrics.planning_steps > 0
    assert result.metrics.completion_time > 0.0
    assert result.metrics.minimum_clearance < 999.0
    assert result.metrics.mean_planning_time_ms >= 0.0
    assert result.metrics.failure_reason in ("", "collision", "timeout")
