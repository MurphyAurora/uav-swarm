#!/usr/bin/env python3
"""Minimal deterministic simulator for planner-level comparisons.

This module intentionally does not import ROS2, Gazebo, or PX4. The vehicle
uses a common velocity-command executor so every native planner is evaluated
under the same simplified dynamics.
"""

from dataclasses import dataclass
from typing import List

from ..planner_framework import (
    EgoLikePlannerAdapter,
    MotionLimits,
    Obstacle,
    PerceptionData,
    PlannerBase,
    PlannerRequest,
    PlannerState,
    Vec3,
)
from .metrics import EpisodeMetrics, MetricsCollector
from .scenario import OfflineScenario, ScenarioObstacle


@dataclass
class OfflineRunResult:
    metrics: EpisodeMetrics
    final_position: Vec3
    reports: List[dict]


class OfflineBenchmarkRunner:
    """Run one planner in one deterministic scenario."""

    def __init__(self, planner: PlannerBase | None = None, vehicle_radius: float = 0.35) -> None:
        self.planner = planner or EgoLikePlannerAdapter()
        self.vehicle_radius = float(vehicle_radius)

    def run(self, scenario: OfflineScenario) -> OfflineRunResult:
        self.planner.reset()
        position = scenario.start
        velocity = Vec3()
        collector = MetricsCollector(scenario.scenario_id, self.planner.name)
        collector.update_position(position)
        reports: List[dict] = []
        collided = False
        reached = False
        elapsed = 0.0

        while elapsed <= scenario.duration_sec:
            obstacles = self._visible_obstacles(scenario, position, elapsed)
            perception = self._perception(position, obstacles, elapsed)
            state = PlannerState(
                drone_id=1,
                position=position,
                velocity=velocity,
                stamp=elapsed,
                heading=0.0,
            )
            result = self.planner.plan(
                PlannerRequest(
                    state=state,
                    final_goal=scenario.goal,
                    perception=perception,
                    scenario_id=scenario.scenario_id,
                    limits=MotionLimits(vehicle_radius=self.vehicle_radius),
                )
            )
            reports.append(result.to_dict())
            command_mode = str(result.raw_report.get("command", {}).get("mode", ""))
            collector.update_planner(result.planning_time_ms, result.mode, command_mode)

            velocity = result.command
            position = position + velocity * scenario.dt
            elapsed += scenario.dt
            collector.update_position(position)

            clearance = self._minimum_clearance(scenario, position, elapsed)
            collector.update_clearance(clearance)
            if clearance <= scenario.collision_margin:
                collided = True
                break
            if position.distance_to(scenario.goal) <= scenario.goal_tolerance:
                reached = True
                break

        metrics = collector.metrics
        metrics.success = reached and not collided
        metrics.collision = collided
        metrics.timeout = not reached and not collided
        metrics.completion_time = elapsed
        metrics.final_distance_to_goal = position.distance_to(scenario.goal)
        if collided:
            metrics.failure_reason = "collision"
        elif not reached:
            metrics.failure_reason = "timeout"
        collector.finish()
        return OfflineRunResult(metrics=metrics, final_position=position, reports=reports)

    def _visible_obstacles(
        self,
        scenario: OfflineScenario,
        position: Vec3,
        timestamp: float,
    ) -> List[Obstacle]:
        visible: List[Obstacle] = []
        for item in scenario.obstacles:
            obstacle_position = item.position_at(timestamp)
            if position.distance_to(obstacle_position) > scenario.sensing_radius:
                continue
            visible.append(
                Obstacle(
                    position=obstacle_position,
                    velocity=item.velocity,
                    radius=item.radius,
                    source=item.source,
                    obstacle_id=item.obstacle_id,
                )
            )
        return visible

    @staticmethod
    def _perception(position: Vec3, obstacles: List[Obstacle], timestamp: float) -> PerceptionData:
        nearest = min(
            (position.distance_to(item.position) - item.radius for item in obstacles),
            default=999.0,
        )
        return PerceptionData(
            obstacles=obstacles,
            nearest_distance=float(nearest),
            near_field_danger=False,
            frame_id="world",
            stamp=float(timestamp),
        )

    def _minimum_clearance(self, scenario: OfflineScenario, position: Vec3, timestamp: float) -> float:
        return min(
            (
                position.distance_to(item.position_at(timestamp))
                - self.vehicle_radius
                - item.radius
                for item in scenario.obstacles
            ),
            default=999.0,
        )
