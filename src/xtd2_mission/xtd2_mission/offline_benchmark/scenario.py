#!/usr/bin/env python3
"""Deterministic, ROS-independent benchmark scenarios."""

from dataclasses import dataclass, field
from typing import Dict, List

from ..planner_framework import Vec3


@dataclass(frozen=True)
class ScenarioObstacle:
    obstacle_id: str
    initial_position: Vec3
    velocity: Vec3 = field(default_factory=Vec3)
    radius: float = 0.30
    source: str = "dynamic"

    def position_at(self, timestamp: float) -> Vec3:
        return self.initial_position + self.velocity * max(0.0, float(timestamp))


@dataclass(frozen=True)
class OfflineScenario:
    scenario_id: str
    start: Vec3
    goal: Vec3
    obstacles: List[ScenarioObstacle] = field(default_factory=list)
    duration_sec: float = 20.0
    dt: float = 0.10
    sensing_radius: float = 7.5
    goal_tolerance: float = 0.50
    collision_margin: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)


def make_scenario(name: str) -> OfflineScenario:
    scenarios = {
        "open_space": OfflineScenario(
            scenario_id="open_space",
            start=Vec3(0.0, 0.0, 1.5),
            goal=Vec3(6.0, 0.0, 1.5),
        ),
        "single_static_column": OfflineScenario(
            scenario_id="single_static_column",
            start=Vec3(0.0, 0.0, 1.5),
            goal=Vec3(6.0, 0.0, 1.5),
            obstacles=[
                ScenarioObstacle(
                    obstacle_id="column_1",
                    initial_position=Vec3(3.0, 0.0, 1.5),
                    radius=0.55,
                    source="static",
                )
            ],
        ),
        "dynamic_crossing": OfflineScenario(
            scenario_id="dynamic_crossing",
            start=Vec3(0.0, 0.0, 1.5),
            goal=Vec3(6.0, 0.0, 1.5),
            obstacles=[
                ScenarioObstacle(
                    obstacle_id="cross_1",
                    initial_position=Vec3(3.0, -2.0, 1.5),
                    velocity=Vec3(0.0, 0.65, 0.0),
                    radius=0.40,
                )
            ],
        ),
        "dynamic_head_on": OfflineScenario(
            scenario_id="dynamic_head_on",
            start=Vec3(0.0, 0.0, 1.5),
            goal=Vec3(6.0, 0.0, 1.5),
            obstacles=[
                ScenarioObstacle(
                    obstacle_id="head_on_1",
                    initial_position=Vec3(5.5, 0.0, 1.5),
                    velocity=Vec3(-0.55, 0.0, 0.0),
                    radius=0.40,
                )
            ],
        ),
    }
    try:
        return scenarios[str(name)]
    except KeyError as exc:
        raise ValueError(f"unknown offline scenario: {name}") from exc
