#!/usr/bin/env python3
"""Metrics used by the lightweight offline planner benchmark."""

from dataclasses import dataclass, field
from typing import Dict, List

from ..planner_framework import Vec3


@dataclass
class EpisodeMetrics:
    scenario_id: str
    planner_name: str
    success: bool = False
    collision: bool = False
    timeout: bool = False
    completion_time: float = 0.0
    path_length: float = 0.0
    minimum_clearance: float = 999.0
    mean_planning_time_ms: float = 0.0
    max_planning_time_ms: float = 0.0
    planning_steps: int = 0
    emergency_count: int = 0
    fallback_count: int = 0
    final_distance_to_goal: float = 999.0
    failure_reason: str = ""
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "planner_name": self.planner_name,
            "success": self.success,
            "collision": self.collision,
            "timeout": self.timeout,
            "completion_time": self.completion_time,
            "path_length": self.path_length,
            "minimum_clearance": self.minimum_clearance,
            "mean_planning_time_ms": self.mean_planning_time_ms,
            "max_planning_time_ms": self.max_planning_time_ms,
            "planning_steps": self.planning_steps,
            "emergency_count": self.emergency_count,
            "fallback_count": self.fallback_count,
            "final_distance_to_goal": self.final_distance_to_goal,
            "failure_reason": self.failure_reason,
            "diagnostics": dict(self.diagnostics),
        }


class MetricsCollector:
    def __init__(self, scenario_id: str, planner_name: str) -> None:
        self.metrics = EpisodeMetrics(scenario_id=scenario_id, planner_name=planner_name)
        self._last_position: Vec3 | None = None
        self._planning_times: List[float] = []

    def update_position(self, position: Vec3) -> None:
        if self._last_position is not None:
            self.metrics.path_length += self._last_position.distance_to(position)
        self._last_position = position

    def update_clearance(self, clearance: float) -> None:
        self.metrics.minimum_clearance = min(self.metrics.minimum_clearance, float(clearance))

    def update_planner(self, planning_time_ms: float, mode: str, command_mode: str) -> None:
        value = max(0.0, float(planning_time_ms))
        self._planning_times.append(value)
        self.metrics.planning_steps += 1
        if "EMERGENCY" in str(mode).upper():
            self.metrics.emergency_count += 1
        if str(command_mode).startswith("fallback"):
            self.metrics.fallback_count += 1

    def finish(self) -> EpisodeMetrics:
        if self._planning_times:
            self.metrics.mean_planning_time_ms = sum(self._planning_times) / len(self._planning_times)
            self.metrics.max_planning_time_ms = max(self._planning_times)
        return self.metrics
