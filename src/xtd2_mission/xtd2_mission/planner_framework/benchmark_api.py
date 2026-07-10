#!/usr/bin/env python3
"""System-level benchmark contracts.

PlannerBase is used for module-level comparisons. BenchmarkSystem is the
higher-level adapter boundary for complete external systems such as
EGO-Swarm or SOGM, whose internal map, optimizer, safety, and communication
pipelines should remain intact.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .planner_api import PlannerBase, PlannerRequest, PlannerResult


@dataclass
class ScenarioContract:
    scenario_id: str
    seed: int = 0
    duration_sec: float = 30.0
    sensing_radius: float = 5.0
    prediction_horizon: float = 2.0
    max_speed: float = 2.0
    max_accel: float = 6.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkStep:
    timestamp: float
    request: PlannerRequest
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkOutput:
    command: Any
    diagnostics: Dict[str, Any]
    planner_result: Optional[PlannerResult] = None


class BenchmarkSystem(ABC):
    """Common experiment boundary for native and external planner systems."""

    @abstractmethod
    def reset(self, scenario: ScenarioContract) -> None:
        pass

    @abstractmethod
    def step(self, data: BenchmarkStep) -> BenchmarkOutput:
        pass

    def close(self) -> None:
        """Optional resource cleanup hook."""


class NativePlannerSystemAdapter(BenchmarkSystem):
    """Wrap a PlannerBase implementation as a complete benchmark system."""

    def __init__(self, planner: PlannerBase) -> None:
        self.planner = planner
        self.scenario: Optional[ScenarioContract] = None

    def reset(self, scenario: ScenarioContract) -> None:
        self.scenario = scenario
        self.planner.reset()

    def step(self, data: BenchmarkStep) -> BenchmarkOutput:
        result = self.planner.plan(data.request)
        diagnostics = {
            "scenario_id": self.scenario.scenario_id if self.scenario else "",
            "seed": self.scenario.seed if self.scenario else None,
            "planner_name": result.planner_name,
            "planning_time_ms": result.planning_time_ms,
            "success": result.success,
            "failure_reason": result.failure_reason,
            "mode": result.mode,
            "trajectory_name": result.trajectory_name,
            "minimum_clearance": result.minimum_clearance,
        }
        return BenchmarkOutput(
            command=result.command,
            diagnostics=diagnostics,
            planner_result=result,
        )
