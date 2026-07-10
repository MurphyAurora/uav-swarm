#!/usr/bin/env python3
"""ROS-independent planner contracts and adapters.

This module deliberately contains no rclpy, ROS message, Gazebo, or PX4
imports.  It provides the stable boundary shared by the existing ROS2
runtime and future offline/system-level benchmark adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, Sequence

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3
from .planner_core import EgoLikePlannerCore


@dataclass
class PlannerRequest:
    """Portable input passed to a local planner implementation."""

    state: PlannerState
    final_goal: Vec3
    perception: PerceptionData = field(default_factory=PerceptionData)
    scenario_id: str = ""
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannerResult:
    """Portable planner result used by ROS2 and offline runners."""

    success: bool
    command: Vec3
    raw_report: Dict[str, Any]
    planning_time_ms: float
    planner_name: str
    failure_reason: str = ""

    @property
    def trajectory_name(self) -> str:
        return str(self.raw_report.get("command", {}).get("source_trajectory", ""))

    @property
    def mode(self) -> str:
        return str(self.raw_report.get("planner_fsm", {}).get("mode", ""))

    @property
    def minimum_clearance(self) -> float:
        risk = self.raw_report.get("planner_fsm", {}).get("risk", {})
        return float(risk.get("min_clearance", 999.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "planner_name": self.planner_name,
            "planning_time_ms": float(self.planning_time_ms),
            "failure_reason": self.failure_reason,
            "mode": self.mode,
            "trajectory_name": self.trajectory_name,
            "minimum_clearance": self.minimum_clearance,
            "command": self.command.to_dict(),
            "raw_report": self.raw_report,
        }


class PlannerBase(ABC):
    """Minimal common interface for replaceable planner algorithms."""

    name = "planner"

    @abstractmethod
    def reset(self) -> None:
        """Clear state that must not leak between benchmark episodes."""

    @abstractmethod
    def plan(self, request: PlannerRequest) -> PlannerResult:
        """Plan once using only portable data models."""


class EgoLikePlannerAdapter(PlannerBase):
    """Thin adapter around the already-running EgoLikePlannerCore.

    The existing core and ROS2 node remain untouched.  This adapter only
    exposes the same implementation through a benchmark-friendly API.
    """

    name = "ego_like"

    def __init__(
        self,
        config: Optional[PlannerConfig] = None,
        waypoints: Optional[Sequence[Vec3]] = None,
    ) -> None:
        self.config = config or PlannerConfig()
        self.waypoints = list(waypoints or [])
        self._core = EgoLikePlannerCore(self.config, waypoints=self.waypoints)

    @property
    def core(self) -> EgoLikePlannerCore:
        return self._core

    def reset(self) -> None:
        # Re-instantiation is intentional: it resets output smoothing, local-goal
        # progress, FSM holds, fallback memory, and planner warm-start state in one
        # place without changing the proven runtime implementation.
        self._core = EgoLikePlannerCore(self.config, waypoints=self.waypoints)

    def plan(self, request: PlannerRequest) -> PlannerResult:
        start = time.perf_counter()
        report = self._core.plan(request.state, request.final_goal, request.perception)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        command_data = report.get("command", {}).get("velocity", {})
        command = Vec3.from_mapping(command_data)
        failure_reason = ""
        if not report.get("mission_reached", False) and not report.get("command"):
            failure_reason = "missing_command"

        return PlannerResult(
            success=not bool(failure_reason),
            command=command,
            raw_report=report,
            planning_time_ms=elapsed_ms,
            planner_name=self.name,
            failure_reason=failure_reason,
        )
