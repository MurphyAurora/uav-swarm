#!/usr/bin/env python3
"""ROS-independent planner contracts and adapters.

This module deliberately contains no rclpy, ROS message, Gazebo, or PX4
imports. It provides the stable boundary shared by the existing ROS2 runtime
and future offline/system-level benchmark adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, Sequence

from .common import PerceptionData, PlannerConfig, PlannerState, Vec3
from .planner_core import EgoLikePlannerCore
from .planner_types import (
    MotionLimits,
    PlannerObservation,
    Trajectory,
    VehicleState,
    trajectory_from_points,
)


@dataclass
class PlannerRequest:
    """Portable input passed to a local planner implementation.

    ``state``, ``final_goal``, and ``perception`` retain compatibility with the
    proven runtime core. The additional portable fields are optional extension
    points for offline baselines such as EGO-Swarm- or SOGM-style adapters.
    """

    state: PlannerState
    final_goal: Vec3
    perception: PerceptionData = field(default_factory=PerceptionData)
    local_goal: Optional[Vec3] = None
    vehicle_state: Optional[VehicleState] = None
    observation: Optional[PlannerObservation] = None
    previous_trajectory: Optional[Trajectory] = None
    limits: MotionLimits = field(default_factory=MotionLimits)
    scenario_id: str = ""
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vehicle_state is None:
            self.vehicle_state = VehicleState.from_planner_state(self.state)
        if self.observation is None:
            self.observation = PlannerObservation.from_perception(self.perception)


@dataclass
class PlannerResult:
    """Portable planner result used by ROS2 and offline runners."""

    success: bool
    command: Vec3
    raw_report: Dict[str, Any]
    planning_time_ms: float
    planner_name: str
    trajectory: Optional[Trajectory] = None
    failure_reason: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

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

    @property
    def minimum_ttc(self) -> float:
        risk = self.raw_report.get("planner_fsm", {}).get("risk", {})
        return float(risk.get("min_ttc", 999.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": bool(self.success),
            "planner_name": self.planner_name,
            "planning_time_ms": float(self.planning_time_ms),
            "failure_reason": self.failure_reason,
            "mode": self.mode,
            "trajectory_name": self.trajectory_name,
            "minimum_clearance": self.minimum_clearance,
            "minimum_ttc": self.minimum_ttc,
            "command": self.command.to_dict(),
            "trajectory": self.trajectory.to_dict() if self.trajectory is not None else None,
            "diagnostics": dict(self.diagnostics),
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

    The existing core and ROS2 node remain untouched. This adapter only exposes
    the same implementation through a benchmark-friendly API.
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
        # Re-instantiation resets output smoothing, local-goal progress, FSM
        # holds, fallback memory, and planner warm-start state without changing
        # the proven runtime implementation.
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

        trajectory = self._trajectory_from_report(report, request)
        diagnostics = {
            "candidate_count": int(report.get("candidate_count", 0) or 0),
            "safe_count": int(report.get("safe_count", 0) or 0),
            "feasible_count": int(report.get("feasible_count", 0) or 0),
            "fallback_active": str(report.get("command", {}).get("mode", "")).startswith("fallback"),
            "scenario_id": request.scenario_id,
            "seed": request.seed,
        }

        return PlannerResult(
            success=not bool(failure_reason),
            command=command,
            raw_report=report,
            planning_time_ms=elapsed_ms,
            planner_name=self.name,
            trajectory=trajectory,
            failure_reason=failure_reason,
            diagnostics=diagnostics,
        )

    def _trajectory_from_report(self, report: Dict[str, Any], request: PlannerRequest) -> Trajectory:
        """Convert the selected native candidate to the portable trajectory.

        The runtime core currently exposes candidate points through the compact
        evaluation diagnostics. If no selected point sequence is available, a
        one-step command trajectory is emitted so benchmark consumers still
        receive a valid, explicit output rather than relying on ROS commands.
        """

        selected_name = str(report.get("command", {}).get("source_trajectory", ""))
        evaluations = report.get("evaluations", []) or []
        positions = []
        for item in evaluations:
            if str(item.get("name", "")) != selected_name:
                continue
            for point in item.get("points", []) or []:
                if isinstance(point, dict):
                    positions.append(Vec3.from_mapping(point))
            break

        if not positions:
            start = request.state.position
            dt = max(1.0e-3, float(getattr(self.config, "dt", 0.5)))
            positions = [start, start + Vec3.from_mapping(report.get("command", {}).get("velocity", {})) * dt]
        else:
            dt = max(1.0e-3, float(getattr(self.config, "dt", 0.5)))

        return trajectory_from_points(
            positions,
            dt=dt,
            start_time=float(request.state.stamp),
            source=selected_name or self.name,
        )
