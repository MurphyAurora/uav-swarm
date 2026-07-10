#!/usr/bin/env python3
"""Portable planner data types shared by runtime and offline benchmarks.

The types in this module intentionally avoid ROS, Gazebo, PX4, and numpy
imports.  They provide a stable interchange format for native planners,
external baseline adapters, and the offline benchmark runner.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .common import Obstacle, PerceptionData, PlannerState, Vec3


@dataclass(frozen=True)
class VehicleState:
    drone_id: int
    position: Vec3
    velocity: Vec3 = field(default_factory=Vec3)
    acceleration: Vec3 = field(default_factory=Vec3)
    yaw: float = 0.0
    timestamp: float = 0.0

    @staticmethod
    def from_planner_state(state: PlannerState) -> "VehicleState":
        return VehicleState(
            drone_id=int(state.drone_id),
            position=state.position,
            velocity=state.velocity,
            yaw=float(state.heading),
            timestamp=float(state.stamp),
        )

    def to_planner_state(self) -> PlannerState:
        return PlannerState(
            drone_id=int(self.drone_id),
            position=self.position,
            velocity=self.velocity,
            stamp=float(self.timestamp),
            heading=float(self.yaw),
        )


@dataclass(frozen=True)
class MotionLimits:
    max_velocity: float = 0.75
    max_acceleration: float = 0.60
    max_jerk: float = 2.0
    vehicle_radius: float = 0.35


@dataclass(frozen=True)
class ObstacleState:
    obstacle_id: str
    position: Vec3
    velocity: Vec3 = field(default_factory=Vec3)
    radius: float = 0.30
    is_dynamic: bool = False
    source: str = "obstacle"

    @staticmethod
    def from_obstacle(obstacle: Obstacle) -> "ObstacleState":
        return ObstacleState(
            obstacle_id=str(obstacle.obstacle_id),
            position=obstacle.position,
            velocity=obstacle.velocity,
            radius=float(obstacle.radius),
            is_dynamic=obstacle.velocity.norm() > 1.0e-6 or bool(obstacle.predictions),
            source=str(obstacle.source),
        )


@dataclass(frozen=True)
class PredictedState:
    timestamp: float
    position: Vec3
    velocity: Vec3 = field(default_factory=Vec3)


@dataclass
class PredictedObstacle:
    obstacle: ObstacleState
    states: List[PredictedState] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(frozen=True)
class TrajectorySample:
    timestamp: float
    position: Vec3
    velocity: Vec3 = field(default_factory=Vec3)
    acceleration: Vec3 = field(default_factory=Vec3)


@dataclass
class Trajectory:
    samples: List[TrajectorySample] = field(default_factory=list)
    frame_id: str = "world"
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def empty(self) -> bool:
        return not self.samples

    @property
    def duration(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return max(0.0, float(self.samples[-1].timestamp) - float(self.samples[0].timestamp))

    @property
    def path_length(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return sum(
            self.samples[index - 1].position.distance_to(self.samples[index].position)
            for index in range(1, len(self.samples))
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "source": self.source,
            "duration": self.duration,
            "path_length": self.path_length,
            "metadata": dict(self.metadata),
            "samples": [
                {
                    "timestamp": float(sample.timestamp),
                    "position": sample.position.to_dict(),
                    "velocity": sample.velocity.to_dict(),
                    "acceleration": sample.acceleration.to_dict(),
                }
                for sample in self.samples
            ],
        }


@dataclass
class PlannerObservation:
    perception: PerceptionData = field(default_factory=PerceptionData)
    obstacles: List[ObstacleState] = field(default_factory=list)
    predicted_obstacles: List[PredictedObstacle] = field(default_factory=list)
    neighbour_states: List[VehicleState] = field(default_factory=list)
    neighbour_trajectories: Dict[int, Trajectory] = field(default_factory=dict)
    temporal_occupancy: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_perception(perception: PerceptionData) -> "PlannerObservation":
        return PlannerObservation(
            perception=perception,
            obstacles=[ObstacleState.from_obstacle(item) for item in perception.obstacles],
        )


def trajectory_from_points(
    positions: Iterable[Vec3],
    dt: float,
    start_time: float = 0.0,
    source: str = "",
) -> Trajectory:
    """Build a portable trajectory from position samples.

    This helper is intentionally conservative: velocities and accelerations are
    estimated by finite differences and remain zero when timing is invalid.
    """

    positions = list(positions)
    samples: List[TrajectorySample] = []
    safe_dt = max(0.0, float(dt))
    previous_velocity = Vec3()
    for index, position in enumerate(positions):
        velocity = Vec3()
        acceleration = Vec3()
        if index > 0 and safe_dt > 0.0:
            velocity = (position - positions[index - 1]) * (1.0 / safe_dt)
            acceleration = (velocity - previous_velocity) * (1.0 / safe_dt)
        samples.append(
            TrajectorySample(
                timestamp=float(start_time) + index * safe_dt,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
            )
        )
        previous_velocity = velocity
    return Trajectory(samples=samples, source=source)
