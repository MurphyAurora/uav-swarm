#!/usr/bin/env python3
"""Common data models for the EGO-like planner framework."""

from dataclasses import dataclass, field
import math
from typing import Dict, Iterable, List, Optional, Tuple


_EPS = 1.0e-9


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale: float) -> "Vec3":
        return Vec3(self.x * scale, self.y * scale, self.z * scale)

    __rmul__ = __mul__

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self, fallback: Optional["Vec3"] = None) -> "Vec3":
        mag = self.norm()
        if mag < _EPS:
            return fallback if fallback is not None else Vec3()
        return Vec3(self.x / mag, self.y / mag, self.z / mag)

    def limit_norm(self, max_norm: float) -> "Vec3":
        mag = self.norm()
        if max_norm <= 0.0:
            return Vec3()
        if mag <= max_norm or mag < _EPS:
            return self
        return self * (max_norm / mag)

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).norm()

    def to_dict(self) -> Dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "z": float(self.z)}

    @staticmethod
    def from_mapping(data: Dict, default: Optional["Vec3"] = None, prefer_world: bool = False) -> "Vec3":
        default = default or Vec3()
        if prefer_world:
            x = data.get("world_x", data.get("x", default.x))
            y = data.get("world_y", data.get("y", default.y))
        else:
            x = data.get("x", data.get("world_x", default.x))
            y = data.get("y", data.get("world_y", default.y))
        return Vec3(
            float(x),
            float(y),
            float(data.get("z", default.z)),
        )


@dataclass
class PlannerState:
    drone_id: int
    position: Vec3
    velocity: Vec3 = field(default_factory=Vec3)
    stamp: float = 0.0

    @staticmethod
    def from_mapping(data: Dict, prefer_world: bool = False) -> "PlannerState":
        return PlannerState(
            drone_id=int(data.get("id", data.get("drone_id", 0))),
            position=Vec3.from_mapping(data, prefer_world=prefer_world),
            velocity=Vec3(
                float(data.get("vx", 0.0)),
                float(data.get("vy", 0.0)),
                float(data.get("vz", 0.0)),
            ),
            stamp=float(data.get("stamp", 0.0)),
        )


@dataclass
class Obstacle:
    position: Vec3
    radius: float = 0.3
    velocity: Vec3 = field(default_factory=Vec3)
    source: str = "obstacle"
    obstacle_id: str = ""
    predictions: List[Tuple[float, Vec3]] = field(default_factory=list)

    def position_at(self, t: float) -> Vec3:
        if self.predictions:
            return nearest_prediction(self.predictions, t)
        return self.position + self.velocity * max(0.0, float(t))


@dataclass
class TrajectoryPoint:
    t: float
    position: Vec3

    def to_dict(self) -> Dict[str, float]:
        return {"t": float(self.t), **self.position.to_dict()}


@dataclass
class CandidateTrajectory:
    name: str
    velocity: Vec3
    points: List[TrajectoryPoint]
    metadata: Dict = field(default_factory=dict)

    def final_position(self) -> Vec3:
        return self.points[-1].position if self.points else Vec3()

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "velocity": self.velocity.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "metadata": dict(self.metadata),
        }


@dataclass
class SafetyReport:
    safe: bool
    feasible: bool
    min_clearance: float = 999.0
    min_clearance_source: str = "none"
    min_static_clearance: float = 999.0
    min_ttc: float = 999.0
    reason: str = "ok"


@dataclass
class CandidateEvaluation:
    trajectory: CandidateTrajectory
    safety: SafetyReport
    score: float
    costs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.trajectory.name,
            "safe": bool(self.safety.safe),
            "feasible": bool(self.safety.feasible),
            "score": float(self.score),
            "costs": dict(self.costs),
            "min_clearance": float(self.safety.min_clearance),
            "min_clearance_source": self.safety.min_clearance_source,
            "min_static_clearance": float(self.safety.min_static_clearance),
            "min_ttc": float(self.safety.min_ttc),
            "reason": self.safety.reason,
            "velocity": self.trajectory.velocity.to_dict(),
        }


@dataclass
class PlannerCommand:
    velocity: Vec3
    mode: str = "planner"
    source_trajectory: str = ""
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "mode": self.mode,
            "source_trajectory": self.source_trajectory,
            "reason": self.reason,
            "velocity": self.velocity.to_dict(),
        }


@dataclass
class PerceptionData:
    obstacles: List[Obstacle] = field(default_factory=list)
    nearest_distance: float = 999.0
    near_field_danger: bool = False
    frame_id: str = "world"
    stamp: float = 0.0


@dataclass
class PlannerConfig:
    horizon: float = 2.5
    dt: float = 0.5
    cruise_speed: float = 0.7
    lateral_speed: float = 0.45
    vertical_speed: float = 0.35
    max_speed: float = 1.0
    max_accel: float = 0.8
    drone_radius: float = 0.35
    obstacle_margin: float = 0.45
    hard_clearance: float = 0.25
    emergency_clearance: float = 0.8
    static_hard_clearance: float = 0.6
    static_emergency_clearance: float = 1.2
    local_goal_distance: float = 3.0
    local_goal_reached_radius: float = 0.8
    goal_weight: float = 1.0
    clearance_weight: float = 8.0
    ttc_weight: float = 4.0
    smooth_weight: float = 0.4
    risk_radius: float = 5.0
    output_alpha: float = 0.55


def nearest_prediction(predictions: Iterable[Tuple[float, Vec3]], t: float) -> Vec3:
    best_err = None
    best_pos = None
    for pred_t, pred_pos in predictions:
        err = abs(float(pred_t) - float(t))
        if best_err is None or err < best_err:
            best_err = err
            best_pos = pred_pos
    return best_pos if best_pos is not None else Vec3()


def safe_div(num: float, den: float, fallback: float = 0.0) -> float:
    if abs(den) < _EPS:
        return fallback
    return num / den
