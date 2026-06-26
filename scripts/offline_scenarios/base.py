from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from xtd2_mission.planner_framework import Obstacle, PerceptionData, Vec3


@dataclass(frozen=True)
class CircleObstacle:
    """Static circular obstacle used by the offline 2D smoke simulator."""

    x: float
    y: float
    radius: float
    source: str = "static"
    obstacle_id: str = ""

    def position_at(self, _stamp: float) -> Tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class MovingCircleObstacle:
    """Constant-velocity circular obstacle for offline dynamic tests."""

    x0: float
    y0: float
    vx: float
    vy: float
    radius: float
    source: str = "dynamic"
    obstacle_id: str = ""
    prediction_horizon: float = 3.0
    prediction_dt: float = 0.5

    def position_at(self, stamp: float) -> Tuple[float, float]:
        return self.x0 + self.vx * stamp, self.y0 + self.vy * stamp

    def predictions(self, stamp: float, z: float) -> List[dict]:
        points: List[dict] = []
        steps = max(1, int(round(self.prediction_horizon / max(self.prediction_dt, 1.0e-6))))
        for idx in range(1, steps + 1):
            t = idx * self.prediction_dt
            x, y = self.position_at(stamp + t)
            points.append({"t": float(t), "x": float(x), "y": float(y), "z": float(z)})
        return points


@dataclass(frozen=True)
class Bounds2D:
    """Finite test-area boundary for offline planner benchmarks."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, pos: Vec3, margin: float = 0.0) -> bool:
        return (
            self.xmin + margin <= pos.x <= self.xmax - margin
            and self.ymin + margin <= pos.y <= self.ymax - margin
        )

    def signed_margin(self, pos: Vec3, margin: float = 0.0) -> float:
        return min(
            pos.x - (self.xmin + margin),
            (self.xmax - margin) - pos.x,
            pos.y - (self.ymin + margin),
            (self.ymax - margin) - pos.y,
        )


@dataclass(frozen=True)
class SimCase:
    """Complete offline scenario definition.

    This is intentionally planner-facing, not Gazebo-facing: the same case can
    later be exported to SDF/world, but offline tests use these analytic obstacle
    definitions directly.
    """

    name: str
    start: Vec3
    goal: Vec3
    obstacles: Tuple[CircleObstacle, ...] = field(default_factory=tuple)
    dynamic_obstacles: Tuple[MovingCircleObstacle, ...] = field(default_factory=tuple)
    bounds: Optional[Bounds2D] = None
    steps: int = 500
    dt: float = 0.2

    @property
    def all_obstacles(self) -> Tuple[object, ...]:
        return tuple(self.obstacles) + tuple(self.dynamic_obstacles)


def v3(x: float, y: float, z: float = -3.0) -> Vec3:
    return Vec3(float(x), float(y), float(z))


def default_bounds(length: float, half_width: float = 3.0, back: float = 0.6) -> Bounds2D:
    return Bounds2D(xmin=-back, xmax=float(length), ymin=-half_width, ymax=half_width)


def boundary_obstacles(case: SimCase, spacing: float = 0.35, radius: float = 0.16) -> Iterable[Obstacle]:
    """Convert test bounds into virtual static obstacles visible to the planner.

    The runner still judges out-of-bounds explicitly, but planner modules also
    need to see the boundary; otherwise they can plan an outside detour and only
    fail after execution.  The boundary is represented by small static disks
    sampled along each side of the rectangle.
    """

    if case.bounds is None:
        return []
    z = case.start.z
    b = case.bounds

    def samples(start: float, end: float) -> int:
        return max(2, int(math.ceil(abs(end - start) / max(spacing, 1.0e-6))) + 1)

    out: List[Obstacle] = []
    nx = samples(b.xmin, b.xmax)
    ny = samples(b.ymin, b.ymax)
    for idx in range(nx):
        alpha = idx / max(nx - 1, 1)
        x = b.xmin + alpha * (b.xmax - b.xmin)
        for y, side in ((b.ymin, "bottom"), (b.ymax, "top")):
            out.append(
                Obstacle(
                    position=Vec3(x, y, z),
                    radius=radius,
                    source="boundary",
                    obstacle_id=f"bound_{side}_{idx:03d}",
                )
            )
    for idx in range(ny):
        alpha = idx / max(ny - 1, 1)
        y = b.ymin + alpha * (b.ymax - b.ymin)
        for x, side in ((b.xmin, "left"), (b.xmax, "right")):
            out.append(
                Obstacle(
                    position=Vec3(x, y, z),
                    radius=radius,
                    source="boundary",
                    obstacle_id=f"bound_{side}_{idx:03d}",
                )
            )
    return out


def make_perception(case: SimCase, stamp: float) -> PerceptionData:
    obstacles: List[Obstacle] = []
    z = case.start.z

    for item in case.obstacles:
        obstacles.append(
            Obstacle(
                position=Vec3(item.x, item.y, z),
                radius=item.radius,
                source=item.source,
                obstacle_id=item.obstacle_id,
            )
        )

    for item in case.dynamic_obstacles:
        x, y = item.position_at(stamp)
        obstacles.append(
            Obstacle(
                position=Vec3(x, y, z),
                radius=item.radius,
                velocity=Vec3(item.vx, item.vy, 0.0),
                source=item.source,
                obstacle_id=item.obstacle_id,
                predictions=item.predictions(stamp, z),
            )
        )

    obstacles.extend(boundary_obstacles(case))

    return PerceptionData(
        obstacles=obstacles,
        nearest_distance=999.0,
        near_field_danger=False,
        frame_id="world",
        stamp=stamp,
    )


def obstacle_snapshots(case: SimCase, stamp: float) -> Iterable[Tuple[float, float, float, str]]:
    for item in case.obstacles:
        yield item.x, item.y, item.radius, item.obstacle_id
    for item in case.dynamic_obstacles:
        x, y = item.position_at(stamp)
        yield x, y, item.radius, item.obstacle_id


def collision_margin(pos: Vec3, case: SimCase, drone_radius: float, stamp: float) -> float:
    margin = 999.0
    for x, y, radius, _obs_id in obstacle_snapshots(case, stamp):
        dist = math.hypot(pos.x - x, pos.y - y)
        margin = min(margin, dist - (drone_radius + radius))
    return margin


def bounds_margin(pos: Vec3, case: SimCase, drone_radius: float) -> float:
    if case.bounds is None:
        return 999.0
    return case.bounds.signed_margin(pos, margin=drone_radius)
