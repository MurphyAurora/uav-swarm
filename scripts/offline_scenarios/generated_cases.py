from __future__ import annotations

import random
from typing import List

from .base import CircleObstacle, SimCase, v3


def random_forest(seed: int = 0, n: int = 18, length: float = 16.0, width: float = 6.0, min_start_clear: float = 1.6) -> SimCase:
    """Generate a reproducible 2D cylinder forest for offline planner evaluation."""

    rng = random.Random(seed)
    obstacles: List[CircleObstacle] = []
    attempts = 0
    while len(obstacles) < n and attempts < n * 80:
        attempts += 1
        x = rng.uniform(2.0, length - 2.0)
        y = rng.uniform(-width * 0.5, width * 0.5)
        radius = rng.uniform(0.25, 0.55)
        if x < min_start_clear and abs(y) < 1.2:
            continue
        # Keep a loose minimum spacing so the generated forest normally contains
        # feasible gaps instead of a fully blocked wall.
        ok = True
        for obs in obstacles:
            min_sep = radius + obs.radius + 0.35
            if (x - obs.x) ** 2 + (y - obs.y) ** 2 < min_sep ** 2:
                ok = False
                break
        if not ok:
            continue
        obstacles.append(CircleObstacle(x, y, radius, obstacle_id=f"tree_{len(obstacles):02d}"))

    return SimCase(
        name=f"random_forest_seed_{seed}_n{len(obstacles)}",
        start=v3(0.0, 0.0),
        goal=v3(length, 0.0),
        obstacles=tuple(obstacles),
        steps=800,
    )


def random_forest_sparse() -> SimCase:
    return random_forest(seed=0, n=12, width=6.5)


def random_forest_medium() -> SimCase:
    return random_forest(seed=1, n=20, width=6.0)


def random_forest_dense() -> SimCase:
    return random_forest(seed=2, n=30, width=6.0)
