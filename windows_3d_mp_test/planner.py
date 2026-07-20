import numpy as np

from cost_function import PrimitiveCost


class MotionPrimitivePlanner:
    """Simple 3D motion primitive selector for algorithm validation."""

    def __init__(self, min_altitude=0.5, max_altitude=8.0):
        self.cost = PrimitiveCost()
        self.min_altitude = float(min_altitude)
        self.max_altitude = float(max_altitude)

    def generate_primitives(self, state, speed=1.0, horizon=2.0, steps=20):
        directions = [
            (1, 0, 0),
            (1, 0.5, 0),
            (1, -0.5, 0),
            (0.8, 0, 0.6),
            (0.8, 0, -0.6),
            (1, 0.5, 0.5),
            (1, -0.5, 0.5),
            (1, 0.5, -0.5),
            (1, -0.5, -0.5),
            (0, 0, 1),
            (0, 0, -1),
        ]

        trajectories = []
        times = np.linspace(0.0, horizon, steps)

        for direction in directions:
            velocity = np.array(direction, dtype=float)
            velocity = velocity / np.linalg.norm(velocity) * speed
            traj = np.array([state + velocity * t for t in times])
            trajectories.append(traj)

        return trajectories, times

    def _inside_altitude_limits(self, trajectory):
        return np.all(trajectory[:, 2] >= self.min_altitude) and np.all(
            trajectory[:, 2] <= self.max_altitude
        )

    def plan(self, state, goal, obstacles, predict_time=2.0):
        candidates, times = self.generate_primitives(state, horizon=predict_time)

        best = None
        best_cost = float("inf")

        for traj in candidates:
            if not self._inside_altitude_limits(traj):
                continue

            if self.cost.has_hard_collision(traj, obstacles, times):
                continue

            cost = self.cost.total_cost(
                traj,
                np.array(goal, dtype=float),
                obstacles,
                times,
            )

            if cost < best_cost:
                best_cost = cost
                best = traj

        return best, best_cost
