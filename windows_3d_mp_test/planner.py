import numpy as np

from cost_function import PrimitiveCost


class MotionPrimitivePlanner:
    """Simple 3D motion primitive selector for algorithm validation."""

    def __init__(self):
        self.cost = PrimitiveCost()

    def generate_primitives(self, state, speed=1.0, horizon=2.0, steps=20):
        directions = [
            (1, 0, 0),
            (1, 0.5, 0),
            (1, -0.5, 0),
            (1, 0, 0.5),
            (1, 0, -0.5),
            (1, 0.5, 0.5),
            (1, -0.5, 0.5),
        ]

        trajectories = []
        times = np.linspace(0, horizon, steps)

        for direction in directions:
            v = np.array(direction, dtype=float)
            v = v / np.linalg.norm(v) * speed
            traj = np.array([state + v * t for t in times])
            trajectories.append(traj)

        return trajectories

    def plan(self, state, goal, obstacles):
        candidates = self.generate_primitives(state)

        best = None
        best_cost = float('inf')

        for traj in candidates:
            cost = self.cost.total_cost(
                traj,
                np.array(goal),
                obstacles
            )

            if cost < best_cost:
                best_cost = cost
                best = traj

        return best, best_cost
