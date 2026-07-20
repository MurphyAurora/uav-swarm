import numpy as np


class PrimitiveCost:
    """Multi-objective cost for 3D UAV motion primitive evaluation."""

    def __init__(self,
                 goal_weight=1.0,
                 collision_weight=100.0,
                 altitude_weight=0.5,
                 smooth_weight=0.5,
                 energy_weight=2.0,
                 safe_distance=2.0,
                 reference_height=3.0):
        self.goal_weight = goal_weight
        self.collision_weight = collision_weight
        self.altitude_weight = altitude_weight
        self.smooth_weight = smooth_weight
        self.energy_weight = energy_weight
        self.safe_distance = safe_distance
        self.reference_height = reference_height

    def goal_cost(self, trajectory, goal):
        end = trajectory[-1]
        return np.linalg.norm(end - goal)

    def collision_cost(self, trajectory, obstacles):
        cost = 0.0
        for p in trajectory:
            for obs in obstacles:
                d = np.linalg.norm(p - obs.position)
                if d < self.safe_distance:
                    cost += (self.safe_distance - d) ** 2
        return cost

    def altitude_cost(self, trajectory):
        z = trajectory[:, 2]
        return np.sum((z - self.reference_height) ** 2)

    def smooth_cost(self, trajectory):
        if len(trajectory) < 3:
            return 0.0
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        return np.sum(acceleration ** 2)

    def energy_cost(self, trajectory):
        velocity = np.diff(trajectory, axis=0)
        return np.sum(velocity ** 2)

    def total_cost(self, trajectory, goal, obstacles):
        return (
            self.goal_weight * self.goal_cost(trajectory, goal)
            + self.collision_weight * self.collision_cost(trajectory, obstacles)
            + self.altitude_weight * self.altitude_cost(trajectory)
            + self.smooth_weight * self.smooth_cost(trajectory)
            + self.energy_weight * self.energy_cost(trajectory)
        )
