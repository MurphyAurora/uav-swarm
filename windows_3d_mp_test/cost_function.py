import numpy as np


class PrimitiveCost:
    """Multi-objective cost for 3D UAV motion primitive evaluation."""

    def __init__(
        self,
        goal_weight=1.0,
        collision_weight=100.0,
        altitude_weight=0.5,
        smooth_weight=0.5,
        energy_weight=2.0,
        risk_weight=5.0,
        safe_distance=0.8,
        reference_height=3.0,
        hard_collision_penalty=1000.0,
    ):
        self.goal_weight = goal_weight
        self.collision_weight = collision_weight
        self.altitude_weight = altitude_weight
        self.smooth_weight = smooth_weight
        self.energy_weight = energy_weight
        self.risk_weight = risk_weight
        self.safe_distance = safe_distance
        self.reference_height = reference_height
        self.hard_collision_penalty = hard_collision_penalty

    def goal_cost(self, trajectory, goal):
        return np.linalg.norm(trajectory[-1] - goal)

    def _obs_position(self, obs, t):
        if hasattr(obs, "position_at"):
            return np.asarray(obs.position_at(t), dtype=float)
        return np.asarray(obs.position, dtype=float)

    def _cylinder_penalty(self, point, obs, t):
        obs_pos = self._obs_position(obs, t)
        horizontal = np.linalg.norm(point[:2] - obs_pos[:2])
        inflated_radius = obs.radius + self.safe_distance
        z_min = obs_pos[2] - self.safe_distance
        z_max = obs_pos[2] + obs.height + self.safe_distance

        if not (z_min <= point[2] <= z_max):
            return 0.0

        if horizontal >= inflated_radius:
            return 0.0

        margin = inflated_radius - horizontal
        penalty = margin ** 2

        if horizontal < obs.radius and obs_pos[2] <= point[2] <= obs_pos[2] + obs.height:
            penalty += self.hard_collision_penalty

        return penalty

    def _sphere_penalty(self, point, obs, t):
        obs_pos = self._obs_position(obs, t)
        distance = np.linalg.norm(point - obs_pos)
        inflated_radius = obs.radius + self.safe_distance

        if distance >= inflated_radius:
            return 0.0

        margin = inflated_radius - distance
        penalty = margin ** 2
        if distance < obs.radius:
            penalty += self.hard_collision_penalty
        return penalty

    def collision_cost(self, trajectory, obstacles, times=None):
        if times is None:
            times = np.zeros(len(trajectory))

        cost = 0.0
        for point, t in zip(trajectory, times):
            for obs in obstacles:
                if getattr(obs, "is_flying", False):
                    cost += self._sphere_penalty(point, obs, t)
                else:
                    cost += self._cylinder_penalty(point, obs, t)
        return cost

    def has_hard_collision(self, trajectory, obstacles, times=None):
        if times is None:
            times = np.zeros(len(trajectory))

        for point, t in zip(trajectory, times):
            for obs in obstacles:
                obs_pos = self._obs_position(obs, t)
                if getattr(obs, "is_flying", False):
                    if np.linalg.norm(point - obs_pos) < obs.radius:
                        return True
                else:
                    horizontal = np.linalg.norm(point[:2] - obs_pos[:2])
                    inside_height = obs_pos[2] <= point[2] <= obs_pos[2] + obs.height
                    if horizontal < obs.radius and inside_height:
                        return True
        return False

    def altitude_cost(self, trajectory):
        return np.sum((trajectory[:, 2] - self.reference_height) ** 2)

    def smooth_cost(self, trajectory):
        if len(trajectory) < 3:
            return 0.0
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        return np.sum(acceleration ** 2)

    def energy_cost(self, trajectory):
        velocity = np.diff(trajectory, axis=0)
        return np.sum(velocity ** 2)

    def risk_cost(self, trajectory, obstacles, times=None):
        return self.collision_cost(trajectory, obstacles, times)

    def total_cost(self, trajectory, goal, obstacles, times=None):
        return (
            self.goal_weight * self.goal_cost(trajectory, goal)
            + self.collision_weight * self.collision_cost(trajectory, obstacles, times)
            + self.altitude_weight * self.altitude_cost(trajectory)
            + self.smooth_weight * self.smooth_cost(trajectory)
            + self.energy_weight * self.energy_cost(trajectory)
            + self.risk_weight * self.risk_cost(trajectory, obstacles, times)
        )
