import numpy as np


class PrimitiveCost:
    """Multi-objective cost for 3D UAV motion primitive evaluation."""

    def __init__(
        self,
        goal_weight=1.0,
        collision_weight=80.0,
        altitude_weight=0.8,
        smooth_weight=1.2,
        energy_weight=0.8,
        transition_weight=6.0,
        risk_weight=10.0,
        safe_distance=0.8,
        reference_height=3.0,
        hard_collision_penalty=1000.0,
        risk_threshold=0.25,
        max_climb_height=8.8,
    ):
        self.goal_weight = float(goal_weight)
        self.collision_weight = float(collision_weight)
        self.altitude_weight = float(altitude_weight)
        self.smooth_weight = float(smooth_weight)
        self.energy_weight = float(energy_weight)
        self.transition_weight = float(transition_weight)
        self.risk_weight = float(risk_weight)
        self.safe_distance = float(safe_distance)
        self.reference_height = float(reference_height)
        self.hard_collision_penalty = float(hard_collision_penalty)
        self.risk_threshold = float(risk_threshold)
        self.max_climb_height = float(max_climb_height)

    def goal_cost(self, trajectory, goal):
        return float(np.linalg.norm(trajectory[-1] - goal))

    def _obs_position(self, obs, t):
        if hasattr(obs, "position_at"):
            return np.asarray(obs.position_at(t), dtype=float)
        return np.asarray(obs.position, dtype=float)

    def _clearance(self, point, obs, t):
        obs_pos = self._obs_position(obs, t)
        if getattr(obs, "is_flying", False):
            return float(np.linalg.norm(point - obs_pos) - obs.radius)

        horizontal = np.linalg.norm(point[:2] - obs_pos[:2])
        if obs_pos[2] <= point[2] <= obs_pos[2] + obs.height:
            return float(horizontal - obs.radius)

        if point[2] < obs_pos[2]:
            vertical = obs_pos[2] - point[2]
        else:
            vertical = point[2] - (obs_pos[2] + obs.height)
        outside = np.hypot(max(0.0, horizontal - obs.radius), vertical)
        return float(outside)

    def _point_collision_penalty(self, point, obs, t):
        clearance = self._clearance(point, obs, t)
        inflated_clearance = clearance - self.safe_distance
        if inflated_clearance >= 0.0:
            return 0.0

        margin = -inflated_clearance
        penalty = margin * margin
        if clearance < 0.0:
            penalty += self.hard_collision_penalty
        return float(penalty)

    def collision_cost(self, trajectory, obstacles, times=None):
        if times is None:
            times = np.zeros(len(trajectory))

        cost = 0.0
        for point, t in zip(trajectory, times):
            for obs in obstacles:
                cost += self._point_collision_penalty(point, obs, t)
        return float(cost)

    def has_hard_collision(self, trajectory, obstacles, times=None):
        if times is None:
            times = np.zeros(len(trajectory))

        for point, t in zip(trajectory, times):
            for obs in obstacles:
                if self._clearance(point, obs, t) < 0.0:
                    return True
        return False

    def obstacle_risk(self, trajectory, obstacles, times=None):
        if not obstacles:
            return 0.0
        if times is None:
            times = np.zeros(len(trajectory))

        risk = 0.0
        risk_radius = max(self.safe_distance * 4.0, 1e-6)
        for point, t in zip(trajectory, times):
            for obs in obstacles:
                clearance = self._clearance(point, obs, t)
                if clearance < 0.0:
                    return 1.0
                if clearance < risk_radius:
                    risk = max(risk, 1.0 - clearance / risk_radius)
        return float(np.clip(risk, 0.0, 1.0))

    def _point_altitude_target(self, point, obstacles, t):
        target = self.reference_height
        risk = 0.0
        risk_radius = max(self.safe_distance * 4.0, 1e-6)

        for obs in obstacles:
            obs_pos = self._obs_position(obs, t)
            clearance = self._clearance(point, obs, t)
            if clearance < risk_radius:
                obs_risk = 1.0 if clearance < 0.0 else 1.0 - clearance / risk_radius
                risk = max(risk, obs_risk)

                if not getattr(obs, "is_flying", False):
                    top_clearance_height = obs_pos[2] + obs.height + self.safe_distance
                    if top_clearance_height <= self.max_climb_height:
                        target = max(target, top_clearance_height)

        if risk < self.risk_threshold:
            return self.reference_height, risk

        blend = np.clip((risk - self.risk_threshold) / (1.0 - self.risk_threshold), 0.0, 1.0)
        target = (1.0 - blend) * self.reference_height + blend * target
        return float(target), float(risk)

    def altitude_cost(self, trajectory, obstacles=None, times=None):
        """Keep z near reference when safe and raise the target near climbable risk.

        Safe flight is attracted to reference_height. Near a cylinder that can be
        cleared vertically, the reference-height term is relaxed and a blended
        target is lifted toward obstacle_top + safe_distance. Once the rollout is
        away from risk, the target returns to reference_height.
        """
        obstacles = obstacles or []
        if times is None:
            times = np.zeros(len(trajectory))

        cost = 0.0
        for point, t in zip(trajectory, times):
            target, risk = self._point_altitude_target(point, obstacles, t)
            keep_reference_weight = 1.0 if risk < self.risk_threshold else max(0.15, 1.0 - 0.85 * risk)
            climb_target_weight = max(0.0, risk - self.risk_threshold)

            reference_error = (point[2] - self.reference_height) ** 2
            target_error = (point[2] - target) ** 2
            cost += keep_reference_weight * reference_error + climb_target_weight * target_error

        over_ceiling = np.maximum(0.0, trajectory[:, 2] - self.max_climb_height)
        ceiling_cost = np.mean(over_ceiling ** 2) * 25.0
        return float(cost / max(1, len(trajectory)) + ceiling_cost)

    def smooth_cost(self, trajectory):
        if len(trajectory) < 3:
            return 0.0
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        return float(np.sum(acceleration ** 2))

    def energy_cost(self, velocity):
        velocity = np.asarray(velocity, dtype=float)
        vertical_effort = 2.0 * velocity[2] * velocity[2]
        return float(np.dot(velocity, velocity) + vertical_effort)

    def transition_cost(self, velocity, previous_velocity=None):
        if previous_velocity is None:
            return 0.0
        velocity = np.asarray(velocity, dtype=float)
        previous_velocity = np.asarray(previous_velocity, dtype=float)
        return float(np.sum((velocity - previous_velocity) ** 2))

    def risk_cost(self, trajectory, obstacles, times=None):
        risk = self.obstacle_risk(trajectory, obstacles, times)
        return float(risk * risk)

    def evaluate(self, trajectory, goal, obstacles, times=None, velocity=None, previous_velocity=None):
        if velocity is None:
            if len(trajectory) >= 2:
                dt = 1.0
                if times is not None and len(times) >= 2:
                    dt = max(1e-6, float(times[1] - times[0]))
                velocity = (trajectory[1] - trajectory[0]) / dt
            else:
                velocity = np.zeros(3)

        terms = {
            "goal_cost": self.goal_cost(trajectory, goal),
            "collision_cost": self.collision_cost(trajectory, obstacles, times),
            "height_cost": self.altitude_cost(trajectory, obstacles, times),
            "smooth_cost": self.smooth_cost(trajectory),
            "energy_cost": self.energy_cost(velocity),
            "transition_cost": self.transition_cost(velocity, previous_velocity),
            "risk_cost": self.risk_cost(trajectory, obstacles, times),
        }
        total = (
            self.goal_weight * terms["goal_cost"]
            + self.collision_weight * terms["collision_cost"]
            + self.altitude_weight * terms["height_cost"]
            + self.smooth_weight * terms["smooth_cost"]
            + self.energy_weight * terms["energy_cost"]
            + self.transition_weight * terms["transition_cost"]
            + self.risk_weight * terms["risk_cost"]
        )
        terms["total_cost"] = float(total)
        terms["obstacle_risk"] = self.obstacle_risk(trajectory, obstacles, times)
        return terms

    def total_cost(self, trajectory, goal, obstacles, times=None, velocity=None, previous_velocity=None):
        return self.evaluate(
            trajectory,
            goal,
            obstacles,
            times=times,
            velocity=velocity,
            previous_velocity=previous_velocity,
        )["total_cost"]
