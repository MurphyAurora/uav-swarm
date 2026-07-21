import numpy as np


class PrimitiveCost:
    """Multi-objective cost for 3D UAV motion primitive evaluation."""

    def __init__(
        self,
        goal_weight=1.0,
        collision_weight=80.0,
        altitude_weight=1.6,
        smooth_weight=1.2,
        energy_weight=1.0,
        vertical_motion_weight=5.0,
        transition_weight=10.0,
        top_clearance_weight=18.0,
        safe_distance=0.8,
        reference_height=3.0,
        hard_collision_penalty=1000.0,
        low_risk_height_weight=1.0,
        high_risk_height_weight=0.15,
        top_safety_margin=0.8,
        vertical_speed_limit=0.3,
        approach_speed_floor=0.5,
        max_climb_height=9.6,
        min_climb_height=0.5,
    ):
        self.goal_weight = float(goal_weight)
        self.collision_weight = float(collision_weight)
        self.altitude_weight = float(altitude_weight)
        self.smooth_weight = float(smooth_weight)
        self.energy_weight = float(energy_weight)
        self.vertical_motion_weight = float(vertical_motion_weight)
        self.transition_weight = float(transition_weight)
        self.top_clearance_weight = float(top_clearance_weight)
        self.safe_distance = float(safe_distance)
        self.reference_height = float(reference_height)
        self.hard_collision_penalty = float(hard_collision_penalty)
        self.low_risk_height_weight = float(low_risk_height_weight)
        self.high_risk_height_weight = float(high_risk_height_weight)
        self.top_safety_margin = float(top_safety_margin)
        self.vertical_speed_limit = float(vertical_speed_limit)
        self.approach_speed_floor = float(approach_speed_floor)
        self.max_climb_height = float(max_climb_height)
        self.min_climb_height = float(min_climb_height)

    def goal_cost(self, trajectory, goal):
        return float(np.linalg.norm(trajectory[-1] - goal))

    def _obs_position(self, obs, t):
        if hasattr(obs, "position_at"):
            return np.asarray(obs.position_at(t), dtype=float)
        return np.asarray(obs.position, dtype=float)

    def _horizontal_distance(self, point, obs, t):
        obs_pos = self._obs_position(obs, t)
        return float(np.linalg.norm(point[:2] - obs_pos[:2]))

    def _horizontal_clearance(self, point, obs, t):
        return self._horizontal_distance(point, obs, t) - float(obs.radius)

    def _clearance(self, point, obs, t):
        obs_pos = self._obs_position(obs, t)
        if getattr(obs, "is_flying", False):
            return float(np.linalg.norm(point - obs_pos) - obs.radius)

        horizontal = self._horizontal_distance(point, obs, t)
        if obs_pos[2] <= point[2] <= obs_pos[2] + obs.height:
            return float(horizontal - obs.radius)

        if point[2] < obs_pos[2]:
            vertical = obs_pos[2] - point[2]
        else:
            vertical = point[2] - (obs_pos[2] + obs.height)
        outside = np.hypot(max(0.0, horizontal - obs.radius), vertical)
        return float(outside)

    def minimum_clearance(self, trajectory, obstacles, times=None):
        if not obstacles:
            return float("inf")
        if times is None:
            times = np.zeros(len(trajectory))

        min_clearance = float("inf")
        for point, t in zip(trajectory, times):
            for obs in obstacles:
                min_clearance = min(min_clearance, self._clearance(point, obs, t))
        return float(min_clearance)

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
        return self.minimum_clearance(trajectory, obstacles, times) < 0.0

    def obstacle_risk(self, trajectory, obstacles, times=None):
        """Horizontal obstacle risk used only to modulate altitude stiffness."""
        if not obstacles:
            return 0.0
        if times is None:
            times = np.zeros(len(trajectory))

        risk = 0.0
        risk_distance = max(self.safe_distance, 1e-6)
        for point, t in zip(trajectory, times):
            for obs in obstacles:
                if getattr(obs, "is_flying", False):
                    horizontal_distance = self._horizontal_distance(point, obs, t)
                    normalized_distance = max(0.0, horizontal_distance - obs.radius)
                else:
                    normalized_distance = max(0.0, self._horizontal_clearance(point, obs, t))
                risk = max(risk, 1.0 - normalized_distance / risk_distance)
        return float(np.clip(risk, 0.0, 1.0))

    def altitude_deviation_cost(self, trajectory, obstacles=None, times=None):
        """Penalize deviation from z_ref and softly reject very high/low flight."""
        obstacles = obstacles or []
        if times is None:
            times = np.zeros(len(trajectory))

        cost = 0.0
        for point, t in zip(trajectory, times):
            point_risk = self.obstacle_risk(np.asarray([point]), obstacles, np.asarray([t]))
            height_weight = (
                self.low_risk_height_weight * (1.0 - point_risk)
                + self.high_risk_height_weight * point_risk
            )
            cost += height_weight * (point[2] - self.reference_height) ** 2

        over_ceiling = np.maximum(0.0, trajectory[:, 2] - self.max_climb_height)
        under_floor = np.maximum(0.0, self.min_climb_height - trajectory[:, 2])
        envelope_cost = 25.0 * np.mean(over_ceiling ** 2 + under_floor ** 2)
        return float(cost / max(1, len(trajectory)) + envelope_cost)

    def altitude_cost(self, trajectory, obstacles=None, times=None):
        return self.altitude_deviation_cost(trajectory, obstacles, times)

    def _top_required_height(self, obs, t):
        obs_pos = self._obs_position(obs, t)
        return float(obs_pos[2] + obs.height + self.top_safety_margin)

    def _inside_over_top_corridor(self, point, obs, t):
        obs_pos = self._obs_position(obs, t)
        lateral_distance = abs(point[1] - obs_pos[1])
        return lateral_distance <= obs.radius + self.safe_distance

    def top_clearance_cost(self, trajectory, obstacles, times=None):
        """Create an early climb ramp only for trajectories staying in the top corridor."""
        if not obstacles:
            return 0.0
        if times is None:
            times = np.zeros(len(trajectory))

        cost = 0.0
        for point, t in zip(trajectory, times):
            for obs in obstacles:
                if getattr(obs, "is_flying", False):
                    continue
                if not self._inside_over_top_corridor(point, obs, t):
                    continue

                required_z = self._top_required_height(obs, t)
                climb_height = max(0.0, required_z - self.reference_height)
                climb_distance = climb_height * self.approach_speed_floor / max(self.vertical_speed_limit, 1e-6)
                approach_distance = climb_distance + obs.radius + self.safe_distance

                horizontal_clearance = max(0.0, self._horizontal_clearance(point, obs, t))
                progress = np.clip(1.0 - horizontal_clearance / max(approach_distance, 1e-6), 0.0, 1.0)
                target_z = self.reference_height + progress * climb_height

                if horizontal_clearance <= self.safe_distance:
                    target_z = required_z

                top_error = max(0.0, target_z - point[2])
                cost += top_error * top_error
        return float(cost)

    def smooth_cost(self, trajectory):
        if len(trajectory) < 3:
            return 0.0
        velocity = np.diff(trajectory, axis=0)
        acceleration = np.diff(velocity, axis=0)
        return float(np.sum(acceleration ** 2))

    def energy_cost(self, velocity):
        velocity = np.asarray(velocity, dtype=float)
        vertical_effort = 3.0 * velocity[2] * velocity[2]
        return float(np.dot(velocity, velocity) + vertical_effort)

    def vertical_motion_cost(self, velocity, previous_velocity=None):
        velocity = np.asarray(velocity, dtype=float)
        cost = velocity[2] * velocity[2]
        if previous_velocity is not None:
            previous_velocity = np.asarray(previous_velocity, dtype=float)
            cost += 2.0 * (velocity[2] - previous_velocity[2]) ** 2
        return float(cost)

    def transition_cost(self, velocity, previous_velocity=None):
        if previous_velocity is None:
            return 0.0
        velocity = np.asarray(velocity, dtype=float)
        previous_velocity = np.asarray(previous_velocity, dtype=float)
        delta = velocity - previous_velocity
        return float(np.dot(delta, delta) + 2.0 * delta[2] * delta[2])

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
            "altitude_deviation_cost": self.altitude_deviation_cost(trajectory, obstacles, times),
            "smooth_cost": self.smooth_cost(trajectory),
            "energy_cost": self.energy_cost(velocity),
            "vertical_motion_cost": self.vertical_motion_cost(velocity, previous_velocity),
            "transition_cost": self.transition_cost(velocity, previous_velocity),
            "top_clearance_cost": self.top_clearance_cost(trajectory, obstacles, times),
        }
        terms["height_cost"] = terms["altitude_deviation_cost"]
        total = (
            self.goal_weight * terms["goal_cost"]
            + self.collision_weight * terms["collision_cost"]
            + self.altitude_weight * terms["altitude_deviation_cost"]
            + self.smooth_weight * terms["smooth_cost"]
            + self.energy_weight * terms["energy_cost"]
            + self.vertical_motion_weight * terms["vertical_motion_cost"]
            + self.transition_weight * terms["transition_cost"]
            + self.top_clearance_weight * terms["top_clearance_cost"]
        )
        terms["total_cost"] = float(total)
        terms["obstacle_risk"] = self.obstacle_risk(trajectory, obstacles, times)
        terms["minimum_clearance"] = self.minimum_clearance(trajectory, obstacles, times)
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
