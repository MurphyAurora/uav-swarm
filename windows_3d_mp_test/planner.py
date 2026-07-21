import numpy as np

from cost_function import PrimitiveCost
from primitive import generate_velocity_primitives


class MotionPrimitivePlanner:
    """Velocity-sampled 3D motion primitive selector for validation."""

    def __init__(
        self,
        min_altitude=0.5,
        max_altitude=9.8,
        vx_range=(0.5, 1.5),
        vy_range=(-0.5, 0.5),
        vz_range=(-0.3, 0.3),
        primitive_dt=0.2,
        xy_bounds=None,
        goal_slowdown_distance=3.0,
    ):
        self.cost = PrimitiveCost(max_climb_height=max_altitude - 0.2)
        self.min_altitude = float(min_altitude)
        self.max_altitude = float(max_altitude)
        self.vx_range = tuple(float(v) for v in vx_range)
        self.vy_range = tuple(float(v) for v in vy_range)
        self.vz_range = tuple(float(v) for v in vz_range)
        self.primitive_dt = float(primitive_dt)
        self.xy_bounds = xy_bounds
        self.goal_slowdown_distance = float(goal_slowdown_distance)
        self.previous_velocity = None
        self.last_debug = None

    def _goal_aware_vx_range(self, state, goal):
        distance_to_goal = float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(state, dtype=float)))
        if distance_to_goal > self.goal_slowdown_distance:
            return self.vx_range
        return (-0.5, min(1.0, self.vx_range[1]))

    def generate_primitives(self, state, goal=None, horizon=2.0):
        vx_range = self.vx_range if goal is None else self._goal_aware_vx_range(state, goal)
        return generate_velocity_primitives(
            state,
            vx_range=vx_range,
            vy_range=self.vy_range,
            vz_range=self.vz_range,
            horizon=horizon,
            dt=self.primitive_dt,
        )

    def _inside_altitude_limits(self, trajectory):
        return np.all(trajectory[:, 2] >= self.min_altitude) and np.all(
            trajectory[:, 2] <= self.max_altitude
        )

    def _bounds_tuple(self):
        bounds = self.xy_bounds
        if bounds is None:
            return None

        if isinstance(bounds, dict):
            return (
                bounds.get("x_min", bounds.get("xmin", -np.inf)),
                bounds.get("x_max", bounds.get("xmax", np.inf)),
                bounds.get("y_min", bounds.get("ymin", -np.inf)),
                bounds.get("y_max", bounds.get("ymax", np.inf)),
            )

        if all(hasattr(bounds, name) for name in ("xmin", "xmax", "ymin", "ymax")):
            return bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax

        return tuple(bounds)

    def _inside_xy_bounds(self, trajectory):
        bounds = self._bounds_tuple()
        if bounds is None:
            return True

        x_min, x_max, y_min, y_max = bounds
        return bool(
            np.all(trajectory[:, 0] >= float(x_min))
            and np.all(trajectory[:, 0] <= float(x_max))
            and np.all(trajectory[:, 1] >= float(y_min))
            and np.all(trajectory[:, 1] <= float(y_max))
        )

    def reset(self):
        self.previous_velocity = None
        self.last_debug = None

    def plan(self, state, goal, obstacles, predict_time=2.0, update_previous=True):
        candidates = self.generate_primitives(state, goal=goal, horizon=predict_time)

        best = None
        best_terms = None
        best_velocity = None

        for primitive in candidates:
            traj = primitive["trajectory"]
            times = primitive["times"]
            velocity = primitive["velocity"]

            if not self._inside_altitude_limits(traj):
                continue

            if not self._inside_xy_bounds(traj):
                continue

            if self.cost.has_hard_collision(traj, obstacles, times):
                continue

            terms = self.cost.evaluate(
                traj,
                np.array(goal, dtype=float),
                obstacles,
                times=times,
                velocity=velocity,
                previous_velocity=self.previous_velocity,
            )

            if best_terms is None or terms["total_cost"] < best_terms["total_cost"]:
                best = traj
                best_terms = terms
                best_velocity = velocity

        if best is None:
            self.last_debug = None
            return None, float("inf")

        if update_previous:
            self.previous_velocity = best_velocity.copy()

        self.last_debug = {
            "velocity": best_velocity.copy(),
            "cost_terms": dict(best_terms),
        }
        return best, best_terms["total_cost"]
