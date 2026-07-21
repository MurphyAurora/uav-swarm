import numpy as np


def _sample_axis(min_value, max_value, count):
    if count <= 1:
        return np.array([(float(min_value) + float(max_value)) * 0.5], dtype=float)
    return np.linspace(float(min_value), float(max_value), int(count))


def rollout_constant_velocity(position, velocity, horizon=2.0, dt=0.2):
    """Integrate one constant-velocity primitive with first-order kinematics."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    dt = max(1e-3, float(dt))
    horizon = max(dt, float(horizon))
    steps = int(np.floor(horizon / dt + 1e-9)) + 1
    times = np.arange(steps, dtype=float) * dt
    return np.array([position + velocity * t for t in times]), times


def _goal_frame(position, goal):
    position = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    delta_xy = goal[:2] - position[:2]
    distance_xy = float(np.linalg.norm(delta_xy))
    if distance_xy < 1e-6:
        forward = np.array([1.0, 0.0], dtype=float)
    else:
        forward = delta_xy / distance_xy
    lateral = np.array([-forward[1], forward[0]], dtype=float)
    return forward, lateral


def generate_goal_directed_velocity_primitives(
    position,
    goal,
    forward_speed_range=(0.3, 1.4),
    lateral_speed_range=(-0.45, 0.45),
    vz_range=(-0.3, 0.3),
    forward_samples=5,
    lateral_samples=7,
    vz_samples=5,
    horizon=2.0,
    dt=0.2,
):
    """Sample primitives in a local frame aligned with the goal direction.

    The planner still evaluates short constant-velocity rollouts, but the
    sampled horizontal velocities are forward-to-goal plus lateral bypass. This
    prevents world-frame sampling from drifting far past the goal after a
    sidestep around an obstacle.
    """
    primitives = []
    forward_axis, lateral_axis = _goal_frame(position, goal)
    forward_values = _sample_axis(forward_speed_range[0], forward_speed_range[1], forward_samples)
    lateral_values = _sample_axis(lateral_speed_range[0], lateral_speed_range[1], lateral_samples)
    vz_values = _sample_axis(vz_range[0], vz_range[1], vz_samples)

    for forward_speed in forward_values:
        for lateral_speed in lateral_values:
            horizontal_velocity = forward_speed * forward_axis + lateral_speed * lateral_axis
            for vz in vz_values:
                velocity = np.array(
                    [horizontal_velocity[0], horizontal_velocity[1], vz],
                    dtype=float,
                )
                trajectory, times = rollout_constant_velocity(
                    position,
                    velocity,
                    horizon=horizon,
                    dt=dt,
                )
                primitives.append(
                    {
                        "velocity": velocity,
                        "trajectory": trajectory,
                        "times": times,
                    }
                )

    return primitives


def generate_velocity_primitives(
    position,
    vx_range=(0.5, 1.5),
    vy_range=(-0.5, 0.5),
    vz_range=(-0.3, 0.3),
    vx_samples=5,
    vy_samples=7,
    vz_samples=5,
    horizon=2.0,
    dt=0.2,
):
    """Sample short kinodynamic primitives in world velocity space."""
    primitives = []
    vx_values = _sample_axis(vx_range[0], vx_range[1], vx_samples)
    vy_values = _sample_axis(vy_range[0], vy_range[1], vy_samples)
    vz_values = _sample_axis(vz_range[0], vz_range[1], vz_samples)

    for vx in vx_values:
        for vy in vy_values:
            for vz in vz_values:
                velocity = np.array([vx, vy, vz], dtype=float)
                trajectory, times = rollout_constant_velocity(
                    position,
                    velocity,
                    horizon=horizon,
                    dt=dt,
                )
                primitives.append(
                    {
                        "velocity": velocity,
                        "trajectory": trajectory,
                        "times": times,
                    }
                )

    return primitives


def generate_primitives(position, speed=1.0, horizon=1.0, steps=20):
    """Compatibility wrapper returning only trajectory arrays.

    Older scripts imported generate_primitives directly. Keep that surface while
    the planner uses velocity-aware costs.
    """
    dt = float(horizon) / max(1, int(steps) - 1)
    primitives = generate_velocity_primitives(
        position,
        vx_range=(0.5 * speed, 1.5 * speed),
        vy_range=(-0.5 * speed, 0.5 * speed),
        vz_range=(-0.3 * speed, 0.3 * speed),
        horizon=horizon,
        dt=dt,
    )
    return [item["trajectory"] for item in primitives]
