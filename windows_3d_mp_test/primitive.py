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
    """Sample short kinodynamic primitives in velocity space.

    Each primitive is a dict containing the sampled constant velocity and the
    rollout trajectory. This keeps the architecture lightweight while avoiding
    the brittle forward/left/right/climb action set.
    """
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
    the planner uses generate_velocity_primitives for velocity-aware costs.
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
