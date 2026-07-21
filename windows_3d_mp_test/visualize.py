import numpy as np
import matplotlib.pyplot as plt


def _value(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _position(obs):
    if isinstance(obs, dict):
        return np.array(obs.get("position", [obs.get("x", 0.0), obs.get("y", 0.0), 0.0]))
    return np.asarray(obs.position, dtype=float)


def _draw_cylinder(ax, center, radius, height, color="tab:green", alpha=0.32):
    theta = np.linspace(0.0, 2.0 * np.pi, 40)
    z = np.linspace(center[2], center[2] + height, 12)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)

    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)

    disk_r = np.linspace(0.0, radius, 12)
    disk_theta, disk_radius = np.meshgrid(theta, disk_r)
    disk_x = center[0] + disk_radius * np.cos(disk_theta)
    disk_y = center[1] + disk_radius * np.sin(disk_theta)
    bottom_z = np.full_like(disk_x, center[2])
    top_z = np.full_like(disk_x, center[2] + height)

    ax.plot_surface(disk_x, disk_y, bottom_z, color=color, alpha=alpha, linewidth=0)
    ax.plot_surface(disk_x, disk_y, top_z, color=color, alpha=alpha, linewidth=0)


def _draw_sphere_marker(ax, center, radius, color="tab:red"):
    ax.scatter(center[0], center[1], center[2], s=max(30.0, radius * 160.0), c=color)


def _plot_obstacles(ax, obstacles, future_horizon=3.0, future_steps=8):
    for obs in obstacles:
        center = _position(obs)
        radius = float(_value(obs, "radius", 0.5))
        height = float(_value(obs, "height", 2.0))
        is_flying = bool(_value(obs, "is_flying", False))

        if is_flying:
            _draw_sphere_marker(ax, center, radius)
        else:
            _draw_cylinder(ax, center, radius, height)

        if hasattr(obs, "position_at") and np.linalg.norm(obs.velocity) > 0.0:
            ts = np.linspace(0.0, future_horizon, future_steps)
            future = np.array([obs.position_at(t) for t in ts])
            ax.plot(
                future[:, 0],
                future[:, 1],
                future[:, 2],
                linestyle="--",
                color="tab:red",
                alpha=0.8,
            )


def _set_axes(ax, title=None):
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1.8, 0.8, 0.7))
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()


def plot_scene(start, goal, obstacles, title="3D obstacle scene"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    ax.scatter(start[0], start[1], start[2], label="start", c="tab:blue")
    ax.scatter(goal[0], goal[1], goal[2], label="goal", c="tab:orange")
    ax.plot([start[0], goal[0]], [start[1], goal[1]], [start[2], goal[2]], linestyle="--", color="0.45", alpha=0.6, label="start-goal line")

    _plot_obstacles(ax, obstacles)
    _set_axes(ax, title=title)
    plt.show()


def plot_result(uav_path, obstacles, goal=None, future_horizon=3.0, future_steps=8):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    uav_path = np.asarray(uav_path, dtype=float)
    ax.plot(uav_path[:, 0], uav_path[:, 1], uav_path[:, 2], label="UAV trajectory")
    ax.scatter(uav_path[0, 0], uav_path[0, 1], uav_path[0, 2], label="start")

    if goal is not None:
        ax.scatter(goal[0], goal[1], goal[2], label="goal")

    _plot_obstacles(ax, obstacles, future_horizon=future_horizon, future_steps=future_steps)
    _set_axes(ax)
    plt.show()
