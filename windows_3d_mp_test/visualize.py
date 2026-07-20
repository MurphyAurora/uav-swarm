import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _get_obstacle_value(obs, key):
    """Support both dict obstacles and DynamicObstacle objects."""
    if isinstance(obs, dict):
        return obs[key]
    return getattr(obs, key)


def plot_result(uav_path, obstacles, goal=None):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    path = list(zip(*uav_path))
    ax.plot(path[0], path[1], path[2], label='UAV trajectory')
    ax.scatter(path[0][0], path[1][0], path[2][0], label='start')

    if goal is not None:
        ax.scatter(goal[0], goal[1], goal[2], label='goal')

    for obs in obstacles:
        x = _get_obstacle_value(obs, 'x')
        y = _get_obstacle_value(obs, 'y')
        h = _get_obstacle_value(obs, 'height')
        r = _get_obstacle_value(obs, 'radius')

        verts = [[
            (x-r, y-r, 0), (x+r, y-r, 0),
            (x+r, y+r, 0), (x-r, y+r, 0)
        ], [
            (x-r, y-r, h), (x+r, y-r, h),
            (x+r, y+r, h), (x-r, y+r, h)
        ]]
        ax.add_collection3d(Poly3DCollection(verts))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
