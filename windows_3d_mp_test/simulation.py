import numpy as np

from obstacle import DynamicObstacle
from planner import MotionPrimitivePlanner
from visualize import plot_result


if __name__ == '__main__':
    state = np.array([0.0, 0.0, 3.0])
    goal = np.array([20.0, 0.0, 3.0])

    obstacles = [
        DynamicObstacle(
            position=[10.0, 0.0, 0.0],
            velocity=[0.0, 0.05, 0.0],
            radius=1.0,
            height=5.0
        )
    ]

    planner = MotionPrimitivePlanner()

    history = [state.copy()]

    for step in range(300):
        best_traj, cost = planner.plan(state, goal, obstacles)

        if best_traj is None:
            print('no feasible trajectory')
            break

        state = best_traj[1]
        history.append(state.copy())

        for obs in obstacles:
            obs.update()

        print(
            'step:', step,
            'state:', state,
            'cost:', round(cost, 3)
        )

        if np.linalg.norm(state - goal) < 1:
            print('goal reached')
            break

    plot_result(
        np.array(history),
        obstacles,
        goal
    )
