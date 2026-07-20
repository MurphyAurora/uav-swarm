import numpy as np

from primitive import generate_primitives
from obstacle import DynamicObstacle
from cost_function import PrimitiveCost


if __name__ == '__main__':
    state = np.array([0.0, 0.0, 3.0])
    goal = np.array([20.0, 0.0, 3.0])

    obstacles = [DynamicObstacle([10, 0, 3], [0, 0.1, 0])]
    cost = PrimitiveCost()

    for step in range(100):
        primitives = generate_primitives(state)

        scores = [cost.total_cost(p, goal, obstacles) for p in primitives]
        best = primitives[int(np.argmin(scores))]

        state = best[1]
        for obs in obstacles:
            obs.update()

        print('step:', step, 'state:', state, 'cost:', min(scores))

        if np.linalg.norm(state-goal) < 1:
            print('goal reached')
            break
