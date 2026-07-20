import numpy as np


def generate_primitives(position, speed=1.0, horizon=1.0, steps=20):
    actions = [
        (1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 0, 0.5),
        (1, 0, -0.5),
    ]

    primitives = []
    t = np.linspace(0, horizon, steps)

    for a in actions:
        v = np.array(a, dtype=float)
        v = v / np.linalg.norm(v) * speed
        traj = np.array([position + v * ti for ti in t])
        primitives.append(traj)

    return primitives
