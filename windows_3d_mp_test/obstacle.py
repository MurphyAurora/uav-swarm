import numpy as np


class DynamicObstacle:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def update(self, dt=0.1):
        self.position += self.velocity * dt
        self.position[2] += np.sin(self.position[0]) * 0.02
