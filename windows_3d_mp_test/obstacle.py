import numpy as np


class DynamicObstacle:
    """3D cylindrical dynamic obstacle model.

    position: [x, y, z] center of the obstacle
    velocity: [vx, vy, vz]
    radius: horizontal collision radius
    height: vertical height for visualization and collision checking
    """

    def __init__(self, position, velocity, radius=0.5, height=3.0):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = float(radius)
        self.height = float(height)

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    def update(self, dt=0.1):
        self.position += self.velocity * dt
        # smooth height disturbance for dynamic scenario
        self.position[2] += np.sin(self.position[0]) * 0.02
