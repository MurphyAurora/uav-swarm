import numpy as np


CYLINDER_STATIC = "static_cylinder"
GROUND_DYNAMIC = "ground_dynamic"
FLYING_DYNAMIC = "flying_dynamic"


class DynamicObstacle:
    """Unified 3D obstacle model for motion-primitive validation.

    The same class covers static pillars, ground moving obstacles, and flying
    dynamic obstacles. Cylindrical obstacles use position as the base center
    [x, y, z_min], radius as horizontal radius, and height as vertical extent.
    Flying obstacles are treated as spheres centered at position.
    """

    def __init__(
        self,
        position,
        velocity=None,
        radius=0.5,
        height=3.0,
        motion_type=CYLINDER_STATIC,
        obstacle_id="",
    ):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(
            [0.0, 0.0, 0.0] if velocity is None else velocity,
            dtype=float,
        )
        self.radius = float(radius)
        self.height = float(height)
        self.motion_type = motion_type
        self.obstacle_id = obstacle_id

    @classmethod
    def static_pillar(cls, x, y, radius=0.5, height=5.0, obstacle_id=""):
        return cls(
            position=[x, y, 0.0],
            velocity=[0.0, 0.0, 0.0],
            radius=radius,
            height=height,
            motion_type=CYLINDER_STATIC,
            obstacle_id=obstacle_id,
        )

    @classmethod
    def ground_vehicle(cls, position, velocity, radius=0.6, height=1.8, obstacle_id=""):
        return cls(
            position=[position[0], position[1], 0.0],
            velocity=[velocity[0], velocity[1], 0.0],
            radius=radius,
            height=height,
            motion_type=GROUND_DYNAMIC,
            obstacle_id=obstacle_id,
        )

    @classmethod
    def flying_obstacle(cls, position, velocity, radius=0.5, obstacle_id=""):
        return cls(
            position=position,
            velocity=velocity,
            radius=radius,
            height=2.0 * radius,
            motion_type=FLYING_DYNAMIC,
            obstacle_id=obstacle_id,
        )

    @classmethod
    def from_2d_circle(cls, circle, height=5.0):
        return cls.static_pillar(
            x=circle.x,
            y=circle.y,
            radius=circle.radius,
            height=height,
            obstacle_id=getattr(circle, "obstacle_id", ""),
        )

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    @property
    def z(self):
        return self.position[2]

    @property
    def z_min(self):
        return self.position[2]

    @property
    def z_max(self):
        return self.position[2] + self.height

    @property
    def is_flying(self):
        return self.motion_type == FLYING_DYNAMIC

    @property
    def is_cylinder(self):
        return not self.is_flying

    def position_at(self, t):
        return self.position + self.velocity * float(t)

    def update(self, dt=0.1):
        self.position = self.position_at(dt)
        if self.is_cylinder:
            self.position[2] = 0.0
        return self.position
