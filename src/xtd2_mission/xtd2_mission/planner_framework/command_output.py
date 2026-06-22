#!/usr/bin/env python3
"""Command output smoothing and acceleration limiting."""

from typing import Optional

from .common import PlannerCommand, PlannerConfig, Vec3


class CommandOutput:
    def __init__(self, config: Optional[PlannerConfig] = None):
        self.config = config or PlannerConfig()
        self._last_velocity = Vec3()
        self._has_last = False

    def process(self, command: PlannerCommand, dt: Optional[float] = None) -> PlannerCommand:
        raw = command.velocity.limit_norm(self.config.max_speed)
        if not self._has_last:
            self._last_velocity = raw
            self._has_last = True
            return PlannerCommand(raw, command.mode, command.source_trajectory, command.reason)

        alpha = max(0.0, min(1.0, self.config.output_alpha))
        smoothed = raw * alpha + self._last_velocity * (1.0 - alpha)
        if dt is not None and dt > 0.0 and self.config.max_accel > 0.0:
            delta = (smoothed - self._last_velocity).limit_norm(self.config.max_accel * dt)
            smoothed = self._last_velocity + delta
        smoothed = smoothed.limit_norm(self.config.max_speed)
        self._last_velocity = smoothed
        return PlannerCommand(smoothed, command.mode, command.source_trajectory, command.reason)

    def reset(self) -> None:
        self._last_velocity = Vec3()
        self._has_last = False
