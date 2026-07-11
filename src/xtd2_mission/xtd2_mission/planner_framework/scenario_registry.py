"""Scenario registry for planner benchmark experiments.

Keeps scenario selection independent from the benchmark runner so new
Gazebo-inspired, offline, or paper baseline scenarios can be added without
changing experiment code.
"""

from dataclasses import dataclass
from typing import Callable, Dict


@dataclass(frozen=True)
class ScenarioSpec:
    """Named scenario factory entry."""

    name: str
    factory: Callable[[], object]
    description: str = ""


class ScenarioRegistry:
    """Registry used by offline benchmark and future baseline experiments."""

    def __init__(self) -> None:
        self._scenarios: Dict[str, ScenarioSpec] = {}

    def register(self, spec: ScenarioSpec) -> None:
        self._scenarios[spec.name] = spec

    def create(self, name: str):
        if name not in self._scenarios:
            available = ", ".join(sorted(self._scenarios))
            raise KeyError(f"Unknown scenario '{name}'. Available: {available}")
        return self._scenarios[name].factory()

    def names(self):
        return tuple(sorted(self._scenarios))


scenario_registry = ScenarioRegistry()
