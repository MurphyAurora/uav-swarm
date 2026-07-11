#!/usr/bin/env python3
"""Planner registry for benchmark experiments.

Keeps benchmark code independent from concrete planner implementations.
Future adapters (A*, EGO-Swarm, SOGM, etc.) can register here without changing
scenario or evaluator code.
"""

from typing import Callable, Dict

from .planner_api import PlannerBase


PlannerFactory = Callable[[], PlannerBase]


class PlannerRegistry:
    """Small factory registry used by offline benchmark entry points."""

    def __init__(self) -> None:
        self._factories: Dict[str, PlannerFactory] = {}

    def register(self, name: str, factory: PlannerFactory) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> PlannerBase:
        if name not in self._factories:
            raise KeyError(f"unknown planner: {name}")
        return self._factories[name]()

    def names(self):
        return tuple(sorted(self._factories.keys()))


registry = PlannerRegistry()
