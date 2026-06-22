#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for the installed planner framework package.

Run from the repository root:
    python3 scripts/tests/test_planner_framework_smoke.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "src", "xtd2_mission"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from xtd2_mission.planner_framework import EgoLikePlannerCore, Obstacle, PerceptionData, PlannerConfig, PlannerState, Vec3


def main():
    config = PlannerConfig(local_goal_distance=3.0, horizon=2.5, dt=0.5)
    planner = EgoLikePlannerCore(config)
    state = PlannerState(drone_id=1, position=Vec3(0.0, 0.0, -3.0), velocity=Vec3(), stamp=0.0)
    goal = Vec3(10.0, 0.0, -3.0)

    # Obstacle in front should trigger a non-forward or fallback-like response.
    perception = PerceptionData(
        obstacles=[Obstacle(position=Vec3(1.2, 0.0, -3.0), radius=0.35, source="static")],
        nearest_distance=1.2,
        near_field_danger=False,
    )
    report = planner.plan(state, goal, perception)
    cmd = report["command"]
    assert report["candidate_count"] > 0
    assert "velocity" in cmd
    assert cmd["mode"] in {"planner", "fallback_escape", "fallback_hover", "fallback_climb"}
    print("planner_framework smoke test passed")
    print(report["command"])


if __name__ == "__main__":
    main()
