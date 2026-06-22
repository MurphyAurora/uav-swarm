#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke test for waypoint support in LocalGoalManager.

Run from repository root:
    python3 scripts/tests/test_goal_manager_waypoints.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "src", "xtd2_mission"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from xtd2_mission.planner_framework import PlannerConfig, PlannerState, Vec3
from xtd2_mission.planner_framework.goal_manager import LocalGoalManager


def main():
    config = PlannerConfig(local_goal_distance=2.0, local_goal_reached_radius=0.6)
    waypoints = LocalGoalManager.parse_waypoints('2,0,-3;4,2,-3;6,2,-3')
    manager = LocalGoalManager(config, waypoints=waypoints)

    state = PlannerState(drone_id=1, position=Vec3(0.0, 0.0, -3.0))
    final_goal = Vec3(8.0, 2.0, -3.0)
    local_goal = manager.compute_local_goal(state, final_goal)
    assert manager.active_waypoint_index() == 0
    assert abs(local_goal.x - 2.0) < 1.0e-6

    state = PlannerState(drone_id=1, position=Vec3(2.0, 0.0, -3.0))
    local_goal = manager.compute_local_goal(state, final_goal)
    assert manager.active_waypoint_index() == 1
    assert local_goal.y > 0.0

    print('goal_manager waypoint smoke test passed')
    print({'active_waypoint_index': manager.active_waypoint_index(), 'local_goal': local_goal.to_dict()})


if __name__ == '__main__':
    main()
