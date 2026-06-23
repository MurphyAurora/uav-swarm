#!/usr/bin/env python3
"""Local A* frontend that delegates path execution to PathTracker."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional

from .common import CandidateTrajectory, PlannerConfig, PlannerState, Vec3
from .local_astar_stable import LocalAStarPlanner as StableLocalAStarPlanner
from .path_tracker import PathTracker


class LocalAStarPlanner(StableLocalAStarPlanner):
    def __init__(self, config: Optional[PlannerConfig] = None):
        super().__init__(config)
        self.tracker = PathTracker(self.config)

    def plan_candidate(self, state: PlannerState, local_goal: Vec3, obstacles) -> Optional[CandidateTrajectory]:
        now = time.time()
        local_map = self.map_builder.build(state, obstacles)
        start = local_map.body_to_grid(0.0, 0.0)
        if start is None:
            self.tracker.reset()
            self.diagnostics = {"status": "no_start_cell"}
            return None

        target_body = self._world_to_body(local_goal.x - state.position.x, local_goal.y - state.position.y, state.heading)
        nearest = min((math.hypot(x, y) for x, y in local_map.points), default=999.0)
        front_blocked = self._front_blocked(local_map.points)
        corridor_blocked = not self._target_corridor_clear(local_map.points, target_body)
        self._update_side_latch(local_map.points, target_body, now, front_blocked, corridor_blocked)

        replan_reason = self._replan_reason(now, state, local_goal, local_map, corridor_blocked)
        if replan_reason:
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self.tracker.reset()
                self._committed_world_path = []
                self.diagnostics = self._diag("no_astar_path", replan_reason, local_map, nearest, front_blocked, corridor_blocked, None)
                return None
            self._commit_path(state, selected, local_goal, now)

        body_path = self._committed_body_path(state)
        if len(body_path) < 2 or not self._path_is_valid(local_map, body_path):
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self.tracker.reset()
                self._committed_world_path = []
                self.diagnostics = self._diag("committed_path_invalid", "invalid_path_no_replacement", local_map, nearest, front_blocked, corridor_blocked, None)
                return None
            self._commit_path(state, selected, local_goal, now)
            body_path = self._committed_body_path(state)

        body_path = self._trim_passed_points(body_path)
        clear = self._sampled_path_clearance(body_path, local_map.points)
        metadata: Dict = {
            "frontend": "local_astar_tracker",
            "path_len": len(body_path),
            "raw_points": local_map.raw_count,
            "min_path_clearance": float(clear),
            "nearest": float(nearest),
            "front_blocked": front_blocked,
            "corridor_blocked": corridor_blocked,
            "side_latch": self._side_latch,
            "label": self._committed_label,
            "replan_reason": replan_reason or "latched",
        }
        candidate = self.tracker.track(
            f"local_astar:{self._committed_label}",
            state,
            body_path,
            local_goal,
            nearest,
            clear,
            metadata,
        )
        self.diagnostics = self._diag(
            "tracking_committed_path", replan_reason or "latched", local_map, nearest, front_blocked, corridor_blocked,
            {
                "label": self._committed_label,
                "committed_left_sec": max(0.0, self._committed_until - now),
                "path_len": len(body_path),
                "min_path_clearance": float(clear),
                "lookahead_body": dict(candidate.metadata.get("lookahead_body", {})),
                "speed": float(candidate.metadata.get("tracker_speed", 0.0)),
                "tracker": candidate.metadata.get("tracker", ""),
                "body_velocity": dict(candidate.metadata.get("body_velocity", {})),
            },
        )
        return candidate
