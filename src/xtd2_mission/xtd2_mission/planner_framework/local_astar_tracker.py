#!/usr/bin/env python3
"""Local A* frontend that delegates path execution to PathTracker."""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

from .common import CandidateTrajectory, PlannerConfig, PlannerState, Vec3
from .local_astar_stable import LocalAStarPlanner as StableLocalAStarPlanner
from .path_tracker import PathTracker


class LocalAStarPlanner(StableLocalAStarPlanner):
    def __init__(self, config: Optional[PlannerConfig] = None):
        super().__init__(config)
        self.tracker = PathTracker(self.config)
        self._bypass_active = False
        self._bypass_side = 0.0
        self._bypass_until = 0.0
        self._bypass_started = 0.0

    def plan_candidate(self, state: PlannerState, local_goal: Vec3, obstacles) -> Optional[CandidateTrajectory]:
        now = time.time()
        local_map = self.map_builder.build(state, obstacles)
        start = local_map.body_to_grid(0.0, 0.0)
        if start is None:
            self.tracker.reset()
            self._bypass_active = False
            self.diagnostics = {"status": "no_start_cell"}
            return None

        mission_target_body = self._world_to_body(local_goal.x - state.position.x, local_goal.y - state.position.y, state.heading)
        nearest = min((math.hypot(x, y) for x, y in local_map.points), default=999.0)
        front_blocked = self._front_blocked(local_map.points)
        mission_corridor_blocked = not self._target_corridor_clear(local_map.points, mission_target_body)
        bypass_target_body, bypass_mode = self._bypass_target(local_map.points, mission_target_body, now, front_blocked, mission_corridor_blocked)
        target_body = bypass_target_body if bypass_target_body is not None else mission_target_body
        corridor_blocked = not self._target_corridor_clear(local_map.points, target_body)
        # Keep the inherited side latch aligned with the explicit bypass state so
        # candidate generation does not choose the opposite side.
        if self._bypass_active and abs(self._bypass_side) > 0.5:
            self._side_latch = self._bypass_side
            self._side_latch_until = max(self._side_latch_until, now + 0.5)
        else:
            self._update_side_latch(local_map.points, target_body, now, front_blocked, corridor_blocked)

        replan_reason = self._replan_reason(now, state, local_goal, local_map, corridor_blocked)
        if bypass_mode in ("bypass_start", "bypass_hold", "rejoin") and not replan_reason:
            replan_reason = bypass_mode
        if replan_reason:
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self.tracker.reset()
                self._committed_world_path = []
                self.diagnostics = self._diag("no_astar_path", replan_reason, local_map, nearest, front_blocked, corridor_blocked, self._bypass_diag(bypass_mode, target_body))
                return None
            self._commit_path(state, selected, local_goal, now)

        body_path = self._committed_body_path(state)
        if len(body_path) < 2 or not self._path_is_valid(local_map, body_path):
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                self.tracker.reset()
                self._committed_world_path = []
                self.diagnostics = self._diag("committed_path_invalid", "invalid_path_no_replacement", local_map, nearest, front_blocked, corridor_blocked, self._bypass_diag(bypass_mode, target_body))
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
            "mission_corridor_blocked": mission_corridor_blocked,
            "side_latch": self._side_latch,
            "label": self._committed_label,
            "replan_reason": replan_reason or "latched",
            "bypass_active": self._bypass_active,
            "bypass_side": self._bypass_side,
            "bypass_mode": bypass_mode,
            "bypass_target_body": {"x": float(target_body[0]), "y": float(target_body[1])},
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
        extra = self._bypass_diag(bypass_mode, target_body)
        extra.update(
            {
                "label": self._committed_label,
                "committed_left_sec": max(0.0, self._committed_until - now),
                "path_len": len(body_path),
                "min_path_clearance": float(clear),
                "lookahead_body": dict(candidate.metadata.get("lookahead_body", {})),
                "speed": float(candidate.metadata.get("tracker_speed", 0.0)),
                "tracker": candidate.metadata.get("tracker", ""),
                "body_velocity": dict(candidate.metadata.get("body_velocity", {})),
                "mission_corridor_blocked": mission_corridor_blocked,
            }
        )
        self.diagnostics = self._diag(
            "tracking_committed_path", replan_reason or "latched", local_map, nearest, front_blocked, corridor_blocked, extra
        )
        return candidate

    def _bypass_target(self, points, mission_target_body: Tuple[float, float], now: float,
                       front_blocked: bool, mission_corridor_blocked: bool):
        """NORMAL/BYPASS/REJOIN state for static-obstacle bring-up."""
        if self._bypass_active:
            min_hold = now - self._bypass_started >= 1.2
            clear_to_goal = self._target_corridor_clear(points, mission_target_body)
            # Do not drop bypass immediately after a single clear frame; holding
            # avoids the common turn-stop-turn oscillation.
            if min_hold and clear_to_goal and now >= self._bypass_until:
                self._bypass_active = False
                self._bypass_side = 0.0
                self._bypass_until = 0.0
                return None, "rejoin"
            self._bypass_until = max(self._bypass_until, now + 0.35)
            return self._make_bypass_goal(self._bypass_side), "bypass_hold"

        if front_blocked or mission_corridor_blocked:
            side = self._pick_side(points, mission_target_body)
            self._bypass_active = True
            self._bypass_side = side
            self._bypass_started = now
            self._bypass_until = now + 2.4
            return self._make_bypass_goal(side), "bypass_start"

        return None, "normal"

    def _make_bypass_goal(self, side: float) -> Tuple[float, float]:
        # Explicit, forward-left/right subgoal.  It is intentionally not behind
        # the drone, so the tracker cannot turn this into a reverse escape.
        bx = min(3.2, max(2.2, 0.65 * self.config.astar_local_goal_dist))
        by = float(side) * max(1.8, self.config.drone_radius + self.config.obstacle_margin + 1.15)
        by = max(-self.config.astar_grid_right + 0.5, min(self.config.astar_grid_left - 0.5, by))
        return bx, by

    def _bypass_diag(self, mode: str, target_body: Tuple[float, float]) -> Dict[str, object]:
        return {
            "bypass_active": self._bypass_active,
            "bypass_side": self._bypass_side,
            "bypass_mode": mode,
            "bypass_target_body": {"x": float(target_body[0]), "y": float(target_body[1])},
            "bypass_left_sec": max(0.0, self._bypass_until - time.time()),
        }
