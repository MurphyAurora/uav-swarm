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
        self._rejoin_forced_until = 0.0

    def plan_candidate(self, state: PlannerState, local_goal: Vec3, obstacles) -> Optional[CandidateTrajectory]:
        now = time.time()
        local_map = self.map_builder.build(state, obstacles)
        start = local_map.body_to_grid(0.0, 0.0)
        if start is None:
            self.tracker.reset()
            self._clear_bypass(now, force_rejoin=False)
            self.diagnostics = {"status": "no_start_cell"}
            return None

        mission_target_body = self._world_to_body(local_goal.x - state.position.x, local_goal.y - state.position.y, state.heading)
        nearest = min((math.hypot(x, y) for x, y in local_map.points), default=999.0)
        front_blocked = self._front_blocked(local_map.points)
        mission_corridor_blocked = not self._target_corridor_clear(local_map.points, mission_target_body)
        bypass_target_body, bypass_mode = self._bypass_target(local_map.points, mission_target_body, now, front_blocked, mission_corridor_blocked)
        target_body = bypass_target_body if bypass_target_body is not None else mission_target_body
        corridor_blocked = not self._target_corridor_clear(local_map.points, target_body)
        if self._bypass_active and abs(self._bypass_side) > 0.5:
            self._side_latch = self._bypass_side
            self._side_latch_until = max(self._side_latch_until, now + 0.5)
        else:
            self._update_side_latch(local_map.points, target_body, now, front_blocked, corridor_blocked)

        replan_reason = self._replan_reason(now, state, local_goal, local_map, corridor_blocked)
        if bypass_mode in ("bypass_start", "bypass_hold", "rejoin", "force_rejoin") and not replan_reason:
            replan_reason = bypass_mode
        if replan_reason:
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                candidate = self._direct_bypass_candidate(state, local_goal, local_map, nearest, target_body, bypass_mode, replan_reason)
                if candidate is not None:
                    return candidate
                self.tracker.reset()
                self._committed_world_path = []
                self.diagnostics = self._diag("no_astar_path", replan_reason, local_map, nearest, front_blocked, corridor_blocked, self._bypass_diag(bypass_mode, target_body))
                return None
            self._commit_path(state, selected, local_goal, now)

        body_path = self._committed_body_path(state)
        if len(body_path) < 2 or not self._path_is_valid(local_map, body_path):
            selected = self._select_path(local_map, start, target_body)
            if selected is None:
                candidate = self._direct_bypass_candidate(state, local_goal, local_map, nearest, target_body, bypass_mode, "invalid_path_direct_bypass")
                if candidate is not None:
                    return candidate
                self.tracker.reset()
                self._committed_world_path = []
                self.diagnostics = self._diag("committed_path_invalid", "invalid_path_no_replacement", local_map, nearest, front_blocked, corridor_blocked, self._bypass_diag(bypass_mode, target_body))
                return None
            self._commit_path(state, selected, local_goal, now)
            body_path = self._committed_body_path(state)

        return self._track_body_path(state, local_goal, local_map, body_path, nearest, front_blocked, corridor_blocked, mission_corridor_blocked, target_body, bypass_mode, replan_reason or "latched", "tracking_committed_path")

    def _track_body_path(self, state: PlannerState, local_goal: Vec3, local_map, body_path, nearest: float,
                         front_blocked: bool, corridor_blocked: bool, mission_corridor_blocked: bool,
                         target_body: Tuple[float, float], bypass_mode: str, replan_reason: str, status: str) -> CandidateTrajectory:
        body_path = self._trim_passed_points(list(body_path))
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
            "replan_reason": replan_reason,
            "bypass_active": self._bypass_active,
            "bypass_side": self._bypass_side,
            "bypass_mode": bypass_mode,
            "bypass_target_body": {"x": float(target_body[0]), "y": float(target_body[1])},
            "rejoin_forced_left_sec": max(0.0, self._rejoin_forced_until - time.time()),
        }
        candidate = self.tracker.track(
            f"local_astar:{self._committed_label}", state, body_path, local_goal, nearest, clear, metadata
        )
        extra = self._bypass_diag(bypass_mode, target_body)
        extra.update(
            {
                "label": self._committed_label,
                "committed_left_sec": max(0.0, self._committed_until - time.time()),
                "path_len": len(body_path),
                "min_path_clearance": float(clear),
                "lookahead_body": dict(candidate.metadata.get("lookahead_body", {})),
                "speed": float(candidate.metadata.get("tracker_speed", 0.0)),
                "tracker": candidate.metadata.get("tracker", ""),
                "body_velocity": dict(candidate.metadata.get("body_velocity", {})),
                "mission_corridor_blocked": mission_corridor_blocked,
            }
        )
        self.diagnostics = self._diag(status, replan_reason, local_map, nearest, front_blocked, corridor_blocked, extra)
        return candidate

    def _direct_bypass_candidate(self, state: PlannerState, local_goal: Vec3, local_map, nearest: float,
                                 target_body: Tuple[float, float], bypass_mode: str, reason: str) -> Optional[CandidateTrajectory]:
        if not self._bypass_active or abs(self._bypass_side) < 0.5:
            return None
        side = self._bypass_side
        body_path = [
            (0.0, 0.0),
            (0.45, side * 0.55),
            (1.05, side * 1.10),
            (1.85, side * 1.65),
            (2.65, side * 1.95),
        ]
        clear = self._sampled_path_clearance(body_path, local_map.points)
        if clear < max(0.55, self.config.hard_clearance + 0.20):
            return None
        self._committed_label = f"direct_bypass_{'left' if side > 0 else 'right'}"
        self._committed_until = time.time() + 0.8
        mission_corridor_blocked = True
        candidate = self._track_body_path(
            state,
            local_goal,
            local_map,
            body_path,
            nearest,
            True,
            False,
            mission_corridor_blocked,
            target_body,
            bypass_mode,
            reason,
            "direct_bypass_path",
        )
        candidate.metadata["direct_bypass"] = True
        return candidate

    def _bypass_target(self, points, mission_target_body: Tuple[float, float], now: float,
                       front_blocked: bool, mission_corridor_blocked: bool):
        if self._bypass_active:
            min_hold = now - self._bypass_started >= 0.8
            clear_to_goal = self._target_corridor_clear(points, mission_target_body)
            goal_behind_or_side = mission_target_body[0] < 0.8
            side_error_large = abs(mission_target_body[1]) > 2.6 and not mission_corridor_blocked
            timed_out = now - self._bypass_started > 4.0
            if min_hold and (clear_to_goal or not mission_corridor_blocked or goal_behind_or_side or side_error_large or timed_out):
                self._clear_bypass(now, force_rejoin=True)
                return None, "force_rejoin"
            self._bypass_until = max(self._bypass_until, now + 0.35)
            return self._make_bypass_goal(self._bypass_side), "bypass_hold"

        if now < self._rejoin_forced_until and not mission_corridor_blocked:
            return None, "rejoin"

        if front_blocked or mission_corridor_blocked:
            side = self._pick_side(points, mission_target_body)
            self._bypass_active = True
            self._bypass_side = side
            self._bypass_started = now
            self._bypass_until = now + 1.8
            return self._make_bypass_goal(side), "bypass_start"

        return None, "normal"

    def _clear_bypass(self, now: float, force_rejoin: bool) -> None:
        self._bypass_active = False
        self._bypass_side = 0.0
        self._bypass_until = 0.0
        self._side_latch = 0.0
        self._side_latch_until = 0.0
        self._committed_world_path = []
        self._committed_label = ""
        self._committed_until = 0.0
        self.tracker.reset()
        if force_rejoin:
            self._rejoin_forced_until = now + 1.2

    def _make_bypass_goal(self, side: float) -> Tuple[float, float]:
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
            "rejoin_forced_left_sec": max(0.0, self._rejoin_forced_until - time.time()),
        }
