#!/usr/bin/env python3
"""ROS2 bridge node for the EGO-like planner framework."""

import argparse
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from .planner_framework import EgoLikePlannerCore, Obstacle, PlannerConfig, PlannerState, Vec3
from .planner_framework.goal_manager import LocalGoalManager
from .planner_framework.perception_interface import PerceptionInterface


def _bool_arg(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class EgoLikePlannerFrameworkNode(Node):
    def __init__(self, args):
        super().__init__("ego_like_planner_framework")
        self.args = args
        self.num_drones = int(args.num_drones)
        self.state_timeout = float(args.state_timeout)
        self.publish_cmd = _bool_arg(args.publish_cmd)
        self.cmd_topic_template = str(args.cmd_topic_template)
        self.manual_waypoints = self._parse_waypoints(args.waypoints, default_z=float(args.target_z))
        self.log_file_path = self._resolve_log_file(str(args.log_file or "").strip())
        self.log_fp = None
        if self.log_file_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file_path)), exist_ok=True)
            self.log_fp = open(self.log_file_path, "a", buffering=1)

        self.config = PlannerConfig(
            frontend_mode=str(args.frontend_mode),
            horizon=float(args.horizon),
            dt=float(args.dt),
            cruise_speed=float(args.cruise_speed),
            lateral_speed=float(args.lateral_speed),
            vertical_speed=float(args.vertical_speed),
            max_speed=float(args.max_speed),
            max_accel=float(args.max_accel),
            drone_radius=float(args.drone_radius),
            obstacle_margin=float(args.obstacle_margin),
            hard_clearance=float(args.hard_clearance),
            emergency_clearance=float(args.emergency_clearance),
            static_hard_clearance=float(args.static_hard_clearance),
            static_emergency_clearance=float(args.static_emergency_clearance),
            local_goal_distance=float(args.local_goal_distance),
            local_goal_nudge_enable=_bool_arg(args.local_goal_nudge_enable),
            corridor_enable=_bool_arg(args.corridor_enable),
            corridor_lookahead=float(args.corridor_lookahead),
            corridor_half_width=float(args.corridor_half_width),
            corridor_side_offset=float(args.corridor_side_offset),
            corridor_forward_offset=float(args.corridor_forward_offset),
            corridor_latch_sec=float(args.corridor_latch_sec),
            goal_weight=float(args.goal_weight),
            clearance_weight=float(args.clearance_weight),
            ttc_weight=float(args.ttc_weight),
            smooth_weight=float(args.smooth_weight),
            progress_weight=float(args.progress_weight),
            reverse_penalty=float(args.reverse_penalty),
            lateral_penalty=float(args.lateral_penalty),
            output_alpha=float(args.output_alpha),
            astar_grid_forward=float(args.astar_grid_forward),
            astar_grid_back=float(args.astar_grid_back),
            astar_grid_left=float(args.astar_grid_left),
            astar_grid_right=float(args.astar_grid_right),
            astar_resolution=float(args.astar_resolution),
            astar_inflation_radius=float(args.astar_inflation_radius),
            astar_local_goal_dist=float(args.astar_local_goal_dist),
            astar_lookahead_dist=float(args.astar_lookahead_dist),
            astar_min_range=float(args.astar_min_range),
            astar_max_range=float(args.astar_max_range),
            astar_z_min=float(args.astar_z_min),
            astar_z_max=float(args.astar_z_max),
            astar_path_latch_sec=float(args.astar_path_latch_sec),
            astar_replan_interval=float(args.astar_replan_interval),
            astar_progress_stall_sec=float(args.astar_progress_stall_sec),
            astar_progress_epsilon=float(args.astar_progress_epsilon),
            astar_progress_move_epsilon=float(args.astar_progress_move_epsilon),
            astar_recovery_sec=float(args.astar_recovery_sec),
            astar_latch_clearance=float(args.astar_latch_clearance),
        )

        self.states = {}
        self.headings = {}
        self.static_tracks = []
        self.dynamic_tracks = []
        self.lidar_clouds = {}
        self.lidar_stats = {}
        self._last_summary_log = 0.0
        self.planners = {i: EgoLikePlannerCore(self.config, waypoints=self.manual_waypoints) for i in range(1, self.num_drones + 1)}

        self.create_subscription(String, args.state_topic, self._state_cb, 10)
        self.create_subscription(String, args.predicted_topic, self._dynamic_cb, 10)
        self.create_subscription(String, args.static_topic, self._static_cb, 10)
        self.lidar_topic_template = str(args.lidar_topic_template)
        for drone_id in range(1, self.num_drones + 1):
            self.create_subscription(
                PointCloud2,
                self.lidar_topic_template.format(id=drone_id),
                lambda msg, i=drone_id: self._lidar_cb(i, msg),
                5,
            )
        self.report_pub = self.create_publisher(String, args.output_topic, 10)
        self.cmd_pubs = {i: self.create_publisher(Twist, self._cmd_topic_for(i), 10) for i in range(1, self.num_drones + 1)}
        self.timer = self.create_timer(float(args.period), self._run)
        self.get_logger().info(
            f"ego_like planner framework started: num={self.num_drones}, publish_cmd={int(self.publish_cmd)}, "
            f"frontend_mode={self.config.frontend_mode}, cmd_topic_template={self.cmd_topic_template}, "
            f"lidar={self.lidar_topic_template}, log_file={self.log_file_path or 'disabled'}"
        )

    def destroy_node(self):
        if self.log_fp is not None:
            self.log_fp.close()
        super().destroy_node()

    def _resolve_log_file(self, raw: str) -> str:
        if not raw:
            return ""
        if raw.lower() == "auto":
            stamp = time.strftime("%Y%m%d_%H%M%S")
            return os.path.expanduser(f"~/ws_xtd2/log/ego_like_debug_{stamp}.jsonl")
        return os.path.expanduser(raw)

    def _cmd_topic_for(self, drone_id):
        return self.cmd_topic_template.format(id=drone_id)

    def _parse_waypoints(self, raw, default_z):
        waypoints = LocalGoalManager.parse_waypoints(raw or "")
        return [Vec3(w.x, w.y, w.z if abs(w.z) > 1.0e-9 else float(default_z)) for w in waypoints]

    def _state_cb(self, msg):
        try:
            data = json.loads(msg.data)
            self.states = {int(s["id"]): PlannerState.from_mapping(s, prefer_world=True) for s in data.get("states", [])}
            self.headings = {int(s["id"]): self._finite(s.get("heading", 0.0)) for s in data.get("states", [])}
        except Exception as exc:
            self.get_logger().warn(f"failed to parse state_exchange: {exc}")

    def _dynamic_cb(self, msg):
        try:
            self.dynamic_tracks = json.loads(msg.data).get("tracks", [])
        except Exception as exc:
            self.get_logger().warn(f"failed to parse dynamic tracks: {exc}")

    def _static_cb(self, msg):
        try:
            self.static_tracks = json.loads(msg.data).get("tracks", [])
        except Exception as exc:
            self.get_logger().warn(f"failed to parse static tracks: {exc}")

    def _lidar_cb(self, drone_id, msg):
        self.lidar_clouds[int(drone_id)] = {"msg": msg, "stamp": time.time()}

    def _target(self, drone_id):
        off = (int(drone_id) - int(self.args.leader_id)) * float(self.args.target_y_spacing)
        return Vec3(float(self.args.target_x), float(self.args.target_y_base) + off, float(self.args.target_z))

    def _run(self):
        now = time.time()
        reports = []
        raw_states = [
            {
                "id": state.drone_id,
                "x": state.position.x,
                "y": state.position.y,
                "z": state.position.z,
                "vx": state.velocity.x,
                "vy": state.velocity.y,
                "vz": state.velocity.z,
                "stamp": state.stamp,
                "heading": state.heading,
            }
            for state in self.states.values()
        ]

        for drone_id, state in sorted(self.states.items()):
            if drone_id not in self.planners:
                continue
            if now - state.stamp > self.state_timeout:
                continue

            perception = PerceptionInterface(self.config.drone_radius, 0.3, self.config.emergency_clearance)
            perception.update_static_tracks(self.static_tracks, now)
            perception.update_dynamic_tracks(self.dynamic_tracks, now)
            perception.update_swarm_states(raw_states, drone_id, self.state_timeout)
            lidar_obstacles = self._lidar_obstacles(drone_id, state, now)
            perception.update_lidar_obstacles(lidar_obstacles, now)
            report = self.planners[drone_id].plan(state, self._target(drone_id), perception.build(state.position))
            report["lidar_obstacle_count"] = len(lidar_obstacles)
            report["lidar_stats"] = dict(self.lidar_stats.get(int(drone_id), {}))
            reports.append(report)
            if self.log_fp is not None:
                self.log_fp.write(json.dumps(report, ensure_ascii=False) + "\n")
            if self.publish_cmd:
                self._publish_cmd(drone_id, report)

        out = String()
        out.data = json.dumps({"framework": "ego_like", "reports": reports}, ensure_ascii=False)
        self.report_pub.publish(out)
        self._log_summary(now, reports)

    def _publish_cmd(self, drone_id, report):
        velocity = report.get("command", {}).get("velocity", {})
        msg = Twist()
        msg.linear.x = float(velocity.get("x", 0.0))
        msg.linear.y = float(velocity.get("y", 0.0))
        msg.linear.z = float(velocity.get("z", 0.0))
        self.cmd_pubs[drone_id].publish(msg)

    def _log_summary(self, now, reports):
        if now - self._last_summary_log < 1.0:
            return
        self._last_summary_log = now
        if not reports:
            self.get_logger().info(
                f"ego_like summary: no fresh states, states={len(self.states)}, "
                f"static={len(self.static_tracks)}, dynamic={len(self.dynamic_tracks)}"
            )
            return
        parts = []
        for report in reports:
            cmd = report.get("command", {})
            velocity = cmd.get("velocity", {})
            local_goal = report.get("local_goal", {})
            final_goal = report.get("final_goal", {})
            best_costs = (report.get("best") or {}).get("costs", {})
            best = report.get("best") or {}
            frontend = ((report.get("frontend") or {}).get("local_astar") or {})
            goal_diag = report.get("goal_manager") or {}
            lidar_stats = report.get("lidar_stats") or {}
            parts.append(
                "x500_{id} mode={mode} traj={traj} v=({vx:.2f},{vy:.2f},{vz:.2f}) "
                "safe={safe}/{cand} feasible={feasible} lidar={lidar} lnear={lnear:.2f} clear={clear:.2f} local=({lx:.1f},{ly:.1f},{lz:.1f}) "
                "goal=({gx:.1f},{gy:.1f},{gz:.1f}) goal_mode={goal_mode} corridor={corr:.0f}/{corr_left:.1f}s "
                "astar={astar} repl={repl} stuck={stuck:.1f}s "
                "cost(g={cg:.2f},p={cp:.2f},r={cr:.2f},l={cl:.2f}) reason={reason}".format(
                    id=int(report.get("drone_id", 0)),
                    mode=str(cmd.get("mode", "")),
                    traj=str(cmd.get("source_trajectory", "")),
                    vx=float(velocity.get("x", 0.0)),
                    vy=float(velocity.get("y", 0.0)),
                    vz=float(velocity.get("z", 0.0)),
                    safe=int(report.get("safe_count", 0)),
                    cand=int(report.get("candidate_count", 0)),
                    feasible=int(report.get("feasible_count", 0)),
                    lidar=int(report.get("lidar_obstacle_count", 0)),
                    lnear=float(lidar_stats.get("nearest_body_range", 999.0)),
                    clear=float(best.get("min_static_clearance", best.get("min_clearance", 999.0))),
                    lx=float(local_goal.get("x", 0.0)),
                    ly=float(local_goal.get("y", 0.0)),
                    lz=float(local_goal.get("z", 0.0)),
                    gx=float(final_goal.get("x", 0.0)),
                    gy=float(final_goal.get("y", 0.0)),
                    gz=float(final_goal.get("z", 0.0)),
                    goal_mode=str(goal_diag.get("goal_reason", "")),
                    corr=float(goal_diag.get("corridor_side", 0.0)),
                    corr_left=float(goal_diag.get("corridor_left_sec", 0.0)),
                    astar=str(frontend.get("status", "off")),
                    repl=str(frontend.get("replan_reason", frontend.get("label", ""))),
                    stuck=float(frontend.get("stalled_sec", 0.0)),
                    cg=float(best_costs.get("goal", 0.0)),
                    cp=float(best_costs.get("progress", 0.0)),
                    cr=float(best_costs.get("reverse", 0.0)),
                    cl=float(best_costs.get("lateral", 0.0)),
                    reason=str(cmd.get("reason", "")),
                )
            )
        self.get_logger().info(
            f"ego_like summary: static={len(self.static_tracks)}, dynamic={len(self.dynamic_tracks)} | "
            + " | ".join(parts)
        )

    def _lidar_obstacles(self, drone_id, state, now):
        cloud = self.lidar_clouds.get(int(drone_id))
        stats = {
            "stamp": now,
            "raw_points": 0,
            "kept_points": 0,
            "nearest_body_range": 999.0,
            "nearest_body_x": 999.0,
            "nearest_body_y": 999.0,
            "nearest_body_z": 999.0,
            "cloud_age": 999.0,
        }
        if cloud is None or now - float(cloud["stamp"]) > float(self.args.lidar_timeout):
            self.lidar_stats[int(drone_id)] = stats
            return []
        stats["cloud_age"] = now - float(cloud["stamp"])
        heading = self.headings.get(int(drone_id), 0.0)
        c = math.cos(heading)
        s = math.sin(heading)
        obstacles = []
        stride = max(1, int(self.args.lidar_stride))
        max_points = max(1, int(self.args.lidar_max_obstacles))
        min_range = float(self.args.lidar_min_range)
        max_range = float(self.args.lidar_max_range)
        min_z = float(self.args.lidar_min_z)
        vertical_limit = float(self.args.lidar_vertical_limit)
        radius = float(self.args.lidar_obstacle_radius)
        for idx, point in enumerate(point_cloud2.read_points(cloud["msg"], field_names=("x", "y", "z"), skip_nans=True)):
            stats["raw_points"] += 1
            if idx % stride != 0:
                continue
            px = self._finite(point[0])
            py = self._finite(point[1])
            pz = self._finite(point[2])
            horizontal = math.hypot(px, py)
            if horizontal < min_range or horizontal > max_range:
                continue
            if pz < min_z or pz > vertical_limit:
                continue
            if horizontal < stats["nearest_body_range"]:
                stats["nearest_body_range"] = horizontal
                stats["nearest_body_x"] = px
                stats["nearest_body_y"] = py
                stats["nearest_body_z"] = pz
            wx = state.position.x + c * px - s * py
            wy = state.position.y + s * px + c * py
            wz = state.position.z + pz
            obstacles.append(
                Obstacle(
                    position=Vec3(wx, wy, wz),
                    radius=radius,
                    source="lidar_near_field",
                    obstacle_id=f"lidar_{drone_id}_{len(obstacles)}",
                )
            )
            if len(obstacles) >= max_points:
                break
        stats["kept_points"] = len(obstacles)
        self.lidar_stats[int(drone_id)] = stats
        return obstacles

    @staticmethod
    def _finite(value, default=0.0):
        try:
            out = float(value)
        except Exception:
            return float(default)
        return out if math.isfinite(out) else float(default)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="EGO-like planner framework node")
    parser.add_argument("--num-drones", type=int, default=1)
    parser.add_argument("--state-topic", default="/xtdrone2/swarm/state_exchange")
    parser.add_argument("--predicted-topic", default="/xtdrone2/swarm/filtered_predicted_dynamic_obstacles")
    parser.add_argument("--static-topic", default="/xtdrone2/swarm/tracked_static_obstacles")
    parser.add_argument("--lidar-topic-template", default="/x500_{id}/lidar/points_local")
    parser.add_argument("--output-topic", default="/xtdrone2/swarm/ego_like_planner_framework")
    parser.add_argument("--publish-cmd", default="0")
    parser.add_argument("--cmd-topic-template", default="/xtdrone2/x500_{id}/primitive_cmd_vel_ned")
    parser.add_argument("--waypoints", default="")
    parser.add_argument("--frontend-mode", default="local_astar", choices=("primitive", "hybrid_astar", "local_astar", "astar", "hybrid"))
    parser.add_argument("--period", type=float, default=0.1)
    parser.add_argument("--state-timeout", type=float, default=2.0)
    parser.add_argument("--target-x", type=float, default=30.0)
    parser.add_argument("--target-y-base", type=float, default=26.0)
    parser.add_argument("--target-y-spacing", type=float, default=1.8)
    parser.add_argument("--target-z", type=float, default=-3.0)
    parser.add_argument("--leader-id", type=int, default=3)
    parser.add_argument("--horizon", type=float, default=2.5)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--cruise-speed", type=float, default=0.55)
    parser.add_argument("--lateral-speed", type=float, default=0.45)
    parser.add_argument("--vertical-speed", type=float, default=0.35)
    parser.add_argument("--max-speed", type=float, default=0.75)
    parser.add_argument("--max-accel", type=float, default=0.6)
    parser.add_argument("--drone-radius", type=float, default=0.35)
    parser.add_argument("--obstacle-margin", type=float, default=0.45)
    parser.add_argument("--hard-clearance", type=float, default=0.25)
    parser.add_argument("--emergency-clearance", type=float, default=0.75)
    parser.add_argument("--static-hard-clearance", type=float, default=0.55)
    parser.add_argument("--static-emergency-clearance", type=float, default=1.0)
    parser.add_argument("--local-goal-distance", type=float, default=3.8)
    parser.add_argument("--local-goal-nudge-enable", default="0")
    parser.add_argument("--corridor-enable", default="1")
    parser.add_argument("--corridor-lookahead", type=float, default=5.0)
    parser.add_argument("--corridor-half-width", type=float, default=1.10)
    parser.add_argument("--corridor-side-offset", type=float, default=2.2)
    parser.add_argument("--corridor-forward-offset", type=float, default=2.8)
    parser.add_argument("--corridor-latch-sec", type=float, default=4.0)
    parser.add_argument("--goal-weight", type=float, default=1.0)
    parser.add_argument("--clearance-weight", type=float, default=8.0)
    parser.add_argument("--ttc-weight", type=float, default=4.0)
    parser.add_argument("--smooth-weight", type=float, default=0.4)
    parser.add_argument("--progress-weight", type=float, default=6.0)
    parser.add_argument("--reverse-penalty", type=float, default=12.0)
    parser.add_argument("--lateral-penalty", type=float, default=1.5)
    parser.add_argument("--output-alpha", type=float, default=0.45)
    parser.add_argument("--astar-grid-forward", type=float, default=7.0)
    parser.add_argument("--astar-grid-back", type=float, default=2.5)
    parser.add_argument("--astar-grid-left", type=float, default=5.0)
    parser.add_argument("--astar-grid-right", type=float, default=5.0)
    parser.add_argument("--astar-resolution", type=float, default=0.25)
    parser.add_argument("--astar-inflation-radius", type=float, default=0.22)
    parser.add_argument("--astar-local-goal-dist", type=float, default=5.0)
    parser.add_argument("--astar-lookahead-dist", type=float, default=1.10)
    parser.add_argument("--astar-min-range", type=float, default=0.18)
    parser.add_argument("--astar-max-range", type=float, default=7.5)
    parser.add_argument("--astar-z-min", type=float, default=-1.4)
    parser.add_argument("--astar-z-max", type=float, default=1.6)
    parser.add_argument("--astar-path-latch-sec", type=float, default=2.2)
    parser.add_argument("--astar-replan-interval", type=float, default=1.0)
    parser.add_argument("--astar-progress-stall-sec", type=float, default=2.0)
    parser.add_argument("--astar-progress-epsilon", type=float, default=0.20)
    parser.add_argument("--astar-progress-move-epsilon", type=float, default=0.22)
    parser.add_argument("--astar-recovery-sec", type=float, default=1.8)
    parser.add_argument("--astar-latch-clearance", type=float, default=0.35)
    parser.add_argument("--lidar-timeout", type=float, default=0.8)
    parser.add_argument("--lidar-max-obstacles", type=int, default=320)
    parser.add_argument("--lidar-stride", type=int, default=3)
    parser.add_argument("--lidar-min-range", type=float, default=0.75)
    parser.add_argument("--lidar-max-range", type=float, default=5.5)
    parser.add_argument("--lidar-min-z", type=float, default=-1.8)
    parser.add_argument("--lidar-vertical-limit", type=float, default=1.8)
    parser.add_argument("--lidar-obstacle-radius", type=float, default=0.18)
    parser.add_argument("--log-file", default="auto")
    return parser


def main():
    args, ros_args = build_arg_parser().parse_known_args()
    rclpy.init(args=ros_args)
    node = EgoLikePlannerFrameworkNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
