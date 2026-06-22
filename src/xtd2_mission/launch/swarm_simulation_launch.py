#!/usr/bin/env python3
"""
XTDrone2 Swarm Simulation — Main Launch

Orchestrates the full simulation pipeline:
  1. PX4 SITL + MicroXRCEAgent
  2. multirotor_communication nodes
  3. swarm_state_exchange + local_avoid_orca + dynamic_obstacle_source
  4. mission_controller

All default parameters are loaded from config/swarm_params.yaml.
Override any parameter via launch arguments:
  ros2 launch xtd2_mission swarm_simulation_launch.py num_drones:=5 swarm_mode:=hybrid
"""

import os
import json
import tempfile
import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def _load_defaults(pkg_dir):
    """Load default parameter values from config/swarm_params.yaml."""
    config_path = os.path.join(pkg_dir, 'config', 'swarm_params.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    defaults = {}
    for key, val in raw.items():
        if isinstance(val, bool):
            defaults[key] = '1' if val else '0'
        elif isinstance(val, (int, float)):
            defaults[key] = str(val)
        else:
            defaults[key] = str(val)
    return defaults


def _declare_args(param_names, defaults):
    """Create DeclareLaunchArgument actions for the given param names."""
    result = []
    for name in param_names:
        default = defaults.get(name, '')
        result.append(DeclareLaunchArgument(name, default_value=default))
    return result


def _optional_float(value):
    text = str(value).strip()
    if text == '':
        return None
    return float(text)


def _is_true(value):
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def generate_launch_description():

    _pkg_dir = get_package_share_directory('xtd2_mission')
    _ws_root = os.path.abspath(os.path.join(_pkg_dir, '..', '..', '..', '..'))

    # Load defaults from YAML config
    defaults = _load_defaults(_pkg_dir)
    defaults['px4_dir'] = os.path.join(_ws_root, 'firmware', 'PX4-Autopilot')
    defaults['ws_root'] = _ws_root
    defaults['start_delay'] = '0'
    defaults.setdefault('px4_use_versioned_local_position', '0')
    defaults.setdefault('spawn_override_ned_x', '')
    defaults.setdefault('spawn_override_ned_y', '')
    defaults.setdefault('ego_like_enable', '0')
    defaults.setdefault('ego_like_publish_cmd', '1')
    defaults.setdefault('ego_like_output_topic', '/xtdrone2/swarm/ego_like_planner_framework')
    defaults.setdefault('ego_like_cmd_topic_template', '/xtdrone2/x500_{id}/primitive_cmd_vel_ned')
    defaults.setdefault('ego_like_waypoints', '')
    defaults.setdefault('ego_like_period', '0.1')
    defaults.setdefault('ego_like_local_goal_distance', '3.0')
    defaults.setdefault('ego_like_output_alpha', '0.55')
    defaults.setdefault('ego_like_obstacle_margin', '0.35')
    defaults.setdefault('ego_like_hard_clearance', '0.15')
    defaults.setdefault('ego_like_emergency_clearance', '0.45')
    defaults.setdefault('ego_like_static_hard_clearance', '0.25')
    defaults.setdefault('ego_like_static_emergency_clearance', '0.55')
    defaults.setdefault('ego_like_goal_weight', '1.0')
    defaults.setdefault('ego_like_clearance_weight', '8.0')
    defaults.setdefault('ego_like_ttc_weight', '4.0')
    defaults.setdefault('ego_like_smooth_weight', '0.4')
    defaults.setdefault('ego_like_progress_weight', '6.0')
    defaults.setdefault('ego_like_reverse_penalty', '12.0')
    defaults.setdefault('ego_like_lateral_penalty', '1.5')
    defaults.setdefault('ego_like_lidar_topic_template', '/x500_{id}/lidar/points_local')
    defaults.setdefault('ego_like_lidar_timeout', '0.8')
    defaults.setdefault('ego_like_lidar_max_obstacles', '260')
    defaults.setdefault('ego_like_lidar_stride', '4')
    defaults.setdefault('ego_like_lidar_min_range', '0.65')
    defaults.setdefault('ego_like_lidar_max_range', '4.5')
    defaults.setdefault('ego_like_lidar_min_z', '-1.8')
    defaults.setdefault('ego_like_lidar_vertical_limit', '1.8')
    defaults.setdefault('ego_like_lidar_obstacle_radius', '0.16')

    # All unique param names across stages
    all_param_names = list(dict.fromkeys([
        # PX4 SITL
        'num_drones', 'gz_world', 'gz_gui', 'px4_sim_model', 'start_port', 'px4_sys_id_base', 'px4_dir',
        'px4_disable_gcs_arm_check', 'leader_id', 'dynamic_obs_wall_x',
        'dynamic_obs_wall_y', 'dynamic_obs_target_y_spacing', 'start_wall_clearance',
        'spawned_formation_axis', 'spawn_override_ned_x', 'spawn_override_ned_y',
        # Comm
        'start_delay', 'ws_root', 'px4_use_versioned_local_position',
        # Mission
        'swarm_mode', 'avoid_source', 'warmup_sec', 'takeoff_z', 'mission_z', 'vertical_takeoff',
        'arm_to_offboard_delay', 'inter_drone_gap', 'post_offboard_hold_sec',
        'arm_offboard_max_retries', 'arm_retry_sleep', 'offboard_retry_sleep',
        'vehicle_status_timeout_sec', 'auto_arm_offboard', 'use_spawned_formation',
        'dynamic_obs_enable', 'dynamic_obs_mode', 'obstacle_config', 'scene_config',
        'scene_freeze_dynamics',
        'dynamic_obs_visualize_gz', 'dynamic_obs_wall_z', 'dynamic_obs_wall_length',
        'dynamic_obs_wall_thickness', 'dynamic_obs_wall_height',
        'dynamic_obs_wall_segment_spacing', 'dynamic_obs_target_x',
        'dynamic_obs_target_y_base', 'second_wall_enable', 'second_wall_dx',
        'second_wall_dy', 'rear_wall_gap', 'rear_wall_length', 'third_wall_enable',
        'orca_safe_radius', 'orca_max_speed', 'orca_gain', 'orca_obs_gain',
        'orca_obs_influence_radius', 'orca_enable_after_z', 'orca_use_world_state',
        'obstacle_tracker_enable', 'tracker_max_match_distance',
        'tracker_dynamic_speed_threshold', 'tracker_static_speed_threshold',
        'tracker_dynamic_confirm_frames', 'tracker_static_confirm_frames',
        'tracker_max_missed_frames', 'tracker_velocity_alpha',
        'tracker_prediction_horizon', 'tracker_prediction_dt', 'tracker_prediction_max_speed',
        'trajectory_filter_enable', 'trajectory_filter_min_confidence',
        'trajectory_filter_min_speed', 'trajectory_filter_min_points',
        'trajectory_filter_max_age', 'trajectory_filter_horizon',
        'trajectory_filter_min_x', 'trajectory_filter_max_x',
        'trajectory_filter_min_y', 'trajectory_filter_max_y',
        'trajectory_filter_min_z', 'trajectory_filter_max_z',
        'trajectory_filter_merge_distance', 'trajectory_filter_relevance_radius',
        'trajectory_filter_state_timeout',
        'collision_monitor_enable', 'collision_horizon', 'collision_drone_radius',
        'collision_safety_margin', 'collision_warning_margin', 'collision_state_timeout',
        'collision_max_reports',
        'motion_primitive_enable',
        'ego_like_enable', 'ego_like_publish_cmd', 'ego_like_output_topic',
        'ego_like_cmd_topic_template', 'ego_like_waypoints', 'ego_like_period',
        'ego_like_local_goal_distance', 'ego_like_output_alpha',
        'ego_like_obstacle_margin', 'ego_like_hard_clearance',
        'ego_like_emergency_clearance', 'ego_like_static_hard_clearance',
        'ego_like_static_emergency_clearance', 'ego_like_goal_weight',
        'ego_like_clearance_weight', 'ego_like_ttc_weight',
        'ego_like_smooth_weight', 'ego_like_progress_weight',
        'ego_like_reverse_penalty', 'ego_like_lateral_penalty',
        'ego_like_lidar_topic_template', 'ego_like_lidar_timeout',
        'ego_like_lidar_max_obstacles', 'ego_like_lidar_stride',
        'ego_like_lidar_min_range', 'ego_like_lidar_max_range',
        'ego_like_lidar_min_z', 'ego_like_lidar_vertical_limit',
        'ego_like_lidar_obstacle_radius',
        'primitive_horizon', 'primitive_dt',
        'primitive_cruise_speed', 'primitive_lateral_speed', 'primitive_vertical_speed',
        'primitive_max_speed', 'primitive_drone_radius', 'primitive_safety_margin',
        'primitive_risk_radius', 'primitive_state_timeout', 'primitive_target_weight',
        'primitive_risk_weight', 'primitive_smooth_weight', 'primitive_publish_shadow_cmd',
        'primitive_emergency_clearance',
        'primitive_projection_alpha', 'primitive_projection_iterations',
        'primitive_projection_max_constraints',
        'primitive_escape_clearance',
        'primitive_hard_clearance',
        'primitive_static_hard_clearance',
        'primitive_static_emergency_clearance',
        'lidar_ttc_enable', 'lidar_ttc_warning_ttc', 'lidar_ttc_emergency_ttc',
        'lidar_ttc_min_speed', 'lidar_ttc_max_points', 'lidar_ttc_max_range',
        'lidar_ttc_min_range', 'lidar_ttc_min_z', 'lidar_ttc_vertical_limit', 'lidar_ttc_cone_angle_deg',
        'lidar_ttc_safety_radius', 'lidar_ttc_self_filter_radius', 'lidar_ttc_projection_gain',
        'lidar_ttc_brake_gain', 'lidar_ttc_back_speed', 'lidar_ttc_climb_speed',
        'path_planner_mode', 'astar_grid_resolution', 'astar_smooth_enable',
        'mission_start_x', 'duration', 'eval_mode', 'mission_mode',
        'formation_kp', 'leader_track_kp', 'max_follower_speed',
        'max_leader_speed', 'use_heading_offsets', 'lf_state_timeout',
        'use_virtual_leader', 'virtual_lag_limit',
    ]))

    args = _declare_args(all_param_names, defaults)

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)

    def launch_setup(context):
        actions = []

        def lc(name):
            return LaunchConfiguration(name).perform(context)

        num_drones = int(lc('num_drones'))

        # Timing
        px4_ready_delay = 3.0 + 7.0 * max(0, num_drones - 1) + 15.0
        comm_delay = px4_ready_delay
        mission_delay = px4_ready_delay + 5.0

        # ===== 1. PX4 SITL + MicroXRCEAgent =====
        actions.extend(_build_px4_sitl(lc, num_drones))

        # ===== 2. Comm nodes (after PX4 ready) =====
        actions.append(TimerAction(
            period=comm_delay,
            actions=_build_comm_nodes(lc, num_drones),
        ))

        # ===== 3. Swarm mission nodes (after comm ready) =====
        actions.append(TimerAction(
            period=mission_delay,
            actions=_build_mission_nodes(lc, num_drones),
        ))

        return actions

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld


# ---------------------------------------------------------------------------
# PX4 SITL
# ---------------------------------------------------------------------------

def _build_px4_sitl(lc, num_drones):
    actions = []
    gz_world = lc('gz_world')
    px4_sim_model = lc('px4_sim_model')
    start_port = int(lc('start_port'))
    px4_dir = lc('px4_dir')
    px4_disable_gcs = lc('px4_disable_gcs_arm_check')
    leader_id = int(lc('leader_id'))
    wall_x = float(lc('dynamic_obs_wall_x'))
    wall_y = float(lc('dynamic_obs_wall_y'))
    target_y_spacing = float(lc('dynamic_obs_target_y_spacing'))
    start_wall_clearance = float(lc('start_wall_clearance'))
    mission_start_x = wall_x - start_wall_clearance
    formation_axis = str(lc('spawned_formation_axis') or 'y').strip().lower()
    spawn_override_x = _optional_float(lc('spawn_override_ned_x'))
    spawn_override_y = _optional_float(lc('spawn_override_ned_y'))
    has_spawn_override = spawn_override_x is not None and spawn_override_y is not None

    if px4_disable_gcs == '1':
        px4_post_dir = os.path.join(
            px4_dir, 'build/px4_sitl_default/etc/init.d-posix/airframes')
        os.makedirs(px4_post_dir, exist_ok=True)
        px4_post_file = os.path.join(px4_post_dir, '4001_gz_x500.post')
        with open(px4_post_file, 'w') as f:
            f.write('#!/bin/sh\n# Simulation-only: allow offboard tests.\n')
            f.write('param set NAV_DLL_ACT 0\n')

    for i in range(1, num_drones + 1):
        port = start_port + i * 10
        actions.append(ExecuteProcess(
            cmd=['MicroXRCEAgent', 'udp4', '-p', str(port)],
            output='log', name=f'microxrceagent_{i}',
        ))

    px4_log_dir = os.path.expanduser('~/.ros/log/px4_sitl')
    os.makedirs(px4_log_dir, exist_ok=True)

    for i in range(1, num_drones + 1):
        dds_port = start_port + i * 10
        formation_offset = (i - leader_id) * target_y_spacing
        if has_spawn_override and num_drones == 1:
            spawn_ned_x = spawn_override_x
            spawn_ned_y = spawn_override_y
        elif formation_axis == 'x':
            spawn_ned_x = mission_start_x + formation_offset
            spawn_ned_y = wall_y
        else:
            spawn_ned_x = mission_start_x
            spawn_ned_y = wall_y + formation_offset
        spawn_enu_x = spawn_ned_y
        spawn_enu_y = spawn_ned_x

        env_parts = [
            f'cd {px4_dir}',
            'export PX4_PARAM_CBRK_SUPPLY_CHK=894415',
            'export PX4_PARAM_MAV_SYS_ID=1',
        ]
        if lc('gz_gui') == '0':
            env_parts.append('export HEADLESS=1')
        if px4_disable_gcs == '1':
            env_parts.append('export PX4_PARAM_NAV_DLL_ACT=0')
        env_parts.extend([
            f'export PX4_UXRCE_DDS_PORT={dds_port}',
            f'export PX4_GZ_WORLD={gz_world}',
        ])
        if i > 1:
            env_parts.append('export PX4_GZ_STANDALONE=1')
        env_parts.extend([
            f'export PX4_GZ_MODEL_POSE="{spawn_enu_x},{spawn_enu_y}"',
            'export PX4_SYS_AUTOSTART=4001',
            f'export PX4_SIM_MODEL={px4_sim_model}',
        ])

        px4_log = os.path.join(px4_log_dir, f'px4_{i}.log')
        cmd_str = (
            ' && '.join(env_parts)
            + f' && ./build/px4_sitl_default/bin/px4 -i {i}'
            + f' 2>&1 | tee {px4_log}'
        )

        delay = 3.0 + 7.0 * (i - 1)
        actions.append(TimerAction(
            period=delay,
            actions=[ExecuteProcess(
                cmd=['bash', '-c', cmd_str],
                output='screen', name=f'px4_{i}',
            )],
        ))

    return actions


# ---------------------------------------------------------------------------
# Comm nodes
# ---------------------------------------------------------------------------

def _comm_remappings(px4_ns, xtd2_ns, use_versioned_local_position=False):
    xtd2_topics = [
        'cmd_pose_local_ned', 'cmd_pose_local_flu',
        'cmd_vel_ned', 'cmd_vel_flu',
        'cmd_accel_ned', 'cmd_accel_flu',
        'cmd_attitude_flu',
    ]
    parts = []
    for topic in xtd2_topics:
        parts.append(f'-r /xtdrone2/{px4_ns}/{topic}:=/xtdrone2/{xtd2_ns}/{topic}')
    parts.append(
        f'-r /xtdrone2/{px4_ns}/debug/vehicle_state:=/xtdrone2/{xtd2_ns}/debug/vehicle_state'
    )
    if use_versioned_local_position:
        parts.append(
            f'-r /{px4_ns}/fmu/out/vehicle_local_position:=/{px4_ns}/fmu/out/vehicle_local_position_v1'
        )
    return ' '.join(parts)


def _build_comm_nodes(lc, num_drones):
    actions = []
    ws_root = lc('ws_root')
    use_versioned_local_position = lc('px4_use_versioned_local_position') == '1'

    setup_cmd = (
        f'source /opt/ros/jazzy/setup.bash && '
        f'source {ws_root}/install/setup.bash && '
        f'exec '
    )

    for i in range(1, num_drones + 1):
        px4_ns = f'px4_{i}'
        xtd2_ns = f'x500_{i}'
        remap_str = _comm_remappings(
            px4_ns,
            xtd2_ns,
            use_versioned_local_position=use_versioned_local_position,
        )

        cmd_str = (
            f'{setup_cmd}'
            f'ros2 run xtd2_communication multirotor_communication '
            f'--model gz_x500 --id {i} --namespace {px4_ns} '
            f'--ros-args -r __node:=comm_{i} {remap_str}'
        )

        actions.append(ExecuteProcess(
            cmd=['bash', '-c', cmd_str],
            output='log', name=f'comm_{i}',
        ))

    return actions


# ---------------------------------------------------------------------------
# Swarm mission nodes (formerly swarm_mission.launch.py)
# ---------------------------------------------------------------------------

def _build_mission_nodes(lc, num_drones):
    actions = []

    swarm_mode = lc('swarm_mode')
    ws_root = lc('ws_root')
    leader_id = int(lc('leader_id'))

    wall_x = float(lc('dynamic_obs_wall_x'))
    wall_y = float(lc('dynamic_obs_wall_y'))
    target_y_spacing = float(lc('dynamic_obs_target_y_spacing'))
    start_wall_clearance = float(lc('start_wall_clearance'))
    mission_start_x = wall_x - start_wall_clearance
    formation_axis = str(lc('spawned_formation_axis') or 'y').strip().lower()
    gz_world = lc('gz_world')
    spawn_override_x = _optional_float(lc('spawn_override_ned_x'))
    spawn_override_y = _optional_float(lc('spawn_override_ned_y'))
    has_spawn_override = spawn_override_x is not None and spawn_override_y is not None
    if has_spawn_override and num_drones == 1:
        mission_start_x = spawn_override_x
        wall_y = spawn_override_y

    warmup_sec = float(lc('warmup_sec'))
    post_offboard_hold = float(lc('post_offboard_hold_sec'))

    setup_cmd = (
        f'source /opt/ros/jazzy/setup.bash && '
        f'source {ws_root}/install/setup.bash && '
    )

    # --- 1. Swarm state exchange ---
    state_exchange_args = (
        f'ros2 run xtd2_mission swarm_state_exchange '
        f'--num-drones {num_drones} --rate 15 '
        f'--spawned-formation {lc("use_spawned_formation")} '
        f'--mission-start-x {mission_start_x} '
        f'--base-y {wall_y} '
        f'--y-spacing {target_y_spacing} '
        f'--formation-axis {formation_axis} '
        f'--leader-id {leader_id} '
        f'--ros-args -r __node:=swarm_state_exchange'
    )
    actions.append(ExecuteProcess(
        cmd=['bash', '-c', setup_cmd + state_exchange_args],
        output='log', name='swarm_state_exchange',
    ))

    # --- 2. Local ORCA (hybrid mode only) ---
    if swarm_mode == 'hybrid':
        if str(lc('avoid_source')).strip().lower() in ('orca', 'fused'):
            orca_args = (
                f'ros2 run xtd2_mission local_avoid_orca '
                f'--num-drones {num_drones} '
                f'--safe-radius {lc("orca_safe_radius")} '
                f'--max-speed {lc("orca_max_speed")} '
                f'--gain {lc("orca_gain")} '
                f'--obs-gain {lc("orca_obs_gain")} '
                f'--obs-influence-radius {lc("orca_obs_influence_radius")} '
                f'--enable-after-z {lc("orca_enable_after_z")} '
                f'--use-world-state {lc("orca_use_world_state")} '
                f'--ros-args -r __node:=local_avoid_orca'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + orca_args],
                output='log', name='local_avoid_orca',
            ))

        # --- 3. Dynamic obstacle source ---
        if lc('dynamic_obs_enable') == '1':
            obstacle_config_val = lc('obstacle_config')
            scene_config_val = lc('scene_config')

            obs_args = (
                f'ros2 run xtd2_mission dynamic_obstacle_source '
                f'--rate 10 --world {gz_world} '
                f'--visualize-gz {lc("dynamic_obs_visualize_gz")} '
                f'--mode {lc("dynamic_obs_mode")} '
            )
            if obstacle_config_val:
                obs_args += f'--obstacle-config {obstacle_config_val} '
            if scene_config_val:
                obs_args += f'--scene-config {scene_config_val} '

            obs_args += (
                f'--wall-x {lc("dynamic_obs_wall_x")} '
                f'--wall-y {lc("dynamic_obs_wall_y")} '
                f'--wall-z {lc("dynamic_obs_wall_z")} '
                f'--wall-length {lc("dynamic_obs_wall_length")} '
                f'--wall-thickness {lc("dynamic_obs_wall_thickness")} '
                f'--wall-height {lc("dynamic_obs_wall_height")} '
                f'--wall-segment-spacing {lc("dynamic_obs_wall_segment_spacing")} '
                f'--second-wall-enable {lc("second_wall_enable")} '
                f'--second-wall-dx {lc("second_wall_dx")} '
                f'--second-wall-dy {lc("second_wall_dy")} '
                f'--rear-wall-length {lc("rear_wall_length")} '
                f'--third-wall-enable {lc("third_wall_enable")} '
                f'--target-ball-enable 1 '
                f'--target-ball-num-drones {num_drones} '
                f'--target-ball-leader-id {leader_id} '
                f'--target-ball-target-x {lc("dynamic_obs_target_x")} '
                f'--target-ball-base-y {lc("dynamic_obs_target_y_base")} '
                f'--target-ball-target-z {lc("mission_z")} '
                f'--target-ball-y-spacing {target_y_spacing} '
                f'--scene-freeze-dynamics {lc("scene_freeze_dynamics")} '
                f'--ros-args -r __node:=dynamic_obstacle_source'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + obs_args],
                output='log', name='dynamic_obstacle_source',
            ))

        if lc('obstacle_tracker_enable') == '1':
            tracker_args = (
                f'ros2 run xtd2_mission obstacle_tracker '
                f'--input-topic /xtdrone2/swarm/dynamic_obstacles '
                f'--rate 10 '
                f'--max-match-distance {lc("tracker_max_match_distance")} '
                f'--dynamic-speed-threshold {lc("tracker_dynamic_speed_threshold")} '
                f'--static-speed-threshold {lc("tracker_static_speed_threshold")} '
                f'--dynamic-confirm-frames {lc("tracker_dynamic_confirm_frames")} '
                f'--static-confirm-frames {lc("tracker_static_confirm_frames")} '
                f'--max-missed-frames {lc("tracker_max_missed_frames")} '
                f'--velocity-alpha {lc("tracker_velocity_alpha")} '
                f'--prediction-horizon {lc("tracker_prediction_horizon")} '
                f'--prediction-dt {lc("tracker_prediction_dt")} '
                f'--prediction-max-speed {lc("tracker_prediction_max_speed")} '
                f'--ros-args -r __node:=obstacle_tracker'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + tracker_args],
                output='log', name='obstacle_tracker',
            ))

        predicted_topic = '/xtdrone2/swarm/predicted_dynamic_obstacles'
        if lc('trajectory_filter_enable') == '1':
            predicted_topic = '/xtdrone2/swarm/filtered_predicted_dynamic_obstacles'
            filter_args = (
                f'ros2 run xtd2_mission trajectory_filter '
                f'--input-topic /xtdrone2/swarm/predicted_dynamic_obstacles '
                f'--output-topic {predicted_topic} '
                f'--state-topic /xtdrone2/swarm/state_exchange '
                f'--num-drones {num_drones} '
                f'--min-confidence {lc("trajectory_filter_min_confidence")} '
                f'--min-speed {lc("trajectory_filter_min_speed")} '
                f'--min-points {lc("trajectory_filter_min_points")} '
                f'--max-age {lc("trajectory_filter_max_age")} '
                f'--horizon {lc("trajectory_filter_horizon")} '
                f'--min-x {lc("trajectory_filter_min_x")} '
                f'--max-x {lc("trajectory_filter_max_x")} '
                f'--min-y {lc("trajectory_filter_min_y")} '
                f'--max-y {lc("trajectory_filter_max_y")} '
                f'--min-z {lc("trajectory_filter_min_z")} '
                f'--max-z {lc("trajectory_filter_max_z")} '
                f'--merge-distance {lc("trajectory_filter_merge_distance")} '
                f'--relevance-radius {lc("trajectory_filter_relevance_radius")} '
                f'--state-timeout {lc("trajectory_filter_state_timeout")} '
                f'--ros-args -r __node:=trajectory_filter'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + filter_args],
                output='log', name='trajectory_filter',
            ))

        if lc('collision_monitor_enable') == '1':
            collision_args = (
                f'ros2 run xtd2_mission collision_risk_monitor '
                f'--num-drones {num_drones} '
                f'--state-topic /xtdrone2/swarm/state_exchange '
                f'--predicted-topic {predicted_topic} '
                f'--static-topic /xtdrone2/swarm/tracked_static_obstacles '
                f'--horizon {lc("collision_horizon")} '
                f'--drone-radius {lc("collision_drone_radius")} '
                f'--safety-margin {lc("collision_safety_margin")} '
                f'--warning-margin {lc("collision_warning_margin")} '
                f'--state-timeout {lc("collision_state_timeout")} '
                f'--max-reports {lc("collision_max_reports")} '
                f'--rate 10 '
                f'--ros-args -r __node:=collision_risk_monitor'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + collision_args],
                output='log', name='collision_risk_monitor',
            ))

        if _is_true(lc('ego_like_enable')):
            ego_like_args = (
                f'ros2 run xtd2_mission ego_like_planner_framework '
                f'--num-drones {num_drones} '
                f'--state-topic /xtdrone2/swarm/state_exchange '
                f'--predicted-topic {predicted_topic} '
                f'--static-topic /xtdrone2/swarm/tracked_static_obstacles '
                f'--lidar-topic-template "{lc("ego_like_lidar_topic_template")}" '
                f'--output-topic {lc("ego_like_output_topic")} '
                f'--publish-cmd {lc("ego_like_publish_cmd")} '
                f'--cmd-topic-template "{lc("ego_like_cmd_topic_template")}" '
                f'--waypoints "{lc("ego_like_waypoints")}" '
                f'--period {lc("ego_like_period")} '
                f'--state-timeout {lc("primitive_state_timeout")} '
                f'--target-x {lc("dynamic_obs_target_x")} '
                f'--target-y-base {lc("dynamic_obs_target_y_base")} '
                f'--target-y-spacing {lc("dynamic_obs_target_y_spacing")} '
                f'--target-z {lc("mission_z")} '
                f'--leader-id {lc("leader_id")} '
                f'--horizon {lc("primitive_horizon")} '
                f'--dt {lc("primitive_dt")} '
                f'--cruise-speed {lc("primitive_cruise_speed")} '
                f'--lateral-speed {lc("primitive_lateral_speed")} '
                f'--vertical-speed {lc("primitive_vertical_speed")} '
                f'--max-speed {lc("primitive_max_speed")} '
                f'--max-accel 0.8 '
                f'--drone-radius {lc("primitive_drone_radius")} '
                f'--obstacle-margin {lc("ego_like_obstacle_margin")} '
                f'--hard-clearance {lc("ego_like_hard_clearance")} '
                f'--emergency-clearance {lc("ego_like_emergency_clearance")} '
                f'--static-hard-clearance {lc("ego_like_static_hard_clearance")} '
                f'--static-emergency-clearance {lc("ego_like_static_emergency_clearance")} '
                f'--local-goal-distance {lc("ego_like_local_goal_distance")} '
                f'--goal-weight {lc("ego_like_goal_weight")} '
                f'--clearance-weight {lc("ego_like_clearance_weight")} '
                f'--ttc-weight {lc("ego_like_ttc_weight")} '
                f'--smooth-weight {lc("ego_like_smooth_weight")} '
                f'--progress-weight {lc("ego_like_progress_weight")} '
                f'--reverse-penalty {lc("ego_like_reverse_penalty")} '
                f'--lateral-penalty {lc("ego_like_lateral_penalty")} '
                f'--output-alpha {lc("ego_like_output_alpha")} '
                f'--lidar-timeout {lc("ego_like_lidar_timeout")} '
                f'--lidar-max-obstacles {lc("ego_like_lidar_max_obstacles")} '
                f'--lidar-stride {lc("ego_like_lidar_stride")} '
                f'--lidar-min-range {lc("ego_like_lidar_min_range")} '
                f'--lidar-max-range {lc("ego_like_lidar_max_range")} '
                f'--lidar-min-z {lc("ego_like_lidar_min_z")} '
                f'--lidar-vertical-limit {lc("ego_like_lidar_vertical_limit")} '
                f'--lidar-obstacle-radius {lc("ego_like_lidar_obstacle_radius")} '
                f'--ros-args -r __node:=ego_like_planner_framework'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + ego_like_args],
                output='log', name='ego_like_planner_framework',
            ))

        if _is_true(lc('motion_primitive_enable')) and not _is_true(lc('ego_like_enable')):
            primitive_args = (
                f'ros2 run xtd2_mission motion_primitive_selector '
                f'--num-drones {num_drones} '
                f'--state-topic /xtdrone2/swarm/state_exchange '
                f'--predicted-topic {predicted_topic} '
                f'--static-topic /xtdrone2/swarm/tracked_static_obstacles '
                f'--target-x {lc("dynamic_obs_target_x")} '
                f'--target-y-base {lc("dynamic_obs_target_y_base")} '
                f'--target-y-spacing {lc("dynamic_obs_target_y_spacing")} '
                f'--target-z {lc("mission_z")} '
                f'--leader-id {lc("leader_id")} '
                f'--horizon {lc("primitive_horizon")} '
                f'--dt {lc("primitive_dt")} '
                f'--cruise-speed {lc("primitive_cruise_speed")} '
                f'--lateral-speed {lc("primitive_lateral_speed")} '
                f'--vertical-speed {lc("primitive_vertical_speed")} '
                f'--max-speed {lc("primitive_max_speed")} '
                f'--drone-radius {lc("primitive_drone_radius")} '
                f'--safety-margin {lc("primitive_safety_margin")} '
                f'--risk-radius {lc("primitive_risk_radius")} '
                f'--state-timeout {lc("primitive_state_timeout")} '
                f'--target-weight {lc("primitive_target_weight")} '
                f'--risk-weight {lc("primitive_risk_weight")} '
                f'--smooth-weight {lc("primitive_smooth_weight")} '
                f'--publish-shadow-cmd {lc("primitive_publish_shadow_cmd")} '
                f'--emergency-clearance {lc("primitive_emergency_clearance")} '
                f'--projection-alpha {lc("primitive_projection_alpha")} '
                f'--projection-iterations {lc("primitive_projection_iterations")} '
                f'--projection-max-constraints {lc("primitive_projection_max_constraints")} '
                f'--escape-clearance {lc("primitive_escape_clearance")} '
                f'--hard-clearance {lc("primitive_hard_clearance")} '
                f'--static-hard-clearance {lc("primitive_static_hard_clearance")} '
                f'--static-emergency-clearance {lc("primitive_static_emergency_clearance")} '
                f'--ros-args -r __node:=motion_primitive_selector'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + primitive_args],
                output='log', name='motion_primitive_selector',
            ))

        if lc('lidar_ttc_enable') == '1':
            lidar_ttc_args = (
                f'ros2 run xtd2_mission lidar_ttc_safety_filter '
                f'--num-drones {num_drones} '
                f'--input-topic-template /xtdrone2/x500_{{id}}/primitive_cmd_vel_ned '
                f'--output-topic-template /xtdrone2/x500_{{id}}/safe_cmd_vel_ned '
                f'--lidar-topic-template /x500_{{id}}/lidar/points_local '
                f'--state-topic /xtdrone2/swarm/state_exchange '
                f'--warning-ttc {lc("lidar_ttc_warning_ttc")} '
                f'--emergency-ttc {lc("lidar_ttc_emergency_ttc")} '
                f'--min-speed {lc("lidar_ttc_min_speed")} '
                f'--max-points {lc("lidar_ttc_max_points")} '
                f'--max-range {lc("lidar_ttc_max_range")} '
                f'--min-range {lc("lidar_ttc_min_range")} '
                f'--min-z {lc("lidar_ttc_min_z")} '
                f'--vertical-limit {lc("lidar_ttc_vertical_limit")} '
                f'--cone-angle-deg {lc("lidar_ttc_cone_angle_deg")} '
                f'--safety-radius {lc("lidar_ttc_safety_radius")} '
                f'--self-filter-radius {lc("lidar_ttc_self_filter_radius")} '
                f'--projection-gain {lc("lidar_ttc_projection_gain")} '
                f'--brake-gain {lc("lidar_ttc_brake_gain")} '
                f'--back-speed {lc("lidar_ttc_back_speed")} '
                f'--climb-speed {lc("lidar_ttc_climb_speed")} '
                f'--ros-args -r __node:=lidar_ttc_safety_filter'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + lidar_ttc_args],
                output='log', name='lidar_ttc_safety_filter',
            ))

    # --- 4. Mission controller (delayed 2s within this group) ---
    namespaces = [f'x500_{i}' for i in range(1, num_drones + 1)]
    px4_namespaces = [f'px4_{i}' for i in range(1, num_drones + 1)]

    params = {
        'num_drones': num_drones,
        'namespaces': namespaces,
        'px4_namespaces': px4_namespaces,
        'warmup_sec': warmup_sec,
        'takeoff_z': float(lc('takeoff_z')),
        'mission_z': float(lc('mission_z')),
        'vertical_takeoff': lc('vertical_takeoff') == '1',
        'arm_offboard_max_retries': int(lc('arm_offboard_max_retries')),
        'arm_retry_sleep': float(lc('arm_retry_sleep')),
        'offboard_retry_sleep': float(lc('offboard_retry_sleep')),
        'arm_to_offboard_delay': float(lc('arm_to_offboard_delay')),
        'inter_drone_gap': float(lc('inter_drone_gap')),
        'post_offboard_hold_sec': post_offboard_hold,
        'vehicle_status_timeout_sec': float(lc('vehicle_status_timeout_sec')),
        'auto_arm_offboard': lc('auto_arm_offboard') == '1',
        'leader_id': leader_id,
        'wall_x': wall_x,
        'wall_y': wall_y,
        'wall_length': float(lc('dynamic_obs_wall_length')),
        'target_x': float(lc('dynamic_obs_target_x')),
        'target_y': float(lc('dynamic_obs_target_y_base')),
        'duration': float(lc('duration')),
        'eval_mode': lc('eval_mode'),
        'mission_mode': lc('mission_mode'),
        'y_spacing': target_y_spacing,
        'formation_kp': float(lc('formation_kp')),
        'leader_track_kp': float(lc('leader_track_kp')),
        'max_follower_speed': float(lc('max_follower_speed')),
        'max_leader_speed': float(lc('max_leader_speed')),
        'use_heading_offsets': lc('use_heading_offsets') == '1',
        'lf_state_timeout': float(lc('lf_state_timeout')),
        'use_virtual_leader': lc('use_virtual_leader') == '1',
        'virtual_lag_limit': float(lc('virtual_lag_limit')),
        'use_spawned_formation': lc('use_spawned_formation') == '1',
        'mission_start_x': mission_start_x,
    }
    params_file = os.path.join(tempfile.gettempdir(), f'launch_params_{os.getpid()}.yaml')
    with open(params_file, 'w') as f:
        f.write('mission_controller:\n')
        f.write('  ros__parameters:\n')
        for k, v in params.items():
            if isinstance(v, bool):
                f.write(f'    {k}: {"true" if v else "false"}\n')
            elif isinstance(v, list):
                f.write(f'    {k}: {json.dumps(v)}\n')
            else:
                f.write(f'    {k}: {v}\n')

    env_vars = {
        'AVOID_SOURCE': lc('avoid_source'),
        'PATH_PLANNER_MODE': lc('path_planner_mode'),
        'ASTAR_GRID_RESOLUTION': lc('astar_grid_resolution'),
        'ASTAR_SMOOTH_ENABLE': lc('astar_smooth_enable'),
        'VERTICAL_TAKEOFF': lc('vertical_takeoff'),
        'SECOND_WALL_ENABLE': lc('second_wall_enable'),
        'SECOND_WALL_DX': lc('second_wall_dx'),
        'REAR_WALL_GAP': lc('rear_wall_gap'),
        'REAR_WALL_LENGTH': lc('rear_wall_length'),
        'SECOND_WALL_DY': lc('second_wall_dy'),
        'THIRD_WALL_ENABLE': lc('third_wall_enable'),
        'OBSTACLE_CONFIG': lc('obstacle_config'),
        'DYNAMIC_OBS_MODE': lc('dynamic_obs_mode'),
    }
    env_exports = ' && '.join(f'export {k}={v}' for k, v in env_vars.items()) + ' && '
    mission_cmd = (
        f'{env_exports} '
        f'ros2 run xtd2_mission mission_controller '
        f'--ros-args -r __node:=mission_controller '
        f'--params-file {params_file}'
    )

    actions.append(TimerAction(
        period=2.0,
        actions=[ExecuteProcess(
            cmd=['bash', '-c', setup_cmd + mission_cmd],
            output='log', name='mission_controller',
        )],
    ))

    return actions
