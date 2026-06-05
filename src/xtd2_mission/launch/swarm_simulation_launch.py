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


def generate_launch_description():

    _pkg_dir = get_package_share_directory('xtd2_mission')
    _ws_root = os.path.abspath(os.path.join(_pkg_dir, '..', '..', '..', '..'))

    # Load defaults from YAML config
    defaults = _load_defaults(_pkg_dir)
    defaults['px4_dir'] = os.path.join(_ws_root, 'firmware', 'PX4-Autopilot')
    defaults['ws_root'] = _ws_root
    defaults['start_delay'] = '0'

    # All unique param names across stages
    all_param_names = list(dict.fromkeys([
        # PX4 SITL
        'num_drones', 'gz_world', 'start_port', 'px4_sys_id_base', 'px4_dir',
        'px4_disable_gcs_arm_check', 'leader_id', 'dynamic_obs_wall_x',
        'dynamic_obs_wall_y', 'dynamic_obs_target_y_spacing', 'start_wall_clearance',
        # Comm
        'start_delay', 'ws_root',
        # Mission
        'swarm_mode', 'warmup_sec', 'takeoff_z', 'mission_z', 'vertical_takeoff',
        'arm_to_offboard_delay', 'inter_drone_gap', 'post_offboard_hold_sec',
        'arm_offboard_max_retries', 'arm_retry_sleep', 'offboard_retry_sleep',
        'vehicle_status_timeout_sec', 'auto_arm_offboard', 'use_spawned_formation',
        'dynamic_obs_enable', 'dynamic_obs_mode', 'obstacle_config', 'scene_config',
        'dynamic_obs_visualize_gz', 'dynamic_obs_wall_z', 'dynamic_obs_wall_length',
        'dynamic_obs_wall_thickness', 'dynamic_obs_wall_height',
        'dynamic_obs_wall_segment_spacing', 'dynamic_obs_target_x',
        'dynamic_obs_target_y_base', 'second_wall_enable', 'second_wall_dx',
        'second_wall_dy', 'rear_wall_gap', 'rear_wall_length', 'third_wall_enable',
        'orca_safe_radius', 'orca_max_speed', 'orca_gain', 'orca_obs_gain',
        'orca_obs_influence_radius', 'orca_enable_after_z', 'orca_use_world_state',
        'path_planner_mode', 'astar_grid_resolution', 'astar_smooth_enable',
        'mission_start_x', 'duration', 'eval_mode', 'mission_mode',
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
    start_port = int(lc('start_port'))
    px4_dir = lc('px4_dir')
    px4_disable_gcs = lc('px4_disable_gcs_arm_check')
    leader_id = int(lc('leader_id'))
    wall_x = float(lc('dynamic_obs_wall_x'))
    wall_y = float(lc('dynamic_obs_wall_y'))
    target_y_spacing = float(lc('dynamic_obs_target_y_spacing'))
    start_wall_clearance = float(lc('start_wall_clearance'))
    mission_start_x = wall_x - start_wall_clearance

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
        spawn_ned_x = mission_start_x
        spawn_ned_y = wall_y + ((i - leader_id) * target_y_spacing)
        spawn_enu_x = spawn_ned_y
        spawn_enu_y = spawn_ned_x

        env_parts = [
            f'cd {px4_dir}',
            'export PX4_PARAM_CBRK_SUPPLY_CHK=894415',
            'export PX4_PARAM_MAV_SYS_ID=1',
        ]
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
            'export PX4_SIM_MODEL=gz_x500',
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

def _comm_remappings(px4_ns, xtd2_ns):
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
    return ' '.join(parts)


def _build_comm_nodes(lc, num_drones):
    actions = []
    ws_root = lc('ws_root')

    setup_cmd = (
        f'source /opt/ros/jazzy/setup.bash && '
        f'source {ws_root}/install/setup.bash && '
        f'exec '
    )

    for i in range(1, num_drones + 1):
        px4_ns = f'px4_{i}'
        xtd2_ns = f'x500_{i}'
        remap_str = _comm_remappings(px4_ns, xtd2_ns)

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
    gz_world = lc('gz_world')

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
        f'--leader-id {leader_id} '
        f'--ros-args -r __node:=swarm_state_exchange'
    )
    actions.append(ExecuteProcess(
        cmd=['bash', '-c', setup_cmd + state_exchange_args],
        output='log', name='swarm_state_exchange',
    ))

    # --- 2. Local ORCA (hybrid mode only) ---
    if swarm_mode == 'hybrid':
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
                f'--ros-args -r __node:=dynamic_obstacle_source'
            )
            actions.append(ExecuteProcess(
                cmd=['bash', '-c', setup_cmd + obs_args],
                output='log', name='dynamic_obstacle_source',
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
        'vertical_takeoff': lc('vertical_takeoff') == '1',
        'arm_offboard_max_retries': int(lc('arm_offboard_max_retries')),
        'arm_retry_sleep': float(lc('arm_retry_sleep')),
        'offboard_retry_sleep': float(lc('offboard_retry_sleep')),
        'arm_to_offboard_delay': float(lc('arm_to_offboard_delay')),
        'inter_drone_gap': float(lc('inter_drone_gap')),
        'post_offboard_hold_sec': post_offboard_hold,
        'vehicle_status_timeout_sec': float(lc('vehicle_status_timeout_sec')),
        'auto_arm_offboard': lc('auto_arm_offboard') == '1',
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
        # actions=[ExecuteProcess(
        #     cmd=[
        #         'xterm',
        #         '-fa', 'Monospace',
        #         '-fs', '16',
        #         '-hold',
        #         '-e',
        #         'bash',
        #         '-c',
        #         setup_cmd + mission_cmd,
        #     ],
        #     output='screen',
        #     name='mission_controller',
        # )],
        actions=[ExecuteProcess(
            cmd=['bash', '-c', setup_cmd + mission_cmd],
            output='log', name='mission_controller',
        )],
    ))

    return actions
