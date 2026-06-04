#!/usr/bin/env python3
"""
XTDrone2 Swarm Simulation Launch File

Replaces start1.sh with a fully ROS 2 launch-based system.

Usage:
  ros2 launch xtd2_mission swarm_simulation_launch.py \
    num_drones:=5 swarm_mode:=hybrid gz_world:=default

Namespace mapping (without modifying XTDrone2 source):
  - comm node launched with --namespace px4_{i}  → DDS topics align with PX4
  - XTDrone2 topics remapped from /xtdrone2/px4_{i}/* → /xtdrone2/x500_{i}/*
  - Service /xtdrone2/px4_{i}/cmd stays as-is (mission_controller uses px4_ns for service calls)
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _comm_remappings(px4_ns, xtd2_ns):
    """Generate remappings: /xtdrone2/{px4_ns}/* → /xtdrone2/{xtd2_ns}/*."""
    xtd2_topics = [
        'cmd_pose_local_ned', 'cmd_pose_local_flu',
        'cmd_vel_ned', 'cmd_vel_flu',
        'cmd_accel_ned', 'cmd_accel_flu',
        'cmd_attitude_flu',
    ]
    remappings = []
    for topic in xtd2_topics:
        remappings.append(
            (f'/xtdrone2/{px4_ns}/{topic}', f'/xtdrone2/{xtd2_ns}/{topic}')
        )
    # Debug topic
    remappings.append(
        (f'/xtdrone2/{px4_ns}/debug/vehicle_state',
         f'/xtdrone2/{xtd2_ns}/debug/vehicle_state')
    )
    return remappings


def generate_launch_description():

    # ===== Declare all launch arguments =====
    args = []

    # Core
    args.append(DeclareLaunchArgument('num_drones', default_value='5'))
    args.append(DeclareLaunchArgument('gz_world', default_value='default'))
    args.append(DeclareLaunchArgument('start_port', default_value='8888'))
    args.append(DeclareLaunchArgument('px4_sys_id_base', default_value='2'))
    args.append(DeclareLaunchArgument('px4_dir',
        default_value=os.path.expanduser('~/test/px4_ws/firmware/PX4-Autopilot')))
    args.append(DeclareLaunchArgument('ws_root',
        default_value=os.path.expanduser('~/test/px4_ws')))

    # Swarm mode
    args.append(DeclareLaunchArgument('swarm_mode', default_value='hybrid'))
    args.append(DeclareLaunchArgument('avoid_scope', default_value='leader'))
    args.append(DeclareLaunchArgument('auto_arm_offboard', default_value='1'))

    # Warmup / takeoff
    args.append(DeclareLaunchArgument('warmup_sec', default_value='25'))
    args.append(DeclareLaunchArgument('takeoff_z', default_value='-2.0'))
    args.append(DeclareLaunchArgument('mission_z', default_value='-3.0'))
    args.append(DeclareLaunchArgument('vertical_takeoff', default_value='1'))

    # ARM / OFFBOARD
    args.append(DeclareLaunchArgument('arm_to_offboard_delay', default_value='0.8'))
    args.append(DeclareLaunchArgument('inter_drone_gap', default_value='0.2'))
    args.append(DeclareLaunchArgument('post_offboard_hold_sec', default_value='3'))
    args.append(DeclareLaunchArgument('arm_offboard_max_retries', default_value='6'))
    args.append(DeclareLaunchArgument('arm_retry_sleep', default_value='0.6'))
    args.append(DeclareLaunchArgument('offboard_retry_sleep', default_value='0.8'))
    args.append(DeclareLaunchArgument('vehicle_status_timeout_sec', default_value='4'))

    # Formation
    args.append(DeclareLaunchArgument('leader_id', default_value='3'))
    args.append(DeclareLaunchArgument('formation_kp', default_value='0.45'))
    args.append(DeclareLaunchArgument('leader_track_kp', default_value='0.45'))
    args.append(DeclareLaunchArgument('max_follower_speed', default_value='0.85'))
    args.append(DeclareLaunchArgument('max_leader_speed', default_value='0.85'))
    args.append(DeclareLaunchArgument('use_heading_offsets', default_value='0'))
    args.append(DeclareLaunchArgument('lf_state_timeout', default_value='4.0'))
    args.append(DeclareLaunchArgument('use_virtual_leader', default_value='1'))
    args.append(DeclareLaunchArgument('use_spawned_formation', default_value='1'))

    # Obstacles
    args.append(DeclareLaunchArgument('dynamic_obs_enable', default_value='1'))
    args.append(DeclareLaunchArgument('dynamic_obs_mode', default_value='static_wall'))
    args.append(DeclareLaunchArgument('obstacle_config', default_value=''))
    args.append(DeclareLaunchArgument('scene_config', default_value=''))
    args.append(DeclareLaunchArgument('dynamic_obs_visualize_gz', default_value='1'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_x', default_value='3.0'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_y', default_value='1.0'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_z', default_value='0.9'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_length', default_value='30.0'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_thickness', default_value='0.25'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_height', default_value='4.8'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_segment_spacing', default_value='0.6'))
    args.append(DeclareLaunchArgument('dynamic_obs_target_x', default_value='30.0'))
    args.append(DeclareLaunchArgument('dynamic_obs_target_y_base', default_value='26.0'))
    args.append(DeclareLaunchArgument('dynamic_obs_target_y_spacing', default_value='1.8'))
    args.append(DeclareLaunchArgument('second_wall_enable', default_value='1'))
    args.append(DeclareLaunchArgument('second_wall_dx', default_value='15.0'))
    args.append(DeclareLaunchArgument('second_wall_dy', default_value='109.0'))
    args.append(DeclareLaunchArgument('rear_wall_gap', default_value='18.0'))
    args.append(DeclareLaunchArgument('rear_wall_length', default_value='200.0'))
    args.append(DeclareLaunchArgument('third_wall_enable', default_value='1'))
    args.append(DeclareLaunchArgument('start_wall_clearance', default_value='8.0'))

    # ORCA
    args.append(DeclareLaunchArgument('orca_safe_radius', default_value='1.2'))
    args.append(DeclareLaunchArgument('orca_max_speed', default_value='0.22'))
    args.append(DeclareLaunchArgument('orca_gain', default_value='0.45'))
    args.append(DeclareLaunchArgument('orca_obs_gain', default_value='0.85'))
    args.append(DeclareLaunchArgument('orca_obs_influence_radius', default_value='2.2'))
    args.append(DeclareLaunchArgument('orca_enable_after_z', default_value='-0.6'))
    args.append(DeclareLaunchArgument('orca_use_world_state', default_value='0'))

    # Path planner
    args.append(DeclareLaunchArgument('path_planner_mode', default_value='astar'))
    args.append(DeclareLaunchArgument('astar_grid_resolution', default_value='1.0'))
    args.append(DeclareLaunchArgument('astar_smooth_enable', default_value='1'))

    # Mission
    args.append(DeclareLaunchArgument('mission_start_x', default_value='-5.0'))
    args.append(DeclareLaunchArgument('duration', default_value='32'))
    args.append(DeclareLaunchArgument('eval_mode', default_value='full'))
    args.append(DeclareLaunchArgument('mission_mode', default_value='wall_follow'))

    # PX4 safety
    args.append(DeclareLaunchArgument('px4_disable_gcs_arm_check', default_value='1'))

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)

    def launch_setup(context):
        actions = []

        # Read all parameters
        num_drones = int(LaunchConfiguration('num_drones').perform(context))
        gz_world = LaunchConfiguration('gz_world').perform(context)
        start_port = int(LaunchConfiguration('start_port').perform(context))
        px4_sys_id_base = int(LaunchConfiguration('px4_sys_id_base').perform(context))
        px4_dir = LaunchConfiguration('px4_dir').perform(context)
        ws_root = LaunchConfiguration('ws_root').perform(context)
        swarm_mode = LaunchConfiguration('swarm_mode').perform(context)
        avoid_scope = LaunchConfiguration('avoid_scope').perform(context)
        leader_id = int(LaunchConfiguration('leader_id').perform(context))
        px4_disable_gcs = LaunchConfiguration('px4_disable_gcs_arm_check').perform(context)

        # Derived
        wall_x = float(LaunchConfiguration('dynamic_obs_wall_x').perform(context))
        wall_y = float(LaunchConfiguration('dynamic_obs_wall_y').perform(context))
        target_y_spacing = float(LaunchConfiguration('dynamic_obs_target_y_spacing').perform(context))
        target_y_base = float(LaunchConfiguration('dynamic_obs_target_y_base').perform(context))
        start_wall_clearance = float(LaunchConfiguration('start_wall_clearance').perform(context))
        mission_start_x = wall_x - start_wall_clearance

        warmup_sec = float(LaunchConfiguration('warmup_sec').perform(context))
        post_offboard_hold = float(LaunchConfiguration('post_offboard_hold_sec').perform(context))
        warmup_run_sec = int(warmup_sec + post_offboard_hold + 1)

        log_dir = os.path.join(ws_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        # PX4 post config
        if px4_disable_gcs == '1':
            px4_post_dir = os.path.join(
                px4_dir, 'build/px4_sitl_default/etc/init.d-posix/airframes')
            os.makedirs(px4_post_dir, exist_ok=True)
            px4_post_file = os.path.join(px4_post_dir, '4001_gz_x500.post')
            with open(px4_post_file, 'w') as f:
                f.write('#!/bin/sh\n# Simulation-only: allow offboard tests.\n')
                f.write('param set NAV_DLL_ACT 0\n')
            print(f'[INFO] PX4 post config: {px4_post_file}')

        # ===== 1. MicroXRCEAgent per drone =====
        for i in range(1, num_drones + 1):
            port = start_port + i * 10
            actions.append(ExecuteProcess(
                cmd=['MicroXRCEAgent', 'udp4', '-p', str(port)],
                output='screen',
                name=f'microxrceagent_{i}',
            ))

        # ===== 2. PX4 SITL per drone (staggered) =====
        for i in range(1, num_drones + 1):
            dds_port = start_port + i * 10
            spawn_ned_x = mission_start_x
            spawn_ned_y = wall_y + ((i - leader_id) * target_y_spacing)
            spawn_enu_x = spawn_ned_y
            spawn_enu_y = spawn_ned_x

            env_str = (
                f'PX4_PARAM_CBRK_SUPPLY_CHK=894415 '
            )
            if px4_disable_gcs == '1':
                env_str += 'PX4_PARAM_NAV_DLL_ACT=0 '
            env_str += (
                f'PX4_UXRCE_DDS_PORT={dds_port} '
                f'PX4_GZ_WORLD={gz_world} '
            )
            if i > 1:
                env_str += 'PX4_GZ_STANDALONE=1 '
            env_str += (
                f'PX4_GZ_MODEL_POSE="{spawn_enu_x},{spawn_enu_y}" '
                f'PX4_SYS_AUTOSTART=4001 '
                f'PX4_SIM_MODEL=gz_x500 '
                f'{px4_dir}/build/px4_sitl_default/bin/px4 -i {i}'
            )

            delay = 3.0 + 7.0 * (i - 1)  # 3s for agents + 7s stagger
            actions.append(TimerAction(
                period=delay,
                actions=[ExecuteProcess(
                    cmd=['bash', '-c', env_str],
                    output='screen',
                    name=f'px4_{i}',
                )],
            ))

        # ===== 3. Communication nodes (after PX4 ready) =====
        px4_ready_delay = 3.0 + 7.0 * max(0, num_drones - 1) + 15.0

        for i in range(1, num_drones + 1):
            px4_ns = f'px4_{i}'
            xtd2_ns = f'x500_{i}'
            target_sys_id = px4_sys_id_base + i - 1

            # Use px4_ns as namespace so DDS topics align with PX4
            comm_args = [
                '--model', 'gz_x500',
                '--id', str(i),
                '--namespace', px4_ns,  # DDS: /px4_1/fmu/*
            ]

            actions.append(TimerAction(
                period=px4_ready_delay,
                actions=[Node(
                    package='xtd2_communication',
                    executable='multirotor_communication',
                    arguments=comm_args,
                    remappings=_comm_remappings(px4_ns, xtd2_ns),
                    output='screen',
                    name=f'comm_{i}',
                )],
            ))

        # ===== 4. Swarm state exchange =====
        state_delay = px4_ready_delay + 5.0
        actions.append(TimerAction(
            period=state_delay,
            actions=[Node(
                package='xtd2_mission',
                executable='swarm_state_exchange',
                arguments=[
                    '--num-drones', str(num_drones),
                    '--rate', '15',
                    '--spawned-formation', LaunchConfiguration('use_spawned_formation'),
                    '--mission-start-x', str(mission_start_x),
                    '--base-y', str(wall_y),
                    '--y-spacing', str(target_y_spacing),
                    '--leader-id', str(leader_id),
                ],
                output='screen',
                name='swarm_state_exchange',
            )],
        ))

        # ===== 5. Local ORCA (hybrid mode only) =====
        if swarm_mode == 'hybrid':
            actions.append(TimerAction(
                period=state_delay,
                actions=[Node(
                    package='xtd2_mission',
                    executable='local_avoid_orca',
                    arguments=[
                        '--num-drones', str(num_drones),
                        '--safe-radius', LaunchConfiguration('orca_safe_radius'),
                        '--max-speed', LaunchConfiguration('orca_max_speed'),
                        '--gain', LaunchConfiguration('orca_gain'),
                        '--obs-gain', LaunchConfiguration('orca_obs_gain'),
                        '--obs-influence-radius', LaunchConfiguration('orca_obs_influence_radius'),
                        '--enable-after-z', LaunchConfiguration('orca_enable_after_z'),
                        '--use-world-state', LaunchConfiguration('orca_use_world_state'),
                    ],
                    output='screen',
                    name='local_avoid_orca',
                )],
            ))

            # ===== 6. Dynamic obstacle source =====
            dynamic_obs_enable = LaunchConfiguration('dynamic_obs_enable').perform(context)
            if dynamic_obs_enable == '1':
                obs_args = [
                    '--rate', '10',
                    '--world', gz_world,
                    '--visualize-gz', LaunchConfiguration('dynamic_obs_visualize_gz'),
                    '--mode', LaunchConfiguration('dynamic_obs_mode'),
                    '--obstacle-config', LaunchConfiguration('obstacle_config'),
                    '--scene-config', LaunchConfiguration('scene_config'),
                    '--wall-x', LaunchConfiguration('dynamic_obs_wall_x'),
                    '--wall-y', LaunchConfiguration('dynamic_obs_wall_y'),
                    '--wall-z', LaunchConfiguration('dynamic_obs_wall_z'),
                    '--wall-length', LaunchConfiguration('dynamic_obs_wall_length'),
                    '--wall-thickness', LaunchConfiguration('dynamic_obs_wall_thickness'),
                    '--wall-height', LaunchConfiguration('dynamic_obs_wall_height'),
                    '--wall-segment-spacing', LaunchConfiguration('dynamic_obs_wall_segment_spacing'),
                    '--second-wall-enable', LaunchConfiguration('second_wall_enable'),
                    '--second-wall-dx', LaunchConfiguration('second_wall_dx'),
                    '--second-wall-dy', LaunchConfiguration('second_wall_dy'),
                    '--rear-wall-length', LaunchConfiguration('rear_wall_length'),
                    '--third-wall-enable', LaunchConfiguration('third_wall_enable'),
                    '--target-ball-enable', '1',
                    '--target-ball-num-drones', str(num_drones),
                    '--target-ball-leader-id', str(leader_id),
                    '--target-ball-target-x', LaunchConfiguration('dynamic_obs_target_x'),
                    '--target-ball-base-y', LaunchConfiguration('dynamic_obs_target_y_base'),
                    '--target-ball-target-z', LaunchConfiguration('mission_z'),
                    '--target-ball-y-spacing', str(target_y_spacing),
                ]
                actions.append(TimerAction(
                    period=state_delay,
                    actions=[Node(
                        package='xtd2_mission',
                        executable='dynamic_obstacle_source',
                        arguments=obs_args,
                        output='screen',
                        name='dynamic_obstacle_source',
                    )],
                ))

        # ===== 7. Mission controller =====
        mission_delay = state_delay + 2.0
        namespaces = [f'x500_{i}' for i in range(1, num_drones + 1)]
        # mission_controller uses px4_ns for service calls (because comm node
        # service lives at /xtdrone2/{px4_ns}/cmd — no remapping for services)
        px4_namespaces = [f'px4_{i}' for i in range(1, num_drones + 1)]

        mission_params = {
            'num_drones': num_drones,
            'namespaces': namespaces,           # warmup/mission topics 用 x500_i
            'px4_namespaces': px4_namespaces,   # service calls 用 px4_i (comm service 不可 remap)
            'warmup_sec': warmup_sec,
            'takeoff_z': float(LaunchConfiguration('takeoff_z').perform(context)),
            'vertical_takeoff': LaunchConfiguration('vertical_takeoff').perform(context) == '1',
            'arm_offboard_max_retries': int(LaunchConfiguration('arm_offboard_max_retries').perform(context)),
            'arm_retry_sleep': float(LaunchConfiguration('arm_retry_sleep').perform(context)),
            'offboard_retry_sleep': float(LaunchConfiguration('offboard_retry_sleep').perform(context)),
            'arm_to_offboard_delay': float(LaunchConfiguration('arm_to_offboard_delay').perform(context)),
            'inter_drone_gap': float(LaunchConfiguration('inter_drone_gap').perform(context)),
            'post_offboard_hold_sec': post_offboard_hold,
            'vehicle_status_timeout_sec': float(LaunchConfiguration('vehicle_status_timeout_sec').perform(context)),
            'auto_arm_offboard': LaunchConfiguration('auto_arm_offboard').perform(context) == '1',
        }

        # Export env vars needed by multi_waypoint2
        env_exports = {
            'PATH_PLANNER_MODE': LaunchConfiguration('path_planner_mode').perform(context),
            'ASTAR_GRID_RESOLUTION': LaunchConfiguration('astar_grid_resolution').perform(context),
            'ASTAR_SMOOTH_ENABLE': LaunchConfiguration('astar_smooth_enable').perform(context),
            'VERTICAL_TAKEOFF': LaunchConfiguration('vertical_takeoff').perform(context),
            'SECOND_WALL_ENABLE': LaunchConfiguration('second_wall_enable').perform(context),
            'SECOND_WALL_DX': LaunchConfiguration('second_wall_dx').perform(context),
            'REAR_WALL_GAP': LaunchConfiguration('rear_wall_gap').perform(context),
            'REAR_WALL_LENGTH': LaunchConfiguration('rear_wall_length').perform(context),
            'SECOND_WALL_DY': LaunchConfiguration('second_wall_dy').perform(context),
            'THIRD_WALL_ENABLE': LaunchConfiguration('third_wall_enable').perform(context),
            'OBSTACLE_CONFIG': LaunchConfiguration('obstacle_config').perform(context),
            'DYNAMIC_OBS_MODE': LaunchConfiguration('dynamic_obs_mode').perform(context),
        }

        # Build command that sets env vars then runs mission_controller
        env_prefix = ' '.join(f'{k}={v}' for k, v in env_exports.items())
        mc_cmd = (
            f'cd {ws_root} && '
            f'source /opt/ros/jazzy/setup.bash && '
            f'source install/setup.bash && '
            f'{env_prefix} '
            f'ros2 run xtd2_mission mission_controller '
            f'--ros-args '
        )
        for k, v in mission_params.items():
            if isinstance(v, list):
                items = ' '.join(str(x) for x in v)
                mc_cmd += f'-p {k}:=[{items}] '
            elif isinstance(v, bool):
                mc_cmd += f'-p {k}:={"true" if v else "false"} '
            else:
                mc_cmd += f'-p {k}:={v} '

        actions.append(TimerAction(
            period=mission_delay,
            actions=[ExecuteProcess(
                cmd=['bash', '-c', mc_cmd],
                output='screen',
                name='mission_controller',
                additional_env=env_exports,
            )],
        ))

        return actions

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
