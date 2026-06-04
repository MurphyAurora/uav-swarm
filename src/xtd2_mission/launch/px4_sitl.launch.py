#!/usr/bin/env python3
"""
PX4 SITL + MicroXRCEAgent Launch

Starts MicroXRCEAgent and PX4 SITL instances.
These are plain processes (no ROS nodes), so RMW is irrelevant.

Usage:
  ros2 launch xtd2_mission px4_sitl.launch.py num_drones:=5
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    _pkg_dir = get_package_share_directory('xtd2_mission')
    _ws_root = os.path.abspath(os.path.join(_pkg_dir, '..', '..', '..', '..'))

    args = []
    args.append(DeclareLaunchArgument('num_drones', default_value='5'))
    args.append(DeclareLaunchArgument('gz_world', default_value='default'))
    args.append(DeclareLaunchArgument('start_port', default_value='8888'))
    args.append(DeclareLaunchArgument('px4_sys_id_base', default_value='2'))
    args.append(DeclareLaunchArgument('px4_dir',
        default_value=os.path.join(_ws_root, 'firmware', 'PX4-Autopilot')))
    args.append(DeclareLaunchArgument('px4_disable_gcs_arm_check', default_value='1'))
    args.append(DeclareLaunchArgument('leader_id', default_value='3'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_x', default_value='3.0'))
    args.append(DeclareLaunchArgument('dynamic_obs_wall_y', default_value='1.0'))
    args.append(DeclareLaunchArgument('dynamic_obs_target_y_spacing', default_value='1.8'))
    args.append(DeclareLaunchArgument('start_wall_clearance', default_value='8.0'))

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)

    def launch_setup(context):
        actions = []

        num_drones = int(LaunchConfiguration('num_drones').perform(context))
        gz_world = LaunchConfiguration('gz_world').perform(context)
        start_port = int(LaunchConfiguration('start_port').perform(context))
        px4_dir = LaunchConfiguration('px4_dir').perform(context)
        px4_disable_gcs = LaunchConfiguration('px4_disable_gcs_arm_check').perform(context)
        leader_id = int(LaunchConfiguration('leader_id').perform(context))
        wall_x = float(LaunchConfiguration('dynamic_obs_wall_x').perform(context))
        wall_y = float(LaunchConfiguration('dynamic_obs_wall_y').perform(context))
        target_y_spacing = float(LaunchConfiguration('dynamic_obs_target_y_spacing').perform(context))
        start_wall_clearance = float(LaunchConfiguration('start_wall_clearance').perform(context))
        mission_start_x = wall_x - start_wall_clearance

        # PX4 post config (disable GCS arm check)
        if px4_disable_gcs == '1':
            px4_post_dir = os.path.join(
                px4_dir, 'build/px4_sitl_default/etc/init.d-posix/airframes')
            os.makedirs(px4_post_dir, exist_ok=True)
            px4_post_file = os.path.join(px4_post_dir, '4001_gz_x500.post')
            with open(px4_post_file, 'w') as f:
                f.write('#!/bin/sh\n# Simulation-only: allow offboard tests.\n')
                f.write('param set NAV_DLL_ACT 0\n')
            print(f'[INFO] PX4 post config: {px4_post_file}')

        # ===== MicroXRCEAgent per drone =====
        for i in range(1, num_drones + 1):
            port = start_port + i * 10
            actions.append(ExecuteProcess(
                cmd=['MicroXRCEAgent', 'udp4', '-p', str(port)],
                output='log',
                name=f'microxrceagent_{i}',
            ))

        # ===== PX4 SITL per drone (staggered) =====
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
                    output='screen',
                    name=f'px4_{i}',
                )],
            ))

        return actions

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
