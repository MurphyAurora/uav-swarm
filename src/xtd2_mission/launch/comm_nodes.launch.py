#!/usr/bin/env python3
"""
Communication Nodes Launch

Starts multirotor_communication nodes for each drone.
Uses ExecuteProcess (not Node) so each subprocess sources setup.bash
independently.

Namespace mapping (without modifying XTDrone2 source):
  - comm node launched with --namespace px4_{i} -> DDS topics align with PX4
  - XTDrone2 topics remapped via ros-args -r from /xtdrone2/px4_{i}/* -> /xtdrone2/x500_{i}/*

Usage:
  ros2 launch xtd2_mission comm_nodes.launch.py num_drones:=5
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


def _comm_remappings(px4_ns, xtd2_ns):
    """Generate ros-args remapping string: /xtdrone2/{px4_ns}/* -> /xtdrone2/{xtd2_ns}/*."""
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


def generate_launch_description():

    _pkg_dir = get_package_share_directory('xtd2_mission')
    _ws_root = os.path.abspath(os.path.join(_pkg_dir, '..', '..', '..', '..'))

    args = []
    args.append(DeclareLaunchArgument('num_drones', default_value='5'))
    args.append(DeclareLaunchArgument('px4_sys_id_base', default_value='2'))
    args.append(DeclareLaunchArgument('start_delay', default_value='0'))
    args.append(DeclareLaunchArgument('ws_root',
        default_value=_ws_root))

    ld = LaunchDescription()
    for a in args:
        ld.add_action(a)

    def launch_setup(context):
        actions = []

        num_drones = int(LaunchConfiguration('num_drones').perform(context))
        px4_sys_id_base = int(LaunchConfiguration('px4_sys_id_base').perform(context))
        start_delay = float(LaunchConfiguration('start_delay').perform(context))
        ws_root = LaunchConfiguration('ws_root').perform(context)

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

            actions.append(TimerAction(
                period=start_delay,
                actions=[ExecuteProcess(
                    cmd=['bash', '-c', cmd_str],
                    output='log',
                    name=f'comm_{i}',
                )],
            ))

        return actions

    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld
