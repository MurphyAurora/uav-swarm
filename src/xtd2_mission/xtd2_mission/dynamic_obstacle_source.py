#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 3 层：ROS/Gazebo 动态障碍发布节点。

输入：
    旧版 obstacles.yaml，或 scenes/*.yaml 场景文件。
输出：
    /xtdrone2/swarm/dynamic_obstacles 和 /xtdrone2/swarm/target_markers。
    /xtdrone2/swarm/dynamic_obstacle_markers 和 /xtdrone2/swarm/target_markers_rviz 可直接给 RViz 显示。

在 scene 模式下，它把可复现 YAML 场景实时转换为避障算法消费的 ROS topic，
并为每个命名动态障碍同步一个 Gazebo 可视化球。
"""

import json
import math
import argparse
import csv
import time
import os
import tempfile
import subprocess
import shutil
import xml.etree.ElementTree as ET
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
try:
    from xtd2_mission.obstacle_config import (
        WallObstacle,
        load_obstacle_config,
        normalized_target,
        normalized_walls,
        obstacles_from_config,
    )
except ImportError:
    from obstacle_config import (
        WallObstacle,
        load_obstacle_config,
        normalized_target,
        normalized_walls,
        obstacles_from_config,
    )

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is expected in the ROS env.
    yaml = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAZEBO_ASSET_DIR = os.path.join(SCRIPT_DIR, 'gazebo_assets')


def _finite_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _vector3(value, default=(0.0, 0.0, 0.0)):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return tuple(float(v) for v in default)
    return tuple(_finite_float(v) for v in value)


def _resolve_path(path):
    expanded = os.path.expandvars(os.path.expanduser(str(path or '')))
    if os.path.isabs(expanded):
        return expanded
    if os.path.exists(expanded):
        return expanded
    return os.path.join(SCRIPT_DIR, expanded)


def _asset_path(filename):
    return os.path.join(GAZEBO_ASSET_DIR, filename)


class DynamicObstacleSource(Node):
    def __init__(
        self,
        rate_hz: float = 10.0,
        world_name: str = 'default',
        visualize_gz: bool = True,
        mode: str = 'moving_ball',
        start_center_x: float = 18.0,
        start_center_y: float = 6.0,
        active_center_x: float = 5.5,
        active_center_y: float = 0.0,
        switch_after_sec: float = 38.0,
        switch_fade_sec: float = 6.0,
    ):
        super().__init__('dynamic_obstacle_source')
        self.pub = self.create_publisher(String, '/xtdrone2/swarm/dynamic_obstacles', 10)
        self.target_pub = self.create_publisher(String, '/xtdrone2/swarm/target_markers', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/xtdrone2/swarm/dynamic_obstacle_markers', 10)
        self.target_marker_pub = self.create_publisher(MarkerArray, '/xtdrone2/swarm/target_markers_rviz', 10)
        self.drone_marker_pub = self.create_publisher(MarkerArray, '/xtdrone2/swarm/drone_markers_rviz', 10)
        self.world_name = world_name
        self.visualize_gz = bool(visualize_gz)
        self.mode = str(mode)
        self.entity_name = 'dynamic_obstacle_ball'
        self.scene_entity_name = 'scene_dynamic_obstacle_1'
        self.wall_entity_name = 'static_obstacle_wall'
        self.target_ball_entity_name = 'target_ball'

        # 参数可运行中修改（阶段2热更新配套）
        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 1.0)
        self.declare_parameter('start_center_x', float(start_center_x))
        self.declare_parameter('start_center_y', float(start_center_y))
        self.declare_parameter('active_center_x', float(active_center_x))
        self.declare_parameter('active_center_y', float(active_center_y))
        self.declare_parameter('switch_after_sec', float(switch_after_sec))
        self.declare_parameter('switch_fade_sec', float(switch_fade_sec))
        # wall_x / wall_y are defined in swarm-control NED frame.
        self.declare_parameter('wall_x', 5.5)
        self.declare_parameter('wall_y', 0.0)
        self.declare_parameter('wall_z', 0.9)
        self.declare_parameter('wall_length', 5.0)
        self.declare_parameter('wall_thickness', 0.25)
        self.declare_parameter('wall_height', 1.8)
        self.declare_parameter('wall_segment_spacing', 0.6)
        self.declare_parameter('obstacle_config', '')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('second_wall_enable', 1)
        self.declare_parameter('second_wall_dx', 15.0)
        self.declare_parameter('second_wall_dy', 109.0)
        self.declare_parameter('rear_wall_length', 200.0)
        self.declare_parameter('third_wall_enable', 1)
        self.declare_parameter('target_ball_enable', 1)
        self.declare_parameter('target_ball_num_drones', 3)
        self.declare_parameter('target_ball_leader_id', 1)
        self.declare_parameter('target_ball_target_x', 20.0)
        self.declare_parameter('target_ball_target_z', -3.0)
        self.declare_parameter('target_ball_base_y', 0.0)
        self.declare_parameter('target_ball_y_spacing', 1.8)
        self.declare_parameter('amplitude_x', 1.2)
        self.declare_parameter('amplitude_y', 0.6)
        self.declare_parameter('omega', 0.5)
        self.declare_parameter('radius', 0.35)
        self.declare_parameter('z', 0.6)
        self.declare_parameter('visual_filter_alpha', 0.35)
        self.declare_parameter('visual_max_step', 0.15)
        self.declare_parameter('trajectory_record_path', '')
        self.declare_parameter('trajectory_record_dt', 0.0)
        self.declare_parameter('scene_clock_mode', 'wall_time')
        self.declare_parameter('scene_freeze_dynamics', 0)
        self.declare_parameter('scene_start_num_drones', 0)
        self.declare_parameter('scene_start_z_threshold', -2.0)
        self.declare_parameter('scene_start_stable_sec', 2.0)
        self.declare_parameter('rviz_markers_enable', 1)
        self.declare_parameter('rviz_marker_frame', 'map')
        self.declare_parameter('rviz_drone_markers_enable', 1)
        self.declare_parameter('rviz_drone_path_enable', 1)
        self.declare_parameter('rviz_drone_path_max_points', 450)

        self.t0 = time.time()
        self.timer = self.create_timer(max(0.05, 1.0 / rate_hz), self._publish)
        self.create_subscription(String, '/xtdrone2/swarm/state_exchange', self._swarm_state_cb, 10)
        self._last_log_t = 0.0
        self._last_pose_set_t = 0.0
        self._last_pose_warn_t = 0.0
        self._last_probe_t = 0.0
        self._last_probe_warn_t = 0.0
        self._last_frame_check_log_t = 0.0
        self._spawn_done = False
        self._gz_set_pose_service = f'/world/{self.world_name}/set_pose'
        self._gz_create_service = f'/world/{self.world_name}/create'
        self._gz_ready = False
        self._visual_last_ok_x = None
        self._visual_last_ok_y = None
        self._visual_last_ok_z = None
        self._switch_logged = False
        self._wall_spawned = False
        self._target_ball_spawned = False
        self._scene_cache_path = ''
        self._scene_cache_mtime = None
        self._scene_cache = {}
        self._scene_sdf_cache = {}
        self._scene_warned_no_config = False
        self._scene_warned_unsupported = set()
        self._scene_visual_spawned = set()
        self._last_trajectory_record_t = None
        self._latest_swarm_states = {}
        self._drone_paths = {}
        self._takeoff_ready_since = None
        self._scene_clock_started = False
        self._scene_clock_start_wall_t = None
        self._last_scene_clock_log_t = 0.0
        self.get_logger().info(
            'dynamic_obstacle_source started, publishing /xtdrone2/swarm/dynamic_obstacles, '
            f'visualize_gz={self.visualize_gz}, world={self.world_name}, mode={self.mode}'
        )

        # 注意：目标球/墙体的初始位姿依赖运行参数。
        # 不要在 __init__ 里立刻 spawn，否则会先用 declare_parameter 的默认值创建，
        # 再被 main() 的 set_parameters 覆盖，导致 Gazebo 视觉实体与发布出去的 marker 不一致。

    def _log_scene_clock_config_once(self):
        if getattr(self, '_scene_clock_config_logged', False):
            return
        self._scene_clock_config_logged = True
        self.get_logger().info(
            'scene clock config: '
            f'mode={self._scene_clock_mode()}, '
            f'start_num_drones={int(self.get_parameter("scene_start_num_drones").value)}, '
            f'z_threshold={float(self.get_parameter("scene_start_z_threshold").value):.2f}, '
            f'stable_sec={float(self.get_parameter("scene_start_stable_sec").value):.1f}'
        )

    @staticmethod
    def _enu_to_ned(x_enu: float, y_enu: float, z_enu: float):
        # Gazebo world frame is ENU; swarm control stack uses NED.
        # ENU(E, N, U) -> NED(N, E, D)
        return float(y_enu), float(x_enu), float(-z_enu)

    @staticmethod
    def _enu_vel_to_ned(vx_enu: float, vy_enu: float, vz_enu: float):
        return float(vy_enu), float(vx_enu), float(-vz_enu)

    @staticmethod
    def _ned_to_enu(x_ned: float, y_ned: float, z_ned: float):
        # NED(N, E, D) -> ENU(E, N, U)
        return float(y_ned), float(x_ned), float(-z_ned)

    def _run_gz_service(self, service: str, reqtype: str, req: str, timeout_ms: int = 400) -> bool:
        cmd = [
            'gz',
            'service',
            '-s',
            service,
            '--reqtype',
            reqtype,
            '--reptype',
            'gz.msgs.Boolean',
            '--timeout',
            str(timeout_ms),
            '--req',
            req,
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(0.2, timeout_ms / 1000.0 + 0.3))
            out = ((p.stdout or '') + (p.stderr or '')).lower()
            return p.returncode == 0 and ('data: true' in out or 'true' in out)
        except Exception:
            return False

    def _setup_gz_visual(self):
        if not shutil.which('gz'):
            self.get_logger().warn('gz command not found, skip visual obstacle sync')
            return
        try:
            p = subprocess.run(['gz', 'service', '-l'], capture_output=True, text=True, timeout=4.0)
            if p.returncode != 0 or self._gz_set_pose_service not in (p.stdout or ''):
                now = time.time()
                if now - self._last_probe_warn_t > 2.0:
                    self._last_probe_warn_t = now
                    self.get_logger().warn('gazebo transport set_pose service unavailable, keep retrying...')
                return
            self._gz_ready = True
            self.get_logger().info('gazebo transport service detected, visual sync enabled')
            self._spawn_visual_obstacle()
        except Exception as e:
            now = time.time()
            if now - self._last_probe_warn_t > 2.0:
                self._last_probe_warn_t = now
                self.get_logger().warn(f'gazebo service probe failed, keep retrying: {e}')

    def _spawn_visual_obstacle(self):
        if self.mode == 'static_wall':
            sdf_path = self._prepare_wall_sdf()
            entity_name = self.wall_entity_name
        elif self.mode == 'scene':
            self._spawn_scene_visual_obstacles()
            return
        else:
            sdf_path = _asset_path('dynamic_obstacle_ball.sdf')
            entity_name = self._visual_obstacle_entity_name()
        if not os.path.exists(sdf_path):
            self.get_logger().warn(f'visual obstacle sdf not found: {sdf_path}')
            return
        if self.mode == 'static_wall':
            base_wall = self._configured_walls()[0]
            wall_x_ned = float(base_wall['x'])
            wall_y_ned = float(base_wall['y'])
            wall_z = float(base_wall['z'])
            wall_x_enu, wall_y_enu, _ = self._ned_to_enu(wall_x_ned, wall_y_ned, 0.0)
            req = (
                f'name: "{entity_name}", '
                f'allow_renaming: false, '
                f'sdf_filename: "{sdf_path}", '
                f'pose: {{ position: {{ x: {wall_x_enu:.3f}, y: {wall_y_enu:.3f}, z: {wall_z:.3f} }}, '
                f'orientation: {{ z: 0.7071068, w: 0.7071068 }} }}'
            )
            ok = self._run_gz_service(self._gz_create_service, 'gz.msgs.EntityFactory', req, timeout_ms=1200)
            self._spawn_done = bool(ok)
            self._wall_spawned = self._spawn_done
            if self._spawn_done:
                self._visual_last_ok_x = wall_x_enu
                self._visual_last_ok_y = wall_y_enu
                self._visual_last_ok_z = wall_z
                self.get_logger().info(
                    'gazebo static wall spawned successfully (gz transport), '
                    f'NED=({wall_x_ned:.2f},{wall_y_ned:.2f}), ENU=({wall_x_enu:.2f},{wall_y_enu:.2f})'
                )
                self._spawn_target_ball()
            else:
                self.get_logger().warn('gazebo static wall spawn failed (gz transport)')
            return

        # 先尝试直接设置位姿；若实体已存在即可立即联动
        if self._set_visual_pose(0.0, 0.0, float(self._p('z')), force=True):
            self._spawn_done = True
            self._visual_last_ok_x = 0.0
            self._visual_last_ok_y = 0.0
            self._visual_last_ok_z = float(self._p('z'))
            self.get_logger().info(f'gazebo visual obstacle already exists, pose sync enabled: entity={entity_name}')
            return

        req = (
            f'name: "{entity_name}", '
            f'allow_renaming: false, '
            f'sdf_filename: "{sdf_path}", '
            f'pose: {{ position: {{ x: 0.0, y: 0.0, z: {float(self._p("z")):.3f} }}, orientation: {{ w: 1.0 }} }}'
        )
        ok = self._run_gz_service(self._gz_create_service, 'gz.msgs.EntityFactory', req, timeout_ms=1200)
        self._spawn_done = bool(ok)
        if self._spawn_done:
            self.get_logger().info(f'gazebo visual obstacle spawned successfully (gz transport): entity={entity_name}')
        else:
            self.get_logger().warn(f'gazebo visual obstacle spawn failed (gz transport): entity={entity_name}')

    def _scene_dynamic_visual_items(self, payload=None):
        if payload is not None:
            dynamic = [
                obs for obs in payload.get('obstacles', [])
                if str(obs.get('name', '')).strip()
            ]
            return dynamic

        scene = self._load_scene_config()
        items = []
        t = 0.0
        for idx, obs in enumerate(scene.get('dynamic_obstacles') or [], start=1):
            if not isinstance(obs, dict):
                continue
            name = str(obs.get('name', f'dynamic_obstacle_{idx}')).strip()
            if not name:
                continue
            x, y, z, vx, vy, vz = self._scene_dynamic_state(obs, t)
            items.append(
                {
                    'obs_id': idx,
                    'name': name,
                    'x': x,
                    'y': y,
                    'z': z,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'radius': max(0.05, _finite_float(obs.get('radius'), 0.5)),
                }
            )
        return items

    def _spawn_scene_visual_obstacles(self, payload=None):
        if not self._gz_ready:
            return False
        spawned_any = False
        for idx, obs in enumerate(self._scene_dynamic_visual_items(payload), start=1):
            entity_name = self._scene_visual_entity_name(obs, idx)
            if entity_name in self._scene_visual_spawned:
                continue
            enu_x, enu_y, enu_z = self._ned_to_enu(obs.get('x', 0.0), obs.get('y', 0.0), obs.get('z', 0.0))
            sdf_path = self._prepare_scene_ball_sdf(entity_name, obs.get('radius', 0.5), idx)
            req = (
                f'name: "{entity_name}", '
                f'allow_renaming: false, '
                f'sdf_filename: "{sdf_path}", '
                f'pose: {{ position: {{ x: {enu_x:.3f}, y: {enu_y:.3f}, z: {enu_z:.3f} }}, '
                f'orientation: {{ w: 1.0 }} }}'
            )
            ok = self._run_gz_service(self._gz_create_service, 'gz.msgs.EntityFactory', req, timeout_ms=1200)
            if ok or self._set_entity_pose(entity_name, enu_x, enu_y, enu_z, force=True):
                self._scene_visual_spawned.add(entity_name)
                spawned_any = True
                self.get_logger().info(
                    f'gazebo scene obstacle visual enabled: entity={entity_name}, '
                    f'ENU=({enu_x:.2f},{enu_y:.2f},{enu_z:.2f})'
                )
            else:
                self.get_logger().warn(f'gazebo scene obstacle spawn failed: entity={entity_name}, sdf={sdf_path}')
        self._spawn_done = bool(self._scene_visual_spawned)
        return spawned_any

    def _sync_scene_visual_obstacles(self, payload):
        if not self._gz_ready:
            return False
        if len(self._scene_visual_spawned) < len(self._scene_dynamic_visual_items(payload)):
            self._spawn_scene_visual_obstacles(payload)

        all_ok = True
        for idx, obs in enumerate(self._scene_dynamic_visual_items(payload), start=1):
            entity_name = self._scene_visual_entity_name(obs, idx)
            enu_x, enu_y, enu_z = self._ned_to_enu(obs.get('x', 0.0), obs.get('y', 0.0), obs.get('z', 0.0))
            ok = self._set_entity_pose(entity_name, enu_x, enu_y, enu_z, force=True)
            all_ok = all_ok and ok
        return all_ok

    def _prepare_wall_sdf(self):
        obstacles = self._configured_obstacles()
        base = obstacles[0]
        link_sdf = ''.join(
            obs.to_sdf_link(base_x=base.x, base_y=base.y, base_z=base.z, index=idx)
            for idx, obs in enumerate(obstacles)
        )
        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="static_obstacle_wall">
    <static>true</static>
{link_sdf}
  </model>
</sdf>
"""
        tmp_path = os.path.join(tempfile.gettempdir(), 'xtd2_static_obstacle_wall_generated.sdf')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(sdf)
        self.get_logger().info(
            f'wall visual sdf prepared from config: path={tmp_path}, obstacles={len(obstacles)}'
        )
        return tmp_path

    def _scene_visual_initial_pose_enu(self):
        scene = self._load_scene_config()
        dynamic = scene.get('dynamic_obstacles') or []
        for obs in dynamic:
            if not isinstance(obs, dict):
                continue
            x, y, z, _, _, _ = self._scene_dynamic_state(obs, 0.0)
            return self._ned_to_enu(x, y, z)
        return 0.0, 0.0, float(self._p('z'))

    def _safe_entity_name(self, value, default='scene_dynamic_obstacle'):
        text = str(value or default).strip().replace(' ', '_')
        return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in text) or default

    def _scene_visual_entity_name(self, obs, index=1):
        raw_name = self._safe_entity_name(obs.get('name'), f'scene_dynamic_obstacle_{index}')
        return f'scene_{raw_name}'

    def _prepare_scene_ball_sdf(self, entity_name=None, radius=0.5, index=1):
        entity_name = self._safe_entity_name(entity_name or self.scene_entity_name, self.scene_entity_name)
        radius = max(0.1, _finite_float(radius, 0.5))
        palette = [
            ('1.00 0.12 0.08 1', '1.00 0.12 0.08 1'),
            ('0.10 0.45 1.00 1', '0.10 0.45 1.00 1'),
            ('1.00 0.78 0.08 1', '1.00 0.78 0.08 1'),
            ('0.10 0.85 0.35 1', '0.10 0.85 0.35 1'),
        ]
        ambient, diffuse = palette[(max(1, int(index)) - 1) % len(palette)]
        sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="{entity_name}">
    <static>true</static>
    <link name="body">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{radius:.3f}</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{ambient}</ambient>
          <diffuse>{diffuse}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        tmp_path = os.path.join(tempfile.gettempdir(), f'xtd2_{entity_name}.sdf')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            f.write(sdf)
        self.get_logger().info(
            f'scene visual sdf prepared: entity={entity_name}, path={tmp_path}, radius={radius:.2f}'
        )
        return tmp_path

    def _set_entity_pose(self, entity_name: str, x: float, y: float, z: float, force: bool = False) -> bool:
        if not self._gz_ready:
            return False
        if not force and not self._spawn_done and self.mode != 'scene':
            return False
        req = (
            f'name: "{entity_name}", '
            f'position: {{ x: {float(x):.3f}, y: {float(y):.3f}, z: {float(z):.3f} }}, '
            f'orientation: {{ w: 1.0 }}'
        )
        return self._run_gz_service(self._gz_set_pose_service, 'gz.msgs.Pose', req, timeout_ms=600)

    def _set_visual_pose(self, x: float, y: float, z: float, force: bool = False) -> bool:
        return self._set_entity_pose(self._visual_obstacle_entity_name(), x, y, z, force=force)

    def _set_target_ball_pose_enu(self, x_enu: float, y_enu: float, z_enu: float) -> bool:
        if not self._gz_ready:
            return False
        req = (
            f'name: "{self.target_ball_entity_name}", '
            f'position: {{ x: {float(x_enu):.3f}, y: {float(y_enu):.3f}, z: {float(z_enu):.3f} }}, '
            f'orientation: {{ w: 1.0 }}'
        )
        return self._run_gz_service(self._gz_set_pose_service, 'gz.msgs.Pose', req, timeout_ms=600)

    def _spawn_target_ball(self):
        if self._target_ball_spawned:
            return
        if int(self.get_parameter('target_ball_enable').value) != 1:
            return

        # 绿色target marker已移除：改为使用红色dynamic_obstacle_ball作为静态目标点。
        sdf_path = _asset_path('dynamic_obstacle_ball.sdf')
        if not os.path.exists(sdf_path):
            self.get_logger().warn(f'target ball sdf not found: {sdf_path}')
            return

        target_cfg = self._configured_target()
        target_x = float(target_cfg['x'])
        target_z = float(target_cfg['z'])
        target_y = float(target_cfg['y'])
        enu_x, enu_y, enu_z = self._ned_to_enu(target_x, target_y, target_z)

        req = (
            f'name: "{self.target_ball_entity_name}", '
            f'allow_renaming: false, '
            f'sdf_filename: "{sdf_path}", '
            f'pose: {{ position: {{ x: {enu_x:.3f}, y: {enu_y:.3f}, z: {enu_z:.3f} }}, orientation: {{ w: 1.0 }} }}'
        )
        ok = self._run_gz_service(self._gz_create_service, 'gz.msgs.EntityFactory', req, timeout_ms=1200)
        self._target_ball_spawned = bool(ok)

        if self._target_ball_spawned:
            self.get_logger().info(
                f'target ball spawned: entity={self.target_ball_entity_name}, '
                f'ENU=({enu_x:.2f},{enu_y:.2f},{enu_z:.2f}), NED=({target_x:.2f},{target_y:.2f},{target_z:.2f})'
            )
        else:
            self.get_logger().warn('target ball spawn failed')

    def _next_visual_pose(self, target_x: float, target_y: float, target_z: float):
        alpha = min(1.0, max(0.0, self._p('visual_filter_alpha')))
        max_step = max(0.01, self._p('visual_max_step'))

        if self._visual_last_ok_x is None:
            return float(target_x), float(target_y), float(target_z)

        dx = float(target_x) - self._visual_last_ok_x
        dy = float(target_y) - self._visual_last_ok_y
        dz = float(target_z) - self._visual_last_ok_z

        # 先做低通，再做限步，减少跳变感
        cand_x = self._visual_last_ok_x + alpha * dx
        cand_y = self._visual_last_ok_y + alpha * dy
        cand_z = self._visual_last_ok_z + alpha * dz

        step_dx = cand_x - self._visual_last_ok_x
        step_dy = cand_y - self._visual_last_ok_y
        step_dz = cand_z - self._visual_last_ok_z
        step_norm = math.sqrt(step_dx * step_dx + step_dy * step_dy + step_dz * step_dz)
        if step_norm > max_step and step_norm > 1e-6:
            s = max_step / step_norm
            cand_x = self._visual_last_ok_x + step_dx * s
            cand_y = self._visual_last_ok_y + step_dy * s
            cand_z = self._visual_last_ok_z + step_dz * s

        return cand_x, cand_y, cand_z

    def _build_target_markers_payload(self):
        if int(self.get_parameter('target_ball_enable').value) != 1:
            return None

        num_drones = max(1, int(self.get_parameter('target_ball_num_drones').value))
        leader_id = int(self.get_parameter('target_ball_leader_id').value)
        target_cfg = self._configured_target()
        target_x = float(target_cfg['x'])
        target_z = float(target_cfg['z'])
        base_y = float(target_cfg['y'])
        spacing = float(target_cfg['y_spacing'])
        axis = str(target_cfg.get('axis', 'y')).strip().lower()

        markers = []
        for i in range(1, num_drones + 1):
            offset = (i - leader_id) * spacing
            if axis == 'x':
                ned_x = target_x + offset
                ned_y = base_y
            else:
                ned_x = target_x
                ned_y = base_y + offset
            ned_z = target_z
            enu_x, enu_y, enu_z = self._ned_to_enu(ned_x, ned_y, ned_z)
            markers.append(
                {
                    'id': i,
                    # Mission/control side consumes x/y/z in NED.
                    'x': ned_x,
                    'y': ned_y,
                    'z': ned_z,
                    # Keep ENU fields for debugging and traceability.
                    'enu_x': enu_x,
                    'enu_y': enu_y,
                    'enu_z': enu_z,
                }
            )

        return {
            'stamp': time.time(),
            'leader_id': leader_id,
            'markers': markers,
        }

    def _log_target_table(self, target_payload, prefix='target-table'):
        markers = target_payload.get('markers', []) if target_payload else []
        if not markers:
            return
        self.get_logger().info(f'{prefix}: begin')
        for m in markers:
            self.get_logger().info(
                f'{prefix}: id={int(m.get("id", -1))} '
                f'NED=({float(m.get("x", 0.0)):.2f},{float(m.get("y", 0.0)):.2f},{float(m.get("z", 0.0)):.2f}) '
                f'ENU=({float(m.get("enu_x", 0.0)):.2f},{float(m.get("enu_y", 0.0)):.2f},{float(m.get("enu_z", 0.0)):.2f})'
            )
        self.get_logger().info(f'{prefix}: end')

    def _p(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _scene_clock_mode(self):
        return str(self.get_parameter('scene_clock_mode').value or 'wall_time').strip().lower()

    def _swarm_state_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        states = payload.get('states', [])
        now = time.time()
        max_points = max(2, int(self.get_parameter('rviz_drone_path_max_points').value))
        for state in states:
            try:
                drone_id = int(state.get('id', -1))
            except Exception:
                continue
            if drone_id <= 0:
                continue
            ned_x = _finite_float(state.get('world_x', state.get('x')), 0.0)
            ned_y = _finite_float(state.get('world_y', state.get('y')), 0.0)
            ned_z = _finite_float(state.get('world_z', state.get('z')), 0.0)
            vx = _finite_float(state.get('vx'), 0.0)
            vy = _finite_float(state.get('vy'), 0.0)
            vz = _finite_float(state.get('vz'), 0.0)
            heading = _finite_float(state.get('heading'), 0.0)
            enu_x, enu_y, enu_z = self._ned_to_enu(ned_x, ned_y, ned_z)
            self._latest_swarm_states[drone_id] = {
                'time': now,
                'x': ned_x,
                'y': ned_y,
                'z': ned_z,
                'vx': vx,
                'vy': vy,
                'vz': vz,
                'heading': heading,
                'enu_x': enu_x,
                'enu_y': enu_y,
                'enu_z': enu_z,
            }
            path = self._drone_paths.setdefault(drone_id, [])
            if not path:
                path.append((enu_x, enu_y, enu_z))
            else:
                last_x, last_y, last_z = path[-1]
                dist = math.sqrt((enu_x - last_x) ** 2 + (enu_y - last_y) ** 2 + (enu_z - last_z) ** 2)
                if dist >= 0.08:
                    path.append((enu_x, enu_y, enu_z))
            if len(path) > max_points:
                del path[:-max_points]

    def _takeoff_ready(self, now):
        expected = int(self.get_parameter('scene_start_num_drones').value)
        if expected <= 0:
            expected = int(self.get_parameter('target_ball_num_drones').value)
        expected = max(1, expected)
        z_threshold = float(self.get_parameter('scene_start_z_threshold').value)
        recent_timeout = 2.0
        ready_ids = [
            drone_id for drone_id, state in self._latest_swarm_states.items()
            if now - state.get('time', 0.0) <= recent_timeout
            and state.get('z', 0.0) <= z_threshold
        ]
        return len(set(ready_ids)) >= expected, len(set(ready_ids)), expected, z_threshold

    def _scene_time(self, wall_t):
        mode = self._scene_clock_mode()
        if mode not in ('auto_takeoff', 'takeoff'):
            return wall_t

        now = time.time()
        if not self._scene_clock_started:
            ready, ready_count, expected, z_threshold = self._takeoff_ready(now)
            stable_sec = max(0.0, float(self.get_parameter('scene_start_stable_sec').value))
            if ready:
                if self._takeoff_ready_since is None:
                    self._takeoff_ready_since = now
                if now - self._takeoff_ready_since >= stable_sec:
                    self._scene_clock_started = True
                    self._scene_clock_start_wall_t = now
                    self.get_logger().info(
                        'scene clock started after takeoff: '
                        f'ready_drones={ready_count}/{expected}, z_threshold={z_threshold:.2f}, '
                        f'stable_sec={stable_sec:.1f}'
                    )
            else:
                self._takeoff_ready_since = None
                if now - self._last_scene_clock_log_t > 2.0:
                    self._last_scene_clock_log_t = now
                    self.get_logger().info(
                        'scene clock waiting for takeoff: '
                        f'ready_drones={ready_count}/{expected}, z_threshold={z_threshold:.2f}'
                    )
            return 0.0

        return max(0.0, now - self._scene_clock_start_wall_t)

    def _visual_obstacle_entity_name(self):
        if self.mode == 'scene':
            return self.scene_entity_name
        return self.entity_name

    def _load_scene_config(self):
        config_path = str(self.get_parameter('scene_config').value or '').strip()
        if not config_path:
            if not self._scene_warned_no_config:
                self._scene_warned_no_config = True
                self.get_logger().warn('scene mode selected but scene_config is empty')
            return {}
        scene_path = _resolve_path(config_path)
        try:
            mtime = os.path.getmtime(scene_path)
        except Exception as e:
            self.get_logger().warn(f'scene config not readable: path={scene_path}, error={e}')
            return {}
        if self._scene_cache_path == scene_path and self._scene_cache_mtime == mtime:
            return self._scene_cache
        if yaml is None:
            raise RuntimeError('PyYAML is required to read scene YAML files')
        with open(scene_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f'scene config must be a mapping: {scene_path}')
        self._scene_cache_path = scene_path
        self._scene_cache_mtime = mtime
        self._scene_cache = data
        meta = data.get('scene') or {}
        dyn = data.get('dynamic_obstacles') or []
        walls = ((data.get('static_obstacles') or {}).get('walls')) or []
        self.get_logger().info(
            f'scene config loaded: path={scene_path}, name={meta.get("name", "unnamed")}, '
            f'dynamic_obstacles={len(dyn)}, static_walls={len(walls)}'
        )
        return data

    def _scene_walls(self, scene):
        walls = ((scene.get('static_obstacles') or {}).get('walls')) or []
        normalized = []
        for idx, wall in enumerate(walls, start=1):
            if not isinstance(wall, dict):
                continue
            normalized.append(
                {
                    'name': str(wall.get('name', f'wall_{idx}')),
                    'x': _finite_float(wall.get('x'), 0.0),
                    'y': _finite_float(wall.get('y'), 0.0),
                    'z': _finite_float(wall.get('z'), 0.9),
                    'length': max(1.0, _finite_float(wall.get('length'), 5.0)),
                    'thickness': max(0.1, _finite_float(wall.get('thickness'), 0.25)),
                    'height': max(0.5, _finite_float(wall.get('height'), 1.8)),
                    'segment_spacing': max(0.2, _finite_float(wall.get('segment_spacing'), 0.6)),
                    'orientation': str(wall.get('orientation', 'y')),
                }
            )
        return normalized

    def _scene_sdf_config(self, scene):
        cfg = scene.get('sdf_structures') or {}
        return cfg if isinstance(cfg, dict) else {}

    def _parse_sdf_structures(self, sdf_path):
        path = _resolve_path(sdf_path)
        if not path or not os.path.exists(path):
            return []
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return []
        cached = self._scene_sdf_cache.get(path)
        if cached and cached.get('mtime') == mtime:
            return cached.get('items', [])

        try:
            root = ET.parse(path).getroot()
        except Exception as e:
            self.get_logger().warn(f'failed to parse sdf structures: path={path}, error={e}')
            return []

        items = []
        for model in root.findall('.//model'):
            name = model.get('name', '')
            if not name or name == 'ground_plane':
                continue
            pose_text = model.findtext('pose', default='0 0 0 0 0 0')
            pose_vals = [float(v) for v in str(pose_text).split()[:6]]
            while len(pose_vals) < 6:
                pose_vals.append(0.0)
            enu_x, enu_y, enu_z, roll, pitch, yaw = pose_vals[:6]

            box = model.find('.//box/size')
            cyl_radius = model.find('.//cylinder/radius')
            cyl_length = model.find('.//cylinder/length')
            if box is not None and box.text:
                size_vals = [float(v) for v in str(box.text).split()[:3]]
                while len(size_vals) < 3:
                    size_vals.append(1.0)
                shape = 'box'
                size = size_vals[:3]
                radius = max(0.15, 0.5 * math.sqrt(size[0] * size[0] + size[1] * size[1]))
            elif cyl_radius is not None and cyl_length is not None:
                r = max(0.05, _finite_float(cyl_radius.text, 0.5))
                length = max(0.1, _finite_float(cyl_length.text, 1.0))
                shape = 'cylinder'
                size = [2.0 * r, 2.0 * r, length]
                radius = r
            else:
                continue

            ned_x, ned_y, ned_z = self._enu_to_ned(enu_x, enu_y, enu_z)
            items.append(
                {
                    'name': name,
                    'shape': shape,
                    'x': ned_x,
                    'y': ned_y,
                    'z': ned_z,
                    'enu_x': enu_x,
                    'enu_y': enu_y,
                    'enu_z': enu_z,
                    'roll': roll,
                    'pitch': pitch,
                    'yaw': yaw,
                    'size': size,
                    'radius': radius,
                }
            )

        self._scene_sdf_cache[path] = {'mtime': mtime, 'items': items}
        self.get_logger().info(f'sdf structures loaded: path={path}, count={len(items)}')
        return items

    def _filter_sdf_items(self, items, include_prefixes=None, exclude_prefixes=None, include_names=None, exclude_names=None):
        include_prefixes = [str(v) for v in (include_prefixes or [])]
        exclude_prefixes = [str(v) for v in (exclude_prefixes or [])]
        include_names = set(str(v) for v in (include_names or []))
        exclude_names = set(str(v) for v in (exclude_names or []))
        selected = []
        for item in items:
            name = str(item.get('name', ''))
            if exclude_names and name in exclude_names:
                continue
            if include_names and name not in include_names:
                continue
            if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
                continue
            if exclude_prefixes and any(name.startswith(prefix) for prefix in exclude_prefixes):
                continue
            selected.append(item)
        return selected

    def _sdf_dynamic_obstacle_items(self, scene):
        cfg = self._scene_sdf_config(scene)
        path = cfg.get('path', '')
        dyn_cfg = cfg.get('dynamic_obstacles') or {}
        if not path or not isinstance(dyn_cfg, dict) or int(dyn_cfg.get('enable', 0)) == 0:
            return []

        items = self._filter_sdf_items(
            self._parse_sdf_structures(path),
            include_prefixes=dyn_cfg.get('include_prefixes') or [],
            exclude_prefixes=dyn_cfg.get('exclude_prefixes') or [],
            include_names=dyn_cfg.get('include_names') or [],
            exclude_names=dyn_cfg.get('exclude_names') or [],
        )
        max_count = int(_finite_float(dyn_cfg.get('max_count'), len(items)))
        selection = str(dyn_cfg.get('selection', 'first')).strip().lower()
        if max_count <= 0:
            items = []
        elif max_count < len(items) and selection in ('even', 'evenly_spaced', 'uniform'):
            if max_count == 1:
                items = [items[len(items) // 2]]
            else:
                step = float(len(items) - 1) / float(max_count - 1)
                items = [items[int(round(i * step))] for i in range(max_count)]
        elif max_count > 0:
            items = items[:max_count]
        return items

    def _sdf_static_marker_items(self, scene):
        cfg = self._scene_sdf_config(scene)
        path = cfg.get('path', '')
        marker_cfg = cfg.get('static_markers') or {}
        if not path or not isinstance(marker_cfg, dict) or int(marker_cfg.get('enable', 0)) == 0:
            return []
        return self._filter_sdf_items(
            self._parse_sdf_structures(path),
            include_prefixes=marker_cfg.get('include_prefixes') or [],
            exclude_prefixes=marker_cfg.get('exclude_prefixes') or [],
            include_names=marker_cfg.get('include_names') or [],
            exclude_names=marker_cfg.get('exclude_names') or [],
        )

    def _sdf_dynamic_state(self, scene, item, index, t):
        cfg = self._scene_sdf_config(scene)
        dyn_cfg = cfg.get('dynamic_obstacles') or {}
        traj = dict(dyn_cfg.get('trajectory') or {})
        traj_type = str(traj.get('type', 'sando_trefoil')).strip().lower()
        phase = _finite_float(traj.get('phase'), 0.0) + _finite_float(traj.get('phase_step'), 0.25) * float(index)
        center = [float(item['x']), float(item['y']), float(item['z'])]
        if traj_type in ('trefoil', 'sando_trefoil'):
            traj.setdefault('center', center)
            traj['phase'] = phase
            return self._scene_trefoil_state(traj, t)
        if traj_type == 'circle':
            traj.setdefault('center', center)
            traj['phase'] = phase
            return self._scene_circle_state(traj, t)
        if traj_type in ('ping_pong_line', 'pingpong_line', 'line_ping_pong'):
            span = _vector3(traj.get('span'), (0.0, 1.5, 0.0))
            traj.setdefault('start', [center[0] - span[0] * 0.5, center[1] - span[1] * 0.5, center[2] - span[2] * 0.5])
            traj.setdefault('end', [center[0] + span[0] * 0.5, center[1] + span[1] * 0.5, center[2] + span[2] * 0.5])
            return self._scene_ping_pong_line_state(traj, t)
        return (center[0], center[1], center[2], 0.0, 0.0, 0.0)

    def _scene_line_state(self, traj, t):
        start_time = _finite_float(traj.get('start_time'), 0.0)
        start = _vector3(traj.get('start'))
        end = _vector3(traj.get('end'), start)
        speed = max(0.0, _finite_float(traj.get('speed'), 0.0))
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= 1e-9 or speed <= 1e-9 or t <= start_time:
            return (*start, 0.0, 0.0, 0.0)
        alpha = min(1.0, (t - start_time) * speed / dist)
        x = start[0] + dx * alpha
        y = start[1] + dy * alpha
        z = start[2] + dz * alpha
        if alpha >= 1.0:
            return (x, y, z, 0.0, 0.0, 0.0)
        return (x, y, z, dx / dist * speed, dy / dist * speed, dz / dist * speed)

    def _scene_ping_pong_line_state(self, traj, t):
        start_time = _finite_float(traj.get('start_time'), 0.0)
        start = _vector3(traj.get('start'))
        end = _vector3(traj.get('end'), start)
        speed = max(0.0, _finite_float(traj.get('speed'), 0.0))
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist <= 1e-9 or speed <= 1e-9 or t <= start_time:
            return (*start, 0.0, 0.0, 0.0)

        active_t = t - start_time
        leg_progress = (active_t * speed / dist) % 2.0
        forward = leg_progress <= 1.0
        alpha = leg_progress if forward else 2.0 - leg_progress
        direction = 1.0 if forward else -1.0
        x = start[0] + dx * alpha
        y = start[1] + dy * alpha
        z = start[2] + dz * alpha
        return (
            x,
            y,
            z,
            direction * dx / dist * speed,
            direction * dy / dist * speed,
            direction * dz / dist * speed,
        )

    def _scene_circle_state(self, traj, t):
        start_time = _finite_float(traj.get('start_time'), 0.0)
        center = _vector3(traj.get('center'))
        radius = max(0.0, _finite_float(traj.get('radius'), 1.0))
        omega = _finite_float(traj.get('angular_speed'), 0.0)
        phase = _finite_float(traj.get('phase'), 0.0)
        active_t = max(0.0, t - start_time)
        theta = phase + omega * active_t
        x = center[0] + radius * math.cos(theta)
        y = center[1] + radius * math.sin(theta)
        z = center[2]
        if t < start_time:
            return (x, y, z, 0.0, 0.0, 0.0)
        vx = -radius * omega * math.sin(theta)
        vy = radius * omega * math.cos(theta)
        return (x, y, z, vx, vy, 0.0)

    def _scene_trefoil_state(self, traj, t):
        start_time = _finite_float(traj.get('start_time'), 0.0)
        center = _vector3(traj.get('center'))
        scale = _vector3(traj.get('scale'), (3.0, 3.0, 1.0))
        phase = _finite_float(traj.get('phase'), 0.0)
        slower = max(0.1, _finite_float(traj.get('slower'), 4.0))
        active_t = max(0.0, t - start_time)
        tt = active_t / slower + phase

        x = scale[0] / 6.0 * (math.sin(tt) + 2.0 * math.sin(2.0 * tt)) + center[0]
        y = scale[1] / 5.0 * (math.cos(tt) - 2.0 * math.cos(2.0 * tt)) + center[1]
        z = scale[2] / 2.0 * (-math.sin(3.0 * tt)) + center[2]
        if t < start_time:
            return (x, y, z, 0.0, 0.0, 0.0)

        inv_slower = 1.0 / slower
        vx = scale[0] / 6.0 * inv_slower * (math.cos(tt) + 4.0 * math.cos(2.0 * tt))
        vy = scale[1] / 5.0 * inv_slower * (-math.sin(tt) + 4.0 * math.sin(2.0 * tt))
        vz = -3.0 * scale[2] / 2.0 * inv_slower * math.cos(3.0 * tt)
        return (x, y, z, vx, vy, vz)

    def _scene_dynamic_state(self, obs, t):
        traj = obs.get('trajectory') or {}
        traj_type = str(traj.get('type', 'line')).strip().lower()
        if traj_type == 'circle':
            return self._scene_circle_state(traj, t)
        if traj_type in ('trefoil', 'sando_trefoil'):
            return self._scene_trefoil_state(traj, t)
        if traj_type in ('ping_pong_line', 'pingpong_line', 'line_ping_pong'):
            return self._scene_ping_pong_line_state(traj, t)
        if traj_type not in ('line', 'static'):
            name = str(obs.get('name', 'dynamic_obstacle'))
            key = (name, traj_type)
            if key not in self._scene_warned_unsupported:
                self._scene_warned_unsupported.add(key)
                self.get_logger().warn(f'unsupported scene trajectory type={traj_type}, fallback to line: obstacle={name}')
        return self._scene_line_state(traj, t)

    def _build_scene_payload(self, t):
        scene = self._load_scene_config()
        obstacles = []
        obs_id = 1

        for wall in self._scene_walls(scene):
            points = WallObstacle(**wall).to_collision_points(start_obs_id=obs_id)
            obstacles.extend(points)
            obs_id += len(points)

        for item in self._sdf_static_marker_items(scene):
            name = str(item.get('name', f'sdf_static_{obs_id}'))
            radius = max(0.05, _finite_float(item.get('radius'), 0.5))
            # Static SDF markers are useful as known-map obstacles for pillars.
            # Do not feed long boundary/box structures as circular obstacles:
            # their fitted radius can be tens of meters and falsely places the
            # UAV inside a collision at spawn.
            if name.startswith('outer_boundary') or radius > 3.0:
                continue
            obstacles.append(
                {
                    'obs_id': obs_id,
                    'name': name,
                    'x': float(item.get('x', 0.0)),
                    'y': float(item.get('y', 0.0)),
                    'z': float(item.get('z', 0.0)),
                    'vx': 0.0,
                    'vy': 0.0,
                    'vz': 0.0,
                    'radius': radius,
                    'shape': item.get('shape', 'obstacle'),
                    'size': item.get('size', []),
                    'sdf_static': True,
                }
            )
            obs_id += 1

        for obs in scene.get('dynamic_obstacles') or []:
            if not isinstance(obs, dict):
                continue
            x, y, z, vx, vy, vz = self._scene_dynamic_state(obs, t)
            obstacles.append(
                {
                    'obs_id': obs_id,
                    'name': str(obs.get('name', f'dynamic_obstacle_{obs_id}')),
                    'x': x,
                    'y': y,
                    'z': z,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'radius': max(0.05, _finite_float(obs.get('radius'), 0.5)),
                }
            )
            obs_id += 1

        for idx, item in enumerate(self._sdf_dynamic_obstacle_items(scene), start=1):
            x, y, z, vx, vy, vz = self._sdf_dynamic_state(scene, item, idx, t)
            obstacles.append(
                {
                    'obs_id': obs_id,
                    'name': str(item.get('name', f'sdf_dynamic_{idx}')),
                    'x': x,
                    'y': y,
                    'z': z,
                    'vx': vx,
                    'vy': vy,
                    'vz': vz,
                    'radius': max(0.05, _finite_float(item.get('radius'), 0.5)),
                    'shape': item.get('shape', 'box'),
                    'size': item.get('size', []),
                    'sdf_dynamic': True,
                }
            )
            obs_id += 1

        return {'stamp': time.time(), 'scene_time': t, 'obstacles': obstacles}

    def _dynamic_items_for_record(self, payload):
        items = []
        for idx, obs in enumerate(payload.get('obstacles', []), start=1):
            name = str(obs.get('name', '')).strip()
            if self.mode == 'scene' and not name:
                continue
            if self.mode == 'static_wall':
                continue
            obstacle_id = name or f'obs_{obs.get("obs_id", idx)}'
            items.append(
                {
                    'time': float(payload.get('scene_time', time.time() - self.t0)),
                    'obstacle_id': obstacle_id,
                    'x': float(obs.get('x', 0.0)),
                    'y': float(obs.get('y', 0.0)),
                    'z': float(obs.get('z', 0.0)),
                    'radius': float(obs.get('radius', 0.5)),
                }
            )
        return items

    def _record_obstacle_trajectory(self, payload):
        record_path = str(self.get_parameter('trajectory_record_path').value or '').strip()
        record_dt = float(self.get_parameter('trajectory_record_dt').value)
        if not record_path or record_dt <= 0.0:
            return

        scene_time = float(payload.get('scene_time', time.time() - self.t0))
        if self._last_trajectory_record_t is not None:
            if scene_time - self._last_trajectory_record_t < record_dt - 1e-9:
                return

        items = self._dynamic_items_for_record(payload)
        if not items:
            return

        path = _resolve_path(record_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        need_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['time', 'obstacle_id', 'x', 'y', 'z', 'radius'])
            if need_header:
                writer.writeheader()
            for item in items:
                writer.writerow(
                    {
                        'time': f"{item['time']:.4f}",
                        'obstacle_id': item['obstacle_id'],
                        'x': f"{item['x']:.4f}",
                        'y': f"{item['y']:.4f}",
                        'z': f"{item['z']:.4f}",
                        'radius': f"{item['radius']:.4f}",
                    }
                )
        self._last_trajectory_record_t = scene_time

    def _rviz_markers_enabled(self):
        return int(self.get_parameter('rviz_markers_enable').value) != 0

    def _rviz_marker_frame(self):
        return str(self.get_parameter('rviz_marker_frame').value or 'map').strip() or 'map'

    def _base_marker(self, ns, marker_id, marker_type):
        marker = Marker()
        marker.header.frame_id = self._rviz_marker_frame()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = int(marker_id)
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _point(x, y, z):
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    def _append_text_marker(self, markers, ns, marker_id, text, enu_x, enu_y, enu_z, size=0.65, color=None):
        marker = self._base_marker(ns, marker_id, Marker.TEXT_VIEW_FACING)
        marker.pose.position.x = float(enu_x)
        marker.pose.position.y = float(enu_y)
        marker.pose.position.z = float(enu_z)
        marker.scale.z = float(size)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color or (0.9, 0.9, 0.9, 0.95)
        marker.text = str(text)
        markers.markers.append(marker)

    def _append_line_marker(self, markers, ns, marker_id, points, width=0.08, color=None):
        marker = self._base_marker(ns, marker_id, Marker.LINE_STRIP)
        marker.points = [self._point(*p) for p in points]
        marker.scale.x = float(width)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color or (0.9, 0.9, 0.9, 0.8)
        markers.markers.append(marker)

    def _append_arrow_marker(self, markers, ns, marker_id, start, end, color):
        marker = self._base_marker(ns, marker_id, Marker.ARROW)
        marker.points = [self._point(*start), self._point(*end)]
        marker.scale.x = 0.12
        marker.scale.y = 0.32
        marker.scale.z = 0.42
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        markers.markers.append(marker)

    def _append_scene_guide_markers(self, markers, scene):
        if not scene:
            return

        # RViz uses ENU, while the mission YAML uses NED:
        # RViz x = NED y, RViz y = NED x, RViz z = -NED z.
        self._append_arrow_marker(markers, 'map_axes', 900001, (0.0, 0.0, 0.15), (8.0, 0.0, 0.15), (0.25, 0.85, 0.25, 0.9))
        self._append_arrow_marker(markers, 'map_axes', 900002, (0.0, 0.0, 0.15), (0.0, 8.0, 0.15), (0.95, 0.25, 0.25, 0.9))
        self._append_arrow_marker(markers, 'map_axes', 900003, (0.0, 0.0, 0.15), (0.0, 0.0, 4.0), (0.25, 0.45, 1.0, 0.9))
        self._append_text_marker(markers, 'map_axes_text', 900011, 'RViz X = NED y', 8.8, 0.0, 0.6, 0.55, (0.25, 0.85, 0.25, 0.95))
        self._append_text_marker(markers, 'map_axes_text', 900012, 'RViz Y = NED x', 0.0, 8.8, 0.6, 0.55, (0.95, 0.25, 0.25, 0.95))
        self._append_text_marker(markers, 'map_axes_text', 900013, 'RViz Z = -NED z', 0.0, 0.0, 4.6, 0.55, (0.25, 0.45, 1.0, 0.95))

        swarm = scene.get('swarm') or {}
        num_drones = max(1, int(_finite_float(swarm.get('num_drones'), 5)))

        def short_edge_points(cfg):
            center_x = _finite_float(cfg.get('x'), 0.0)
            center_y = _finite_float(cfg.get('y'), 0.0)
            z = _finite_float(cfg.get('z'), -3.0)
            axis = str(cfg.get('axis', 'y')).strip().lower()
            spacing = _finite_float(cfg.get('y_spacing'), _finite_float(swarm.get('y_spacing'), 1.8))
            half = 0.5 * float(num_drones - 1) * spacing
            if axis == 'x':
                p1_ned = (center_x - half, center_y, z)
                p2_ned = (center_x + half, center_y, z)
            else:
                p1_ned = (center_x, center_y - half, z)
                p2_ned = (center_x, center_y + half, z)
            return self._ned_to_enu(*p1_ned), self._ned_to_enu(*p2_ned), self._ned_to_enu(center_x, center_y, z), axis

        start_cfg = scene.get('start') or {}
        target_cfg = scene.get('target') or {}
        if start_cfg:
            p1, p2, center, axis = short_edge_points(start_cfg)
            p1_ground = (p1[0], p1[1], 0.12)
            p2_ground = (p2[0], p2[1], 0.12)
            center_ground = (center[0], center[1], 0.12)
            self._append_line_marker(markers, 'mission_edges', 900101, [p1_ground, p2_ground], 0.055, (0.15, 0.55, 1.0, 0.55))
            self._append_text_marker(
                markers, 'mission_edges_text', 900111,
                f"START y={_finite_float(start_cfg.get('y'), 0.0):.1f}",
                center_ground[0], center_ground[1], 0.7, 0.38, (0.15, 0.55, 1.0, 0.85)
            )
        if target_cfg:
            p1, p2, center, axis = short_edge_points(target_cfg)
            p1_ground = (p1[0], p1[1], 0.12)
            p2_ground = (p2[0], p2[1], 0.12)
            center_ground = (center[0], center[1], 0.12)
            self._append_line_marker(markers, 'mission_edges', 900201, [p1_ground, p2_ground], 0.055, (0.1, 0.95, 0.25, 0.55))
            self._append_text_marker(
                markers, 'mission_edges_text', 900211,
                f"TARGET y={_finite_float(target_cfg.get('y'), 0.0):.1f}",
                center_ground[0], center_ground[1], 0.7, 0.38, (0.1, 0.95, 0.25, 0.85)
            )

    def _delete_all_marker(self):
        marker = Marker()
        marker.header.frame_id = self._rviz_marker_frame()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.action = Marker.DELETEALL
        return marker

    def _build_obstacle_marker_array(self, payload):
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker())
        scene = self._load_scene_config() if self.mode == 'scene' else {}
        self._append_scene_guide_markers(markers, scene)

        static_offset = 100000
        for idx, item in enumerate(self._sdf_static_marker_items(scene), start=1):
            marker = self._base_marker(
                'sdf_static_structures',
                static_offset + idx,
                Marker.CYLINDER if item.get('shape') == 'cylinder' else Marker.CUBE,
            )
            marker.pose.position.x = float(item.get('enu_x', 0.0))
            marker.pose.position.y = float(item.get('enu_y', 0.0))
            marker.pose.position.z = float(item.get('enu_z', 0.0))
            marker.pose.orientation.w = math.cos(float(item.get('yaw', 0.0)) * 0.5)
            marker.pose.orientation.z = math.sin(float(item.get('yaw', 0.0)) * 0.5)
            size = item.get('size') or [1.0, 1.0, 1.0]
            marker.scale.x = max(0.05, float(size[0]))
            marker.scale.y = max(0.05, float(size[1]))
            marker.scale.z = max(0.05, float(size[2]))
            marker.color.r = 0.45
            marker.color.g = 0.47
            marker.color.b = 0.46
            marker.color.a = 0.45
            markers.markers.append(marker)

        for idx, obs in enumerate(payload.get('obstacles', []), start=1):
            name = str(obs.get('name', '')).strip()
            is_dynamic = bool(name)
            radius = max(0.05, _finite_float(obs.get('radius'), 0.35))
            enu_x, enu_y, enu_z = self._ned_to_enu(obs.get('x', 0.0), obs.get('y', 0.0), obs.get('z', 0.0))

            marker = self._base_marker(
                'dynamic_obstacles' if is_dynamic else 'static_obstacle_points',
                idx,
                Marker.CUBE if is_dynamic else Marker.SPHERE,
            )
            marker.pose.position.x = float(enu_x)
            marker.pose.position.y = float(enu_y)
            marker.pose.position.z = float(enu_z)

            if is_dynamic:
                size = obs.get('size') if isinstance(obs.get('size'), list) else []
                if len(size) >= 3:
                    marker.scale.x = max(0.15, float(size[0]))
                    marker.scale.y = max(0.15, float(size[1]))
                    marker.scale.z = max(0.15, float(size[2]))
                else:
                    side = max(0.25, radius * 1.6)
                    marker.scale.x = side
                    marker.scale.y = side
                    marker.scale.z = side
                marker.color.r = 0.72
                marker.color.g = 0.64
                marker.color.b = 0.48
                marker.color.a = 0.9
                marker.text = name
            else:
                marker.scale.x = radius * 1.6
                marker.scale.y = radius * 1.6
                marker.scale.z = max(0.12, radius * 0.4)
                marker.color.r = 0.75
                marker.color.g = 0.78
                marker.color.b = 0.80
                marker.color.a = 0.28
            markers.markers.append(marker)

            if is_dynamic:
                vel_marker = self._base_marker('dynamic_obstacle_velocity', idx, Marker.ARROW)
                vel_marker.points.append(marker.pose.position)
                end = type(marker.pose.position)()
                enu_vx, enu_vy, enu_vz = self._ned_to_enu(obs.get('vx', 0.0), obs.get('vy', 0.0), obs.get('vz', 0.0))
                velocity_marker_scale = 0.32
                end.x = float(enu_x) + float(enu_vx) * velocity_marker_scale
                end.y = float(enu_y) + float(enu_vy) * velocity_marker_scale
                end.z = float(enu_z) + float(enu_vz) * velocity_marker_scale
                vel_marker.points.append(end)
                vel_marker.scale.x = 0.025
                vel_marker.scale.y = 0.065
                vel_marker.scale.z = 0.09
                vel_marker.color.r = 1.0
                vel_marker.color.g = 0.88
                vel_marker.color.b = 0.18
                vel_marker.color.a = 0.45
                markers.markers.append(vel_marker)

        return markers

    def _build_target_marker_array(self, target_payload):
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker())
        if not target_payload:
            return markers

        for idx, item in enumerate(target_payload.get('markers', []), start=1):
            marker = self._base_marker('target_markers', idx, Marker.SPHERE)
            marker.pose.position.x = float(item.get('enu_x', 0.0))
            marker.pose.position.y = float(item.get('enu_y', 0.0))
            marker.pose.position.z = float(item.get('enu_z', 0.0))
            marker.scale.x = 0.45
            marker.scale.y = 0.45
            marker.scale.z = 0.45
            marker.color.r = 0.1
            marker.color.g = 0.9
            marker.color.b = 0.25
            marker.color.a = 0.85
            markers.markers.append(marker)

            label = self._base_marker('target_marker_labels', 1000 + idx, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = float(item.get('enu_x', 0.0))
            label.pose.position.y = float(item.get('enu_y', 0.0))
            label.pose.position.z = float(item.get('enu_z', 0.0)) + 0.65
            label.scale.z = 0.45
            label.color.r = 0.1
            label.color.g = 1.0
            label.color.b = 0.25
            label.color.a = 0.95
            label.text = f"T{int(item.get('drone_id', idx))}"
            markers.markers.append(label)
        return markers

    def _build_drone_marker_array(self):
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker())
        if int(self.get_parameter('rviz_drone_markers_enable').value) == 0:
            return markers

        now = time.time()
        stale_after = 3.0
        colors = {
            1: (0.15, 0.55, 1.0),
            2: (1.0, 0.45, 0.15),
            3: (0.75, 0.35, 1.0),
            4: (0.1, 0.85, 0.85),
            5: (1.0, 0.9, 0.2),
        }
        for drone_id in sorted(self._latest_swarm_states.keys()):
            state = self._latest_swarm_states[drone_id]
            if now - state.get('time', 0.0) > stale_after:
                continue
            color = colors.get(drone_id, (0.25, 0.75, 1.0))
            enu_x = float(state.get('enu_x', 0.0))
            enu_y = float(state.get('enu_y', 0.0))
            enu_z = float(state.get('enu_z', 0.0))
            heading = float(state.get('heading', 0.0))

            body = self._base_marker('drone_body', drone_id, Marker.SPHERE)
            body.pose.position.x = enu_x
            body.pose.position.y = enu_y
            body.pose.position.z = enu_z
            body.scale.x = 0.55
            body.scale.y = 0.55
            body.scale.z = 0.22
            body.color.r = color[0]
            body.color.g = color[1]
            body.color.b = color[2]
            body.color.a = 0.95
            markers.markers.append(body)

            # PX4 heading is in NED; convert a short forward vector to RViz ENU.
            forward_ned_x = math.cos(heading)
            forward_ned_y = math.sin(heading)
            forward_enu_x, forward_enu_y, _ = self._ned_to_enu(forward_ned_x, forward_ned_y, 0.0)
            arrow = self._base_marker('drone_heading', 1000 + drone_id, Marker.ARROW)
            arrow.points = [
                self._point(enu_x, enu_y, enu_z + 0.08),
                self._point(enu_x + 0.9 * forward_enu_x, enu_y + 0.9 * forward_enu_y, enu_z + 0.08),
            ]
            arrow.scale.x = 0.05
            arrow.scale.y = 0.16
            arrow.scale.z = 0.22
            arrow.color.r = color[0]
            arrow.color.g = color[1]
            arrow.color.b = color[2]
            arrow.color.a = 0.8
            markers.markers.append(arrow)

            label = self._base_marker('drone_labels', 2000 + drone_id, Marker.TEXT_VIEW_FACING)
            label.pose.position.x = enu_x
            label.pose.position.y = enu_y
            label.pose.position.z = enu_z + 0.65
            label.scale.z = 0.42
            label.color.r = color[0]
            label.color.g = color[1]
            label.color.b = color[2]
            label.color.a = 0.95
            label.text = f"x500_{drone_id}"
            markers.markers.append(label)

            if int(self.get_parameter('rviz_drone_path_enable').value) != 0:
                path = self._drone_paths.get(drone_id, [])
                if len(path) >= 2:
                    line = self._base_marker('drone_paths', 3000 + drone_id, Marker.LINE_STRIP)
                    line.points = [self._point(*p) for p in path]
                    line.scale.x = 0.055
                    line.color.r = color[0]
                    line.color.g = color[1]
                    line.color.b = color[2]
                    line.color.a = 0.55
                    markers.markers.append(line)

        return markers

    def _blend_center(self, t: float):
        start_x = self._p('start_center_x')
        start_y = self._p('start_center_y')
        active_x = self._p('active_center_x')
        active_y = self._p('active_center_y')
        switch_after = max(0.0, self._p('switch_after_sec'))
        switch_fade = max(0.1, self._p('switch_fade_sec'))

        if t <= switch_after:
            return start_x, start_y
        if t >= switch_after + switch_fade:
            if not self._switch_logged:
                self._switch_logged = True
                self.get_logger().info(
                    f'obstacle center switched to active path center=({active_x:.2f}, {active_y:.2f})'
                )
            return active_x, active_y

        ratio = (t - switch_after) / switch_fade
        cx = start_x + (active_x - start_x) * ratio
        cy = start_y + (active_y - start_y) * ratio
        return cx, cy

    def _configured_walls(self):
        config_path = str(self.get_parameter('obstacle_config').value or '').strip()
        if config_path:
            return normalized_walls(load_obstacle_config(config_path))

        wall_x_ned = float(self._p('wall_x'))
        wall_y_ned = float(self._p('wall_y'))
        wall_length = max(1.0, float(self._p('wall_length')))
        rear_wall_length = max(wall_length, float(self._p('rear_wall_length')))
        wall = {
            'name': 'front_wall',
            'x': wall_x_ned,
            'y': wall_y_ned,
            'z': float(self._p('wall_z')),
            'length': wall_length,
            'thickness': max(0.1, float(self._p('wall_thickness'))),
            'height': max(0.5, float(self._p('wall_height'))),
            'segment_spacing': max(0.2, float(self._p('wall_segment_spacing'))),
        }
        walls = [wall]
        if int(self._p('second_wall_enable')) != 0:
            walls.append(
                {
                    **wall,
                    'name': 'rear_right_wall',
                    'x': wall_x_ned + float(self._p('second_wall_dx')),
                    'y': wall_y_ned + float(self._p('second_wall_dy')),
                    'length': rear_wall_length,
                }
            )
        if int(self._p('third_wall_enable')) != 0:
            walls.append(
                {
                    **wall,
                    'name': 'rear_left_wall',
                    'x': wall_x_ned + float(self._p('second_wall_dx')),
                    'y': wall_y_ned - float(self._p('second_wall_dy')),
                    'length': rear_wall_length,
                }
            )
        return walls

    def _configured_obstacles(self):
        config_path = str(self.get_parameter('obstacle_config').value or '').strip()
        if config_path:
            return obstacles_from_config(load_obstacle_config(config_path))
        return [WallObstacle(**wall) for wall in self._configured_walls()]

    def _configured_target(self):
        scene_path = str(self.get_parameter('scene_config').value or '').strip()
        if self.mode == 'scene' and scene_path:
            scene = self._load_scene_config()
            target = scene.get('target') or {}
            if isinstance(target, dict):
                default = {'x': 20.0, 'y': 0.0, 'z': -3.0, 'y_spacing': 1.8, 'axis': 'y'}
                return {
                    'x': _finite_float(target.get('x', default['x']), default['x']),
                    'y': _finite_float(target.get('y', default['y']), default['y']),
                    'z': _finite_float(target.get('z', default['z']), default['z']),
                    'y_spacing': _finite_float(target.get('y_spacing', default['y_spacing']), default['y_spacing']),
                    'axis': str(target.get('axis', default['axis'])),
                }
        config_path = str(self.get_parameter('obstacle_config').value or '').strip()
        if config_path:
            return normalized_target(load_obstacle_config(config_path))
        return {
            'x': float(self.get_parameter('target_ball_target_x').value),
            'y': float(self.get_parameter('target_ball_base_y').value),
            'z': float(self.get_parameter('target_ball_target_z').value),
            'y_spacing': float(self.get_parameter('target_ball_y_spacing').value),
            'axis': 'y',
        }

    def _publish(self):
        wall_t = time.time() - self.t0
        t = wall_t
        z = self._p('z')

        if self.mode == 'scene':
            self._log_scene_clock_config_once()
            if int(self.get_parameter('scene_freeze_dynamics').value) != 0:
                t = 0.0
            else:
                t = self._scene_time(wall_t)
            payload = self._build_scene_payload(t)
            dynamic_items = [
                obs for obs in payload.get('obstacles', [])
                if str(obs.get('name', '')).strip()
            ]
            if dynamic_items:
                first = dynamic_items[0]
                x, y, z = self._ned_to_enu(first.get('x', 0.0), first.get('y', 0.0), first.get('z', 0.0))
                vx, vy, _ = self._ned_to_enu(first.get('vx', 0.0), first.get('vy', 0.0), 0.0)
                r = float(first.get('radius', 0.3))
            else:
                x = y = vx = vy = 0.0
                r = 0.3
        elif self.mode == 'static_wall':
            obstacles_cfg = self._configured_obstacles()
            base_obstacle = obstacles_cfg[0]
            wall_x_ned = float(base_obstacle.x)
            wall_y_ned = float(base_obstacle.y)
            obstacles = []
            obs_id = 1
            for obs in obstacles_cfg:
                points = obs.to_collision_points(start_obs_id=obs_id)
                obstacles.extend(points)
                obs_id += len(points)
            payload = {'stamp': time.time(), 'obstacles': obstacles}
            x, y, _ = self._ned_to_enu(wall_x_ned, wall_y_ned, 0.0)
            vx = 0.0
            vy = 0.0
            r = getattr(base_obstacle, 'radius', 0.3)

            # Compute current target ball ENU pose from NED target parameters.
            target_cfg = self._configured_target()
            target_x_ned = float(target_cfg['x'])
            target_z_ned = float(target_cfg['z'])
            base_y_ned = float(target_cfg['y'])
            target_y_ned = base_y_ned
            target_ball_enu = self._ned_to_enu(target_x_ned, target_y_ned, target_z_ned)
        else:
            base_cx, base_cy = self._blend_center(t)
            # center_x/center_y 作为运行时微调偏置，便于热更新
            cx = base_cx + self._p('center_x')
            cy = base_cy + self._p('center_y')
            ax = self._p('amplitude_x')
            ay = self._p('amplitude_y')
            w = self._p('omega')
            r = max(0.05, self._p('radius'))

            x = cx + ax * math.sin(w * t)
            y = cy + ay * math.cos(w * t)
            vx = ax * w * math.cos(w * t)
            vy = -ay * w * math.sin(w * t)

            ned_x, ned_y, ned_z = self._enu_to_ned(x, y, 0.0)
            ned_vx, ned_vy, ned_vz = self._enu_vel_to_ned(vx, vy, 0.0)

            payload = {
                'stamp': time.time(),
                'obstacles': [
                    {
                        'obs_id': 1,
                        'x': ned_x,
                        'y': ned_y,
                        'z': ned_z,
                        'vx': ned_vx,
                        'vy': ned_vy,
                        'vz': ned_vz,
                        'radius': r,
                    }
                ],
            }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)
        self._record_obstacle_trajectory(payload)

        target_payload = self._build_target_markers_payload()
        if target_payload is not None:
            target_msg = String()
            target_msg.data = json.dumps(target_payload, ensure_ascii=False)
            self.target_pub.publish(target_msg)

        if self._rviz_markers_enabled():
            self.marker_pub.publish(self._build_obstacle_marker_array(payload))
            self.target_marker_pub.publish(self._build_target_marker_array(target_payload))
            self.drone_marker_pub.publish(self._build_drone_marker_array())

        if self.visualize_gz:
            now = time.time()
            if not self._gz_ready and now - self._last_probe_t > 1.5:
                self._last_probe_t = now
                self._setup_gz_visual()
            if now - self._last_pose_set_t > 0.10:
                self._last_pose_set_t = now
                if self.mode == 'static_wall':
                    if not self._wall_spawned and self._gz_ready:
                        self._spawn_visual_obstacle()
                    if self._gz_ready and int(self.get_parameter('target_ball_enable').value) == 1:
                        if not self._target_ball_spawned:
                            self._spawn_target_ball()
                        ok_target_pose = self._set_target_ball_pose_enu(
                            float(target_ball_enu[0]),
                            float(target_ball_enu[1]),
                            float(target_ball_enu[2]),
                        )
                        if not ok_target_pose and now - self._last_pose_warn_t > 1.0:
                            self._last_pose_warn_t = now
                            self.get_logger().warn('target ball pose sync failed, will retry')
                elif self.mode == 'scene':
                    ok = self._sync_scene_visual_obstacles(payload)
                    if not ok and now - self._last_pose_warn_t > 1.0:
                        self._last_pose_warn_t = now
                        self.get_logger().warn(
                            f'scene visual pose sync failed, spawned={len(self._scene_visual_spawned)}, '
                            f'world={self.world_name}'
                        )
                else:
                    vis_x, vis_y, vis_z = self._next_visual_pose(x, y, z)
                    ok = self._set_visual_pose(vis_x, vis_y, vis_z)
                    if not ok and not self._spawn_done and self._gz_ready:
                        self._spawn_visual_obstacle()
                    elif ok:
                        self._visual_last_ok_x = float(vis_x)
                        self._visual_last_ok_y = float(vis_y)
                        self._visual_last_ok_z = float(vis_z)
                    elif not ok:
                        if now - self._last_pose_warn_t > 1.0:
                            self._last_pose_warn_t = now
                            self.get_logger().warn(
                                f'visual pose sync failed, keep last pose, '
                                f'entity={self._visual_obstacle_entity_name()}, world={self.world_name}'
                            )

        now = time.time()
        if now - self._last_log_t > 1.0:
            self._last_log_t = now
            if self.mode == 'static_wall':
                self.get_logger().info(
                    f'wall publish: NED=({wall_x_ned:.2f},{wall_y_ned:.2f}), '
                    f'ENU=({x:.2f},{y:.2f},{z:.2f}), segments={len(payload["obstacles"])}'
                )
            elif self.mode == 'scene':
                dynamic_count = len([obs for obs in payload.get('obstacles', []) if str(obs.get('name', '')).strip()])
                self.get_logger().info(
                    f'scene publish: t={payload.get("scene_time", 0.0):.2f}, '
                    f'obstacles={len(payload["obstacles"])}, dynamic={dynamic_count}'
                )
            else:
                self.get_logger().info(
                    f'obstacle publish (ENU for viz): x={x:.2f}, y={y:.2f}, z={z:.2f}, vx={vx:.2f}, vy={vy:.2f}, r={r:.2f}'
                )
            if target_payload is not None and len(target_payload.get('markers', [])) > 0:
                m0 = target_payload['markers'][0]
                self.get_logger().info(
                    f'target markers publish: count={len(target_payload["markers"])}, '
                    f'leader={target_payload.get("leader_id", 1)}, '
                    f'sample_ned=({m0["x"]:.2f},{m0["y"]:.2f},{m0["z"]:.2f}), '
                    f'sample_enu=({m0["enu_x"]:.2f},{m0["enu_y"]:.2f},{m0["enu_z"]:.2f})'
                )
            if target_payload is not None and len(target_payload.get('markers', [])) > 0 and now - self._last_frame_check_log_t > 10.0:
                self._last_frame_check_log_t = now
                m0 = target_payload['markers'][0]
                self.get_logger().info(
                    'frame-check(10s): same marker '
                    f'ENU=({m0["enu_x"]:.2f},{m0["enu_y"]:.2f},{m0["enu_z"]:.2f}) -> '
                    f'NED=({m0["x"]:.2f},{m0["y"]:.2f},{m0["z"]:.2f})'
                )
                self._log_target_table(target_payload, prefix='target-table(10s)')


def main():
    parser = argparse.ArgumentParser(description='Dynamic obstacle source for hybrid swarm tests')
    parser.add_argument('--rate', type=float, default=10.0)
    parser.add_argument('--world', type=str, default='default')
    parser.add_argument('--visualize-gz', type=int, default=1, help='1 to visualize obstacle entity in Gazebo')
    parser.add_argument('--mode', type=str, default='moving_ball', choices=['moving_ball', 'static_wall', 'scene'])
    parser.add_argument('--start-center-x', type=float, default=18.0)
    parser.add_argument('--start-center-y', type=float, default=6.0)
    parser.add_argument('--active-center-x', type=float, default=5.5)
    parser.add_argument('--active-center-y', type=float, default=0.0)
    parser.add_argument('--switch-after-sec', type=float, default=38.0)
    parser.add_argument('--switch-fade-sec', type=float, default=6.0)
    parser.add_argument('--wall-x', type=float, default=5.5)
    parser.add_argument('--wall-y', type=float, default=0.0)
    parser.add_argument('--wall-z', type=float, default=0.9)
    parser.add_argument('--wall-length', type=float, default=5.0)
    parser.add_argument('--wall-thickness', type=float, default=0.25)
    parser.add_argument('--wall-height', type=float, default=1.8)
    parser.add_argument('--wall-segment-spacing', type=float, default=0.6)
    parser.add_argument('--obstacle-config', type=str, default='')
    parser.add_argument('--scene-config', type=str, default='')
    parser.add_argument('--second-wall-enable', type=int, default=1)
    parser.add_argument('--second-wall-dx', type=float, default=15.0)
    parser.add_argument('--second-wall-dy', type=float, default=109.0)
    parser.add_argument('--rear-wall-length', type=float, default=200.0)
    parser.add_argument('--third-wall-enable', type=int, default=1)
    parser.add_argument('--target-ball-enable', '--target-marker-enable', dest='target_ball_enable', type=int, default=1)
    parser.add_argument('--target-ball-num-drones', '--target-marker-num-drones', dest='target_ball_num_drones', type=int, default=3)
    parser.add_argument('--target-ball-leader-id', '--target-marker-leader-id', dest='target_ball_leader_id', type=int, default=1)
    parser.add_argument('--target-ball-target-x', '--target-marker-target-x', dest='target_ball_target_x', type=float, default=20.0)
    parser.add_argument('--target-ball-target-z', '--target-marker-target-z', dest='target_ball_target_z', type=float, default=-3.0)
    parser.add_argument('--target-ball-base-y', '--target-marker-base-y', dest='target_ball_base_y', type=float, default=0.0)
    parser.add_argument('--target-ball-y-spacing', '--target-marker-y-spacing', dest='target_ball_y_spacing', type=float, default=1.8)
    parser.add_argument('--trajectory-record-path', type=str, default='')
    parser.add_argument('--trajectory-record-dt', type=float, default=0.0)
    parser.add_argument('--scene-clock-mode', type=str, default='wall_time', choices=['wall_time', 'auto_takeoff', 'takeoff'])
    parser.add_argument('--scene-freeze-dynamics', type=int, default=0)
    parser.add_argument('--scene-start-num-drones', type=int, default=0)
    parser.add_argument('--scene-start-z-threshold', type=float, default=-2.0)
    parser.add_argument('--scene-start-stable-sec', type=float, default=2.0)
    parser.add_argument('--rviz-markers-enable', type=int, default=1)
    parser.add_argument('--rviz-marker-frame', type=str, default='map')
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = DynamicObstacleSource(
        rate_hz=args.rate,
        world_name=args.world,
        visualize_gz=(int(args.visualize_gz) == 1),
        mode=args.mode,
        start_center_x=args.start_center_x,
        start_center_y=args.start_center_y,
        active_center_x=args.active_center_x,
        active_center_y=args.active_center_y,
        switch_after_sec=args.switch_after_sec,
        switch_fade_sec=args.switch_fade_sec,
    )
    node.set_parameters([
        Parameter('wall_x', Parameter.Type.DOUBLE, float(args.wall_x)),
        Parameter('wall_y', Parameter.Type.DOUBLE, float(args.wall_y)),
        Parameter('wall_z', Parameter.Type.DOUBLE, float(args.wall_z)),
        Parameter('wall_length', Parameter.Type.DOUBLE, float(args.wall_length)),
        Parameter('wall_thickness', Parameter.Type.DOUBLE, float(args.wall_thickness)),
        Parameter('wall_height', Parameter.Type.DOUBLE, float(args.wall_height)),
        Parameter('wall_segment_spacing', Parameter.Type.DOUBLE, float(args.wall_segment_spacing)),
        Parameter('obstacle_config', Parameter.Type.STRING, str(args.obstacle_config)),
        Parameter('scene_config', Parameter.Type.STRING, str(args.scene_config)),
        Parameter('second_wall_enable', Parameter.Type.INTEGER, int(args.second_wall_enable)),
        Parameter('second_wall_dx', Parameter.Type.DOUBLE, float(args.second_wall_dx)),
        Parameter('second_wall_dy', Parameter.Type.DOUBLE, float(args.second_wall_dy)),
        Parameter('rear_wall_length', Parameter.Type.DOUBLE, float(args.rear_wall_length)),
        Parameter('third_wall_enable', Parameter.Type.INTEGER, int(args.third_wall_enable)),
        Parameter('target_ball_enable', Parameter.Type.INTEGER, int(args.target_ball_enable)),
        Parameter('target_ball_num_drones', Parameter.Type.INTEGER, int(args.target_ball_num_drones)),
        Parameter('target_ball_leader_id', Parameter.Type.INTEGER, int(args.target_ball_leader_id)),
        Parameter('target_ball_target_x', Parameter.Type.DOUBLE, float(args.target_ball_target_x)),
        Parameter('target_ball_target_z', Parameter.Type.DOUBLE, float(args.target_ball_target_z)),
        Parameter('target_ball_base_y', Parameter.Type.DOUBLE, float(args.target_ball_base_y)),
        Parameter('target_ball_y_spacing', Parameter.Type.DOUBLE, float(args.target_ball_y_spacing)),
        Parameter('trajectory_record_path', Parameter.Type.STRING, str(args.trajectory_record_path)),
        Parameter('trajectory_record_dt', Parameter.Type.DOUBLE, float(args.trajectory_record_dt)),
        Parameter('scene_clock_mode', Parameter.Type.STRING, str(args.scene_clock_mode)),
        Parameter('scene_freeze_dynamics', Parameter.Type.INTEGER, int(args.scene_freeze_dynamics)),
        Parameter('scene_start_num_drones', Parameter.Type.INTEGER, int(args.scene_start_num_drones)),
        Parameter('scene_start_z_threshold', Parameter.Type.DOUBLE, float(args.scene_start_z_threshold)),
        Parameter('scene_start_stable_sec', Parameter.Type.DOUBLE, float(args.scene_start_stable_sec)),
        Parameter('rviz_markers_enable', Parameter.Type.INTEGER, int(args.rviz_markers_enable)),
        Parameter('rviz_marker_frame', Parameter.Type.STRING, str(args.rviz_marker_frame)),
    ])
    if node.visualize_gz:
        node._setup_gz_visual()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
