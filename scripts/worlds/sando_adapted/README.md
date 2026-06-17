# SANDO 本机适配 World

这个目录保存从 MIT ACL SANDO world 参考生成的本机适配版 SDF。它们不是直接照搬
SANDO 的原始启动环境，而是为了适配当前 XTD2/PX4/Gazebo 流程做了简化。

## 适配原则

- 移除 SANDO 专用的 ROS/Gazebo 插件。
- 移除 Fuel 在线模型依赖，避免运行时还要下载外部模型。
- 保留 PX4/Gazebo 需要的基础 world 元素：地面、光照、重力、ENU 坐标。
- 结构化场景保留 SANDO 的障碍物布局思想，并映射到当前 XTD2 任务常用区域。
- 不在 world 里生成无人机，仍然由 PX4 启动流程生成 `gz_x500`。

## 文件说明

- `sando_simple_tunnel_xtd2.sdf`：轻量隧道/瓶颈场景，参考 `simple_tunnel.world`。
- `sando_forest3_walls_xtd2.sdf`：柱子+墙体的结构化场景，参考 `forest3_with_walls.world`，外观更接近 SANDO README 视频中的静态结构环境。

配套场景配置：

- `../../scenes/sando/forest3_walls_dynamic.yaml`：给
  `sando_forest3_walls_xtd2.sdf` 使用的算法场景，包含外圈感知边界和 3 个动态障碍，其中一个使用 SANDO 风格 trefoil 轨迹。

## 怎么接入当前流程

当前 `swarm_simulation_launch.py` 通过 PX4 的 `PX4_GZ_WORLD` 选择 world。PX4 默认会从：

```text
firmware/PX4-Autopilot/Tools/simulation/gz/worlds/
```

查找对应的 `.sdf` 文件。因此接入方式是：先把这里的适配 world 同步到 PX4 的 worlds
目录，再启动 launch 时指定 `gz_world`。

示例：

```bash
cd ~/ws_xtd2

./scripts/worlds/install_sando_worlds.sh

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch xtd2_mission swarm_simulation_launch.py \
  gz_world:=sando_forest3_walls_xtd2 \
  px4_use_versioned_local_position:=1 \
  dynamic_obs_enable:=1 \
  dynamic_obs_mode:=scene \
  dynamic_obs_visualize_gz:=0 \
  spawned_formation_axis:=x \
  scene_config:=$HOME/ws_xtd2/scripts/scenes/sando/forest3_walls_dynamic.yaml \
  dynamic_obs_wall_x:=27.0 \
  start_wall_clearance:=8.0 \
  dynamic_obs_wall_y:=-14.0 \
  dynamic_obs_target_x:=19.0 \
  dynamic_obs_target_y_base:=30.0
```

其他 world 同理，只需要把 `gz_world` 改成不带 `.sdf` 后缀的文件名。

上面这组参数会让无人机出生在外墙内：

- 起点中心：NED `(19.0, -14.0, -2.0)`，因为 `dynamic_obs_wall_x - start_wall_clearance = 27.0 - 8.0 = 19.0`。
- 终点中心：NED `(19.0, 30.0, -3.0)`。
- 5 架无人机按 `leader_id` 和 `dynamic_obs_target_y_spacing` 在 x 方向展开，即沿短边排成一列，再飞向对面短边。
- 动态障碍不在 Gazebo 里生成，`struct_wall_*` 小木块会从 SDF 批量转为
  `/xtdrone2/swarm/dynamic_obstacles` 和 RViz MarkerArray。

RViz 调试：

```bash
source /opt/ros/jazzy/setup.bash
source ~/ws_xtd2/install/setup.bash
rviz2
```

在 RViz 里设置 `Fixed Frame=map`，添加：

- `/xtdrone2/swarm/dynamic_obstacle_markers`
- `/xtdrone2/swarm/target_markers_rviz`

## 3D LiDAR 模型

`../px4_models/x500_lidar_3d/` 提供了一个本机适配的 PX4 x500 3D GPU LiDAR 模型。
它不会修改 PX4 原始 `x500` 模型，安装脚本会把它同步到 PX4 的 models 目录。

使用方式：

```bash
cd ~/ws_xtd2
./scripts/worlds/install_sando_worlds.sh

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch xtd2_mission swarm_simulation_launch.py \
  gz_world:=sando_forest3_walls_xtd2 \
  px4_sim_model:=gz_x500_lidar_3d \
  px4_use_versioned_local_position:=1 \
  dynamic_obs_enable:=1 \
  dynamic_obs_mode:=scene \
  dynamic_obs_visualize_gz:=0 \
  spawned_formation_axis:=x \
  scene_config:=$HOME/ws_xtd2/scripts/scenes/sando/forest3_walls_dynamic.yaml \
  dynamic_obs_wall_x:=27.0 \
  start_wall_clearance:=8.0 \
  dynamic_obs_wall_y:=-14.0 \
  dynamic_obs_target_x:=19.0 \
  dynamic_obs_target_y_base:=30.0
```

另开一个终端桥接 3D 点云到 ROS2：

```bash
cd ~/ws_xtd2
./scripts/bridge_x500_lidar_3d.sh 5
```

然后在 RViz 添加 `PointCloud2`，选择脚本桥接出的 `lidar_3d` 点云 topic。

## 注意

这些 world 主要负责 Gazebo 视觉/物理环境。动态障碍、算法可感知的边界、目标点和难度参数放在 YAML 场景里。这样
`dynamic_obstacle_source.py`、难度分级和后处理分析都能直接读取统一的场景配置。
