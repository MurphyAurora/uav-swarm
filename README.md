# UAV Swarm 3D Motion Primitive Planning and Dynamic Obstacle Avoidance

## 项目简介

本项目面向多无人机集群动态避障仿真实验，基于 **ROS 2 + PX4 SITL + Gazebo + XTDrone2** 构建闭环仿真平台。

当前系统已从早期的 A* + ORCA-lite 架构演进为：

- 三维 Motion Primitive 局部规划
- 动态障碍跟踪与短期预测
- 时空碰撞风险评价
- 安全速度投影与紧急避障机制
- 多无人机状态交换与协同验证

系统目标是研究未知动态环境下无人机集群实时安全规划方法。

---

## 当前规划架构

```text
                 PX4 / Gazebo SITL
                         |
                         |
                  ROS2 Communication
                         |
        +----------------+----------------+
        |                                 |
 state_exchange                  obstacle tracking
        |                                 |
        |                         dynamic prediction
        |                                 |
        +---------------+-----------------+
                        |
                        v
          Motion Primitive Selector
                        |
        +---------------+----------------+
        |                                |
  3D primitive rollout          risk evaluation
        |                                |
        +---------------+----------------+
                        |
              safety projection
                        |
              primitive_cmd_vel_ned
                        |
                       PX4
```

---

## 核心模块

### 1. 三维 Motion Primitive Library

位置：

```
src/xtd2_mission/xtd2_mission/motion_primitive_library.py
```

功能：

- 生成短时三维运动基元
- 支持水平、垂直以及组合运动
- 生成未来轨迹 rollout

当前支持：

- straight
- left/right
- climb/descend
- back climb
- diagonal climb

---

### 2. Motion Primitive Selector

位置：

```
src/xtd2_mission/xtd2_mission/motion_primitive_selector.py
```

功能：

- 接收无人机状态
- 接收动态/静态障碍预测结果
- 对候选 primitive 进行评分
- 选择风险最低轨迹

评价因素包括：

- collision feasibility
- clearance
- TTC/risk cost
- trajectory smoothness

---

## 运行环境

推荐：

- Ubuntu 24.04
- ROS 2 Jazzy
- PX4-Autopilot
- Gazebo Harmonic
- Python 3
- colcon

---

## 编译

```bash
cd ~/px4_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 启动仿真

```bash
ros2 launch xtd2_mission swarm_simulation_launch.py
```

或使用启动脚本：

```bash
cd scripts
bash start1.sh
```

---

## 主要话题

| Topic | 功能 |
|-|-|
| `/xtdrone2/swarm/state_exchange` | 多无人机状态交换 |
| `/xtdrone2/swarm/filtered_predicted_dynamic_obstacles` | 动态障碍预测 |
| `/xtdrone2/swarm/tracked_static_obstacles` | 静态障碍信息 |
| `/xtdrone2/swarm/selected_motion_primitives` | primitive选择结果 |
| `/xtdrone2/x500_i/primitive_cmd_vel_ned` | 三维速度控制输出 |

---

## 坐标系

系统统一采用 PX4 NED 坐标：

- x/y：水平运动
- z：高度方向（NED正方向向下）

Motion Primitive、碰撞检测和控制输出均保持 NED 一致。

---

## 当前研究路线

当前版本：

```
3D Motion Primitive
        +
动态障碍预测
        +
风险评价
        +
安全控制层
```

后续计划：

```
不确定性预测
        +
风险感知 Motion Primitive
        +
多无人机动态避障验证
```

---

## 说明

README 已同步当前代码架构。
旧版本中的 A* + ORCA-lite 描述属于早期实验版本，不代表当前 develop 分支规划框架。