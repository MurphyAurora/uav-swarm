# UAV Swarm 3D Motion Primitive Planning and Dynamic Obstacle Avoidance

## 项目简介

本项目面向多无人机动态避障研究，基于 **ROS 2 + PX4 SITL + Gazebo + XTDrone2** 构建闭环仿真平台。

当前系统包含：

- 三维 Motion Primitive 局部规划
- 动态障碍跟踪与短期预测
- 时空碰撞风险评价
- 安全速度投影
- 多无人机状态交换

系统目标是研究未知动态环境下无人机实时安全规划方法。

---

## Windows算法验证

为降低ROS2/PX4调试成本，新增 `windows_3d_mp_test` 用于算法级验证。

验证内容：

- 三维运动基元生成
- 随机高度动态障碍场景
- 多目标代价函数评价
- 三维轨迹可视化

验证流程：

```
UAV state
   |
3D Motion Primitive generation
   |
Trajectory rollout
   |
Cost evaluation
   |
Best primitive selection
```

采用多目标代价：

```
J = J_goal + J_collision + J_altitude + J_smooth + J_energy
```

其中通过高度偏离代价和高度变化代价，避免无人机通过无限升高规避障碍。

---

## 当前规划架构

```
PX4/Gazebo
     |
ROS2 Communication
     |
Obstacle tracking and prediction
     |
Motion Primitive Selector
     |
3D rollout + Risk evaluation
     |
Safety layer
     |
PX4 control
```

---

## 后续研究路线

```
3D Motion Primitive
        +
动态障碍预测
        +
风险评价
        +
高度约束代价
        |
        v
不确定性预测
        +
风险感知规划
        +
多无人机验证
```

旧版本 A* + ORCA 描述属于早期实验版本。