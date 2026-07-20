# Windows 3D Motion Primitive Validation

该目录用于在 Windows + Python 环境下验证三维运动基元规划算法，不依赖 ROS2/PX4/Gazebo。

## 目标

验证：

- 三维 Motion Primitive 生成
- 随机高度动态障碍环境
- 多目标代价函数选择
- 三维轨迹可视化

## 规划流程

```
UAV state (x,y,z)
        |
Generate 3D primitives
        |
Trajectory rollout
        |
Cost evaluation
        |
Select best primitive
        |
Update state
```

## Cost Function

综合考虑：

```
J = wg*J_goal
  + wc*J_collision
  + wh*J_altitude
  + ws*J_smooth
  + we*J_energy
```

其中：

- goal cost：目标距离
- collision cost：障碍安全距离风险
- altitude cost：偏离期望高度惩罚
- smooth cost：轨迹平滑
- energy cost：速度/升降能耗

## 设计目的

避免无人机通过无限升高规避障碍，引入高度约束和高度变化代价，使无人机在水平绕行和垂直避障之间进行合理选择。
