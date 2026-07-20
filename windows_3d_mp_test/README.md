# Windows 3D Motion Primitive Validation

该目录用于在 Windows + Python 环境下验证三维 Motion Primitive 避障算法，不依赖 ROS2/PX4/Gazebo。

## Current Focus

当前阶段不是重新设计二维场景，而是继承 `scripts/offline_scenarios` 中已经验证过的二维 XY 障碍分布，并补充：

- 3D cylinder obstacle model
- dynamic obstacle motion model
- cylindrical collision checking
- 3D trajectory visualization
- lifted `forest_gap` validation scene

## Structure

```text
windows_3d_mp_test/
├── primitive.py
├── obstacle.py
├── cost_function.py
├── planner.py
├── scene_generator.py
├── visualize.py
├── simulation.py
├── experiment.py
└── README.md
```

## Run

在仓库根目录运行：

```powershell
cd D:\uav-swarm
python windows_3d_mp_test\simulation.py --scenario forest_gap
```

高柱验证：

```powershell
python windows_3d_mp_test\simulation.py --scenario forest_gap_high
```

地面动态障碍验证：

```powershell
python windows_3d_mp_test\simulation.py --scenario forest_gap_dynamic
```

飞行动态障碍验证：

```powershell
python windows_3d_mp_test\simulation.py --scenario forest_gap_flying
```

## Planning Flow

```text
3D scene
    |
Dynamic obstacle model
    |
Motion primitive rollout
    |
Spatio-temporal collision checking
    |
Risk-aware cost evaluation
    |
Lowest-cost feasible primitive
    |
Safe trajectory
```

## Cost Function

```text
J = wg*J_goal
  + wc*J_collision
  + wh*J_altitude
  + ws*J_smooth
  + we*J_energy
  + wr*J_risk
```

其中：

- goal cost: final distance to goal
- collision cost: cylinder/sphere obstacle clearance
- altitude cost: deviation from reference height
- smooth cost: acceleration-like trajectory smoothness
- energy cost: velocity and climb/descent effort
- risk cost: deterministic predicted obstacle risk

## Stage-1 Validation Criteria

- `forest_gap` 的 XY 障碍分布来自 `scripts/offline_scenarios/static_cases.py`
- 二维圆障碍被提升为 z=0 起始的三维柱体
- 柱体可视化包含侧面、顶面和底面
- 碰撞检测区分柱体障碍和飞行障碍
- Planner 会跳过硬碰撞 primitive
- 输出最大高度和高度变化量，用于检查是否出现无限爬升或频繁升降
