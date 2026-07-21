# Windows 3D Motion Primitive Validation

该目录用于在 Windows + Python 环境下验证三维 Motion Primitive 避障算法，不依赖 ROS2/PX4/Gazebo。

## Current Focus

当前阶段保持 Motion Primitive + Cost Function 架构，但把离散动作扩展为速度空间采样：

- continuous velocity sampled primitive
- short-horizon kinematic rollout
- spatio-temporal collision checking
- previous-velocity transition cost
- risk-aware altitude cost
- deterministic 3D validation scenes

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

无障碍高度保持验证：

```powershell
python windows_3d_mp_test\simulation.py --scenario no_obstacle
```

单柱绕行/爬升验证。默认单柱半径为 `1.6`：

```powershell
python windows_3d_mp_test\simulation.py --scenario single_pillar
```

进一步扫描半径，观察无人机是升高还是横向绕行：

```powershell
python windows_3d_mp_test\simulation.py --scenario single_pillar --pillar-radius 1.2
python windows_3d_mp_test\simulation.py --scenario single_pillar --pillar-radius 1.8
python windows_3d_mp_test\simulation.py --scenario single_pillar --pillar-radius 2.4
```

高阻挡验证。该场景为 start `(0,0,3)`、goal `(15,0,3)`、柱体中心 `(7.5,0)`、半径 `2`、高度 `8`：

```powershell
python windows_3d_mp_test\simulation.py --scenario high_block
```

只看终端调试输出、不弹出图像窗口：

```powershell
python windows_3d_mp_test\simulation.py --scenario high_block --no-plot
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
3D state
    |
Velocity-space primitive sampling
    |
Kinematic rollout
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
  + wt*J_transition
  + wr*J_risk
```

其中：

- goal cost: final distance to goal
- collision cost: cylinder/sphere obstacle clearance over rollout time
- altitude cost: keep reference height when safe, relax near obstacle risk
- smooth cost: rollout smoothness
- energy cost: velocity and climb/descent effort
- transition cost: squared difference between current and previous primitive velocity
- risk cost: deterministic predicted obstacle risk

## Debug Output

每一步会打印：

```text
primitive: vx=... vy=... vz=... height=... collision=... transition=... risk=... risk_level=...
```

判断逻辑：

- `vz` 长期接近 `0`，且 `collision/risk` 很低：说明水平绕行已经更便宜
- `collision/risk` 高但 `vz` 仍不升：说明高度、能量或 transition 权重仍压制爬升
- `vz` 在正负之间快速切换，且 `transition` 高：说明动作连续性还需要继续加强
- `max height` 很高但 `risk` 已低：说明高度上限或下降恢复项需要继续调

## Validation Criteria

- `no_obstacle`: 轨迹保持在 `z≈3`
- `single_pillar`: 低/中等阻挡时优先平滑水平绕行
- `high_block`: 出现有限高度变化，表现为 `z` 升高、通过障碍、再恢复
- 禁止无限升高飞行
- 禁止 climb/descend 高频震荡
- 禁止大角度折线轨迹
