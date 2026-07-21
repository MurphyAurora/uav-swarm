# Windows 3D Motion Primitive Validation

该目录用于在 Windows + Python 环境下验证三维 Motion Primitive 避障算法，不依赖 ROS2/PX4/Gazebo。

## Current Focus

当前阶段保持 Motion Primitive + Cost Function 架构，但把离散动作扩展为速度空间采样：

- continuous velocity sampled primitive
- short-horizon kinematic rollout
- spatio-temporal collision checking
- previous-velocity transition cost
- risk-aware altitude weight scheduling
- top clearance cost for over-obstacle motion
- deterministic 3D validation scenes
- lifted 2D benchmark maps as 3D cylinder scenes

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

低障碍侧绕验证。期望 `vz` 基本接近 `0`，`z` 维持在 `3m` 附近：

```powershell
python windows_3d_mp_test\simulation.py --scenario test_side_bypass
```

高障碍越顶验证。该场景有窄 XY bounds，水平绕行空间受限，期望出现升高、通过、下降恢复：

```powershell
python windows_3d_mp_test\simulation.py --scenario test_over_top
```

二维地图提升为三维柱体地图：

```powershell
python windows_3d_mp_test\simulation.py --scenario s_curve_easy --pillar-height 5
python windows_3d_mp_test\simulation.py --scenario s_curve_medium --pillar-height 5
python windows_3d_mp_test\simulation.py --scenario forest_gap --pillar-height 5
python windows_3d_mp_test\simulation.py --scenario random_forest_sparse --pillar-height 5
python windows_3d_mp_test\simulation.py --scenario random_forest_medium --pillar-height 5
python windows_3d_mp_test\simulation.py --scenario random_forest_dense --pillar-height 5
```

如果要测试更强的三维越障倾向，可以提高柱体高度：

```powershell
python windows_3d_mp_test\simulation.py --scenario s_curve_medium --pillar-height 8
python windows_3d_mp_test\simulation.py --scenario random_forest_medium --pillar-height 8
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

只看终端调试输出、不弹出图像窗口：

```powershell
python windows_3d_mp_test\simulation.py --scenario test_over_top --no-plot
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
  + wtop*J_top_clearance
```

其中：

- goal cost: final distance to goal
- collision cost: cylinder/sphere obstacle clearance over rollout time
- altitude cost: keep reference height with a risk-aware weight schedule
- smooth cost: rollout smoothness
- energy cost: velocity and climb/descent effort
- transition cost: squared difference between current and previous primitive velocity
- top clearance cost: when inside the obstacle XY region, enforce `z > obstacle_height + safety_margin`

## Debug Output

每一步会打印：

```text
primitive: vx=... vy=... vz=... goal=... collision=... height=... transition=... top_clearance=... risk_level=...
```

判断逻辑：

- `test_side_bypass` 中 `vz` 长期接近 `0`：说明侧绕没有被无意义爬升污染
- `top_clearance` 高且 `vz` 仍不升：说明越顶安全代价或预测时间还不够
- `transition` 高且 `vz` 正负切换：说明动作连续性还需要继续加强
- `max height` 很高但 `top_clearance` 已经低：说明高度上限或能量项需要继续调

## Validation Criteria

- `no_obstacle`: 轨迹保持在 `z≈3`
- `test_side_bypass`: 低障碍优先平滑水平绕行，避免随机爬升
- `test_over_top`: 高障碍在水平空间受限时升高，满足顶部安全间隙，通过后下降恢复
- `s_curve_*` / `forest_*`: 二维多障碍地图被提升为三维柱体后，检查连续绕行、局部越障和高度恢复
- 禁止无限升高飞行
- 禁止 climb/descend 高频震荡
- 禁止大角度折线轨迹
