# Windows 3D Motion Primitive Validation

该目录用于在 Windows + Python 环境下验证三维 Motion Primitive 避障算法，不依赖 ROS2/PX4/Gazebo。

## Current Focus

当前阶段保持 Motion Primitive + Cost Function 架构，但把离散动作扩展为速度空间采样：

- continuous velocity sampled primitive
- short-horizon kinematic rollout
- spatio-temporal collision checking
- previous-velocity transition cost
- altitude deviation cost around `z_ref=3`
- vertical motion cost for `vz` and `delta_vz`
- top clearance cost for over-obstacle motion
- deterministic and randomized mixed-height 3D validation scenes
- lifted 2D benchmark maps as 3D cylinder scenes
- static map feasibility precheck for random scene filtering
- CSV logs and JSON summaries for trajectory/cost metrics

## Structure

```text
windows_3d_mp_test/
├── primitive.py
├── obstacle.py
├── cost_function.py
├── feasibility.py
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

固定混合高度场景。部分障碍较低，部分障碍较高，用于避免实验全部变成绕行或全部变成爬升：

```powershell
python windows_3d_mp_test\simulation.py --scenario s_curve_mixed_height
python windows_3d_mp_test\simulation.py --scenario forest_gap_mixed_height
python windows_3d_mp_test\simulation.py --scenario random_forest_mixed_sparse
python windows_3d_mp_test\simulation.py --scenario random_forest_mixed_medium
python windows_3d_mp_test\simulation.py --scenario random_forest_mixed_dense
```

对任意二维地图随机采样每个障碍的高度：

```powershell
python windows_3d_mp_test\simulation.py --scenario s_curve_medium --height-range 2 8 --height-seed 7
python windows_3d_mp_test\simulation.py --scenario forest_gap --height-range 2 8 --height-seed 11
python windows_3d_mp_test\simulation.py --scenario random_forest_medium --height-range 2 8 --height-seed 21
```

随机或密集地图建议先看可行性预检查。默认会打印 `feasible_xy`、`feasibility_reason`、`xy_min_clearance`：

```powershell
python windows_3d_mp_test\simulation.py --scenario random_forest_mixed_dense --no-plot
```

如果只想统计可行静态地图，使用 `--require-feasible`。不可行地图会被标记为 `status=infeasible_map` 并跳过规划：

```powershell
python windows_3d_mp_test\simulation.py --scenario random_forest_mixed_dense --require-feasible --no-plot --summary-file logs\dense_prechecked_summary.json
```

预检查参数：

```powershell
python windows_3d_mp_test\simulation.py --scenario random_forest_medium --height-range 2 8 --height-seed 21 --feasibility-resolution 0.15 --feasibility-margin 0.8
```

说明：`feasibility.py` 只做 XY 平面膨胀障碍连通性检查，不替代 3D planner。它用于剔除随机生成时本身无路的地图，避免把地图不可行误算成 planner 失败。

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

同时保存 CSV 日志和 JSON 汇总：

```powershell
python windows_3d_mp_test\simulation.py --scenario s_curve_easy --pillar-height 3.5 --no-plot --log-file logs\s_curve_easy_3d.csv --summary-file logs\s_curve_easy_3d_summary.json
python windows_3d_mp_test\simulation.py --scenario random_forest_medium --height-range 2 8 --height-seed 21 --no-plot --log-file logs\forest_medium_3d.csv --summary-file logs\forest_medium_3d_summary.json
python windows_3d_mp_test\simulation.py --scenario test_over_top --no-plot --log-file logs\test_over_top.csv --summary-file logs\test_over_top_summary.json
```

CSV 字段包括：

```text
step,time,x,y,z,vx,vy,vz,total_cost,goal_cost,collision_cost,altitude_deviation_cost,height_cost,smooth_cost,energy_cost,vertical_motion_cost,transition_cost,top_clearance_cost,minimum_clearance,risk_level,computation_time_ms,status
```

JSON 汇总指标包括：

```text
success_rate,collision_rate,path_length,flight_time,minimum_clearance,max_altitude,height_variation,computation_time,avg_computation_time,max_computation_time,status,steps,feasible_xy,feasibility_reason,xy_path_length,xy_min_clearance
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
  + wa*J_altitude_deviation
  + ws*J_smooth
  + we*J_energy
  + wv*J_vertical_motion
  + wt*J_transition
  + wtop*J_top_clearance
```

其中：

- goal cost: final distance to goal
- collision cost: cylinder/sphere obstacle clearance over rollout time
- altitude deviation cost: keep `z` near `z_ref=3` and softly penalize too high/low flight
- smooth cost: rollout smoothness
- energy cost: velocity and climb/descent effort
- vertical motion cost: penalize `vz` and `delta_vz`
- transition cost: squared difference between current and previous primitive velocity, with extra vertical-change penalty
- top clearance cost: when inside the obstacle XY region, enforce `z > obstacle_height + safety_margin`

## Debug Output

每一步会打印：

```text
primitive: vx=... vy=... vz=... goal=... collision=... altitude=... vertical=... transition=... top_clearance=... min_clearance=... risk_level=...
```

判断逻辑：

- `test_side_bypass` 中 `vz` 长期接近 `0`：说明侧绕没有被无意义爬升污染
- `top_clearance` 高且 `vz` 仍不升：说明越顶安全代价或预测时间还不够
- `vertical` 或 `transition` 高且 `vz` 正负切换：说明动作连续性还需要继续加强
- `max_altitude` 很高但 `top_clearance` 已经低：说明高度上限或能量项需要继续调

## Validation Criteria

- `no_obstacle`: 轨迹保持在 `z≈3`
- `test_side_bypass`: 低障碍优先平滑水平绕行，避免随机爬升
- `test_over_top`: 高障碍在水平空间受限时升高，满足顶部安全间隙，通过后下降恢复
- `s_curve_*` / `forest_*`: 二维多障碍地图被提升为三维柱体后，检查连续绕行、局部越障和高度恢复
- `*_mixed_height` / `--height-range`: 同时验证水平绕行、局部爬升、以及高度恢复
- 静态地图成功率评估应优先统计 `feasible_xy=true` 的地图
- 禁止无限升高飞行
- 禁止 climb/descend 高频震荡
- 禁止大角度折线轨迹
