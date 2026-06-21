# EGO-like Planner 骨架与分阶段填充原则

这个分支的目标不是把 planner 收缩成一个小的静态单机版本，而是先保留完整 EGO-like 分层 planner 骨架，再按阶段填充各层内容。

换句话说：

```text
完整框架要在：
perception_interface
→ goal_manager
→ frontend_planner
→ safety_checker
→ backend_selector
→ fallback_manager
→ command_output
→ swarm_interface

但不是每一层一开始都必须写满。
```

## 1. 总原则

```text
骨架完整保留
接口提前定义
核心链路先跑通
复杂能力分阶段填充
```

这样做的目的，是避免继续在一个大文件里到处加 if，也避免一上来写一个过大的 planner 导致调试失控。

## 2. 各模块当前填充状态

| 模块 | 框架中是否保留 | 当前阶段填充程度 | 后续扩展 |
|---|---:|---|---|
| perception_interface | 是 | 静态/动态 track、其他无人机状态、近场 LiDAR 摘要接口 | 点云裁剪、local map、risk map |
| goal_manager | 是 | final_goal 到 local_goal | manual waypoints、active waypoint switcher |
| frontend_planner | 是 | 轻量 primitive/候选速度生成 | local A* 接入、kinodynamic primitive、轨迹库 |
| safety_checker | 是 | clearance、static clearance、near-field、TTC 基础检查 | 动态障碍时空检查、多机轨迹冲突、动力学可行性 |
| backend_selector | 是 | goal/clearance/TTC/smoothness 评分 | progress cost、path deviation、拓扑路径选择 |
| fallback_manager | 是 | hover、escape、climb fallback | lateral escape、更稳定的 wait-and-replan |
| command_output | 是 | 速度平滑、限速、加速度限制 | ENU/NED 显式转换、yaw/heading 处理 |
| swarm_interface | 是 | 预留 committed trajectory 接口 | EGO-Swarm/MADER 风格轨迹共享与 recheck |

## 3. 正确的数据流顺序

安全检查必须在后端评分之前：

```text
state + perception
↓
goal_manager 生成 local_goal
↓
frontend_planner 生成候选轨迹/候选速度
↓
safety_checker 做硬过滤
↓
backend_selector 只在安全候选中评分选择
↓
如果无安全候选，fallback_manager 接管
↓
command_output 平滑并发布
```

关键点：

```text
clearance cost 是软约束
collision / hard clearance check 是硬约束
危险轨迹不能只扣分，必须先被过滤掉
```

## 4. 阶段化填充路线

### V2.1：完整骨架 + 行为尽量不变

目标：

```text
先确认模块边界清楚，节点能运行，输出报告可信。
```

主要做：

```text
1. 保留完整模块目录
2. 原有能跑的 planner 不被替换
3. 新 framework 先以 observe/report 模式运行
4. smoke test 能通过
```

### V2.2：单机静态核心链路

目标：

```text
单机、静态障碍、短路径和中等路径稳定。
```

主要填充：

```text
perception_interface: 点云/静态障碍输入
local_map_builder: 局部占据/膨胀
safety_checker: 静态 path clearance hard reject
fallback_manager: hover/back/lateral_escape
command_output: 平滑与限幅
```

### V2.3：长路径与对角线路径

目标：

```text
不再被 final_goal 强拉，支持分段通过障碍区。
```

主要填充：

```text
goal_manager:
  manual waypoint list
  active waypoint switcher
  local_goal_generator
```

### V2.4：动态障碍

目标：

```text
对动态障碍做时空碰撞检查，而不是只看当前距离。
```

主要填充：

```text
dynamic obstacle prediction
TTC / time-indexed clearance
future trajectory risk
```

### V2.5：多机集群

目标：

```text
其他无人机不再只是当前位置障碍，而是短时未来轨迹约束。
```

主要填充：

```text
swarm_interface:
  publish own committed trajectory
  receive other committed trajectories
  multi_uav_collision_check
  conflict_recheck
```

## 5. 当前不需要一次性写满的部分

这些模块可以保留接口，但暂时不要求完整实现：

```text
1. 完整 dynamic_collision_check
2. 完整 multi_uav_collision_check
3. MADER/RMADER 式 check-recheck
4. 完整 kinodynamic trajectory optimization
5. PointFlow / 学习式预测
6. 编队重构策略
```

不是删除这些模块，而是暂时让它们处于 placeholder / simple adapter 状态。

## 6. 当前最重要的落地点

下一步最优先填：

```text
1. goal_manager 的 manual waypoints / waypoint_switcher
2. safety_checker 的 hard reject 逻辑和日志
3. fallback_manager 的 lateral_escape
4. command_output 的坐标系检查与限幅
```

这样既保留完整 planner 架构，又不会因为一次性实现动态、多机、预测、优化而把系统复杂度拉爆。
