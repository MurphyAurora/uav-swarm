# EGO-like 分层局部重规划框架设计

> 分支：`feature/ego-like-planner-framework`  
> 基础分支：`fix/lidar-risk-astar-safety`  
> 目标：在保留原有局部单机避障能力的基础上，将现有补丁式避障逻辑整理为完整的 EGO-like 分层局部重规划框架。

---

## 1. 当前状态判断

`fix/lidar-risk-astar-safety` 分支已经实现了局部单机避障能力，说明以下链路基本可用：

```text
LiDAR / 局部障碍感知
↓
风险判断 / 安全距离判断
↓
局部 A* / ego-like 引导
↓
安全修正
↓
cmd_vel_ned 输出
```

但当前结构更接近“局部避障补丁集合”，在以下场景中容易失败：

```text
1. 长路径任务
2. 对角线路径任务
3. 动态障碍密集任务
4. 多机集群任务
5. 编队保持与避障同时存在的任务
```

主要原因不是某个参数错误，而是 planner 缺少完整分层结构，导致目标跟踪、避障、安全修正、平滑控制、多机约束之间互相耦合。

---

## 2. 重构原则

本分支不建议推翻已有代码，也不建议马上引入完整 MPC / ORCA / RL / PointFlow / 完整 EGO-Planner。

本阶段采用“渐进式重构”原则：

```text
第一步：结构分层，尽量保持行为不变
第二步：加入 local_goal_manager，解决长路径和对角线路径问题
第三步：独立 safety_checker，防止危险轨迹被执行
第四步：独立 fallback_manager，处理无可行解情况
第五步：加入 command_output 平滑和限幅
第六步：再加入动态障碍预测
第七步：最后扩展多机轨迹共享和冲突检查
```

核心原则：

```text
可以增加输入、外层保护、输出平滑；
不要在 planner 内部继续堆互相冲突的避障补丁。
```

---

## 3. 推荐总体架构

完整 EGO-like planner 建议整理为以下层级：

```text
final_goal / mission target
        ↓
goal_manager / local_goal_generator
        ↓
frontend_planner / candidate trajectory generator
        ↓
safety_checker / feasible trajectory filter
        ↓
backend_selector / best trajectory selection
        ↓
fallback_manager / emergency recovery
        ↓
command_output / smoothing + PX4 command output
```

多机扩展时增加：

```text
swarm_interface
        ↓
other UAV committed trajectories
        ↓
multi-UAV spatiotemporal collision checking
```

---

## 4. 建议代码骨架

可根据当前代码语言和 ROS2 包结构，整理为如下模块：

```text
planner_framework/
├── planner_node.py / planner_node.cpp
├── perception_interface.py / .cpp
├── goal_manager.py / .cpp
├── frontend_planner.py / .cpp
├── backend_selector.py / .cpp
├── safety_checker.py / .cpp
├── fallback_manager.py / .cpp
├── command_output.py / .cpp
└── swarm_interface.py / .cpp
```

如果暂时不想移动文件，也可以先在现有节点中按这些类或函数分区整理。

---

## 5. 各模块需要填充的内容

### 5.1 perception_interface：感知输入层

职责：只负责感知和风险信息提取，不直接决定无人机怎么飞。

需要填充：

```text
1. LiDAR / 点云订阅
2. 障碍物点提取
3. 最近障碍距离计算
4. 前方 / 侧前方障碍判断
5. 障碍物相对方向计算
6. 局部风险区域构建
7. 近场 emergency 触发条件输出
```

输出建议：

```text
nearest_obstacle_distance
nearest_obstacle_direction
front_blocked
left_front_blocked
right_front_blocked
risk_level
emergency_lidar_triggered
local_obstacle_points
```

注意：该层不发布最终控制指令。

---

### 5.2 goal_manager：目标管理层

职责：把最终目标转换成局部目标，避免 planner 被远距离 final_goal 强拉。

需要填充：

```text
1. final_goal 读取
2. current_pose 读取
3. local_goal 生成
4. local_goal 到达判断
5. local_goal 切换
6. 长路径 / 对角线路径分段引导
7. 障碍物附近 local_goal 偏移策略
```

基本逻辑：

```text
if distance(current_pose, final_goal) > local_goal_radius:
    local_goal = current_pose + direction_to_goal * local_goal_radius
else:
    local_goal = final_goal
```

如果前方 local_goal 被障碍物阻挡，可做轻量偏移：

```text
if front_blocked:
    choose left/right shifted local_goal according to lower risk side
```

建议先使用简单策略，不要一开始引入复杂全局规划。

---

### 5.3 frontend_planner：前端候选轨迹生成层

职责：生成“可能怎么走”的候选轨迹或候选速度。

可填充：

```text
1. 当前已有局部 A* 引导
2. ego-like 轨迹生成
3. primitive rollout
4. 若暂时只有一条轨迹，也先通过该层输出
```

输出形式建议：

```text
candidate_trajectories = [traj_1, traj_2, traj_3, ...]
```

或者轻量版本：

```text
candidate_commands = [cmd_1, cmd_2, cmd_3, ...]
```

第一阶段重点是接口清晰，不要求候选数量很多。

---

### 5.4 safety_checker：安全检查层

职责：判断候选轨迹能不能执行。

该层是硬约束，不安全的轨迹直接删除，不参与软评分。

需要填充：

```text
1. 最小安全距离检查
2. LiDAR 近场碰撞检查
3. 静态障碍碰撞检查
4. TTC 检查
5. 动力学可行性检查
6. 后续动态障碍预测检查
7. 后续多机轨迹冲突检查
```

建议逻辑：

```text
safe_candidates = []
for traj in candidate_trajectories:
    if collision_free(traj) and ttc_safe(traj) and kinodynamic_feasible(traj):
        safe_candidates.append(traj)
```

硬规则：

```text
if emergency_lidar_triggered:
    reject forward-moving trajectories

if min_clearance < emergency_clearance:
    reject trajectory

if ttc < emergency_ttc:
    reject trajectory
```

注意：不要在 safety_checker 中做“勉强修正后继续飞”。安全层只做通过 / 否决。

---

### 5.5 backend_selector：后端评价 / 选择层

职责：在已经安全的候选轨迹中选择最好的一条。

需要填充：

```text
1. goal_cost：接近 local_goal
2. clearance_cost：远离障碍
3. ttc_cost：避开即将碰撞方向
4. smooth_cost：减少速度突变
5. path_deviation_cost：减少不必要绕行
6. progress_cost：保持任务推进
```

建议代价函数：

```text
cost =
    w_goal * goal_cost
  + w_clearance * clearance_cost
  + w_ttc * ttc_cost
  + w_smooth * smooth_cost
  + w_progress * progress_cost
```

注意：backend_selector 只在 safe_candidates 非空时运行。

---

### 5.6 fallback_manager：失败恢复层

职责：当没有安全轨迹时，执行兜底动作。

需要填充：

```text
1. hover
2. brake
3. back
4. climb
5. side_step
6. wait_and_replan
```

必须满足：

```text
if safe_candidates is empty:
    ignore goal tracking
    execute fallback
```

不要再出现：

```text
无可行轨迹，但仍然继续朝 final_goal 推进
```

推荐 fallback 优先级：

```text
1. brake / hover
2. 如果前方障碍持续逼近，back
3. 如果平面方向无解，climb
4. 等待动态障碍通过后重新规划
```

---

### 5.7 command_output：控制输出层

职责：将轨迹或速度命令转换成 PX4 / ROS2 可执行指令。

需要填充：

```text
1. 轨迹到速度命令转换
2. cmd_vel_ned 发布
3. NED / ENU 坐标转换检查
4. 速度限幅
5. 加速度限幅
6. 低通滤波
7. yaw / heading 处理
```

建议加入：

```text
v_cmd = alpha * v_new + (1 - alpha) * v_last
```

以及：

```text
|v_cmd - v_last| <= a_max * dt
```

重点检查坐标系：

```text
ROS / Gazebo 常用 ENU
PX4 常用 NED
```

对角线路径失败时，尤其要检查 NED/ENU 转换、障碍物相对角度、速度方向是否一致。

---

### 5.8 swarm_interface：多机接口层

第一阶段可先保留接口，不要求实现。

后续需要填充：

```text
1. 发布本机短时轨迹
2. 接收其他无人机短时轨迹
3. 将其他无人机作为动态障碍
4. 多机间时空碰撞检查
5. 简单优先级让行机制
6. committed trajectory
7. check-recheck
```

轻量实现建议：

```text
other_uav_pred(t) = other_uav_pos + other_uav_vel * t
```

先实现“其他无人机作为动态障碍”，再考虑复杂编队重构。

---

## 6. planner_node 主循环建议

主循环可以整理为：

```text
while running:

    perception_data = perception_interface.update()

    local_goal = goal_manager.get_local_goal(
        current_state,
        final_goal,
        perception_data
    )

    candidates = frontend_planner.generate_candidates(
        current_state,
        local_goal,
        perception_data
    )

    safe_candidates = safety_checker.filter(
        candidates,
        perception_data
    )

    if safe_candidates is empty:
        cmd = fallback_manager.get_fallback_command(
            current_state,
            perception_data
        )
    else:
        best_traj = backend_selector.select_best(
            safe_candidates,
            local_goal,
            current_state
        )

        cmd = command_output.convert_to_command(best_traj)

    cmd = command_output.smooth_and_limit(cmd)

    command_output.publish(cmd)
```

关键点：

```text
safety_checker 在 backend_selector 前面；
fallback_manager 独立于普通评分；
command_output 只做输出转换和平滑，不做高层决策。
```

---

## 7. 推荐开发阶段

### 阶段 1：结构整理，行为尽量不变

目标：把已有逻辑按层级拆开，但不改变主要参数和运行效果。

完成标志：

```text
原来的局部单机避障测试仍然可以成功
```

---

### 阶段 2：加入 local_goal_manager

目标：解决长路径、对角线路径被 final_goal 强拉的问题。

测试：

```text
单机短路径
单机中距离路径
单机长路径
单机对角线路径
```

---

### 阶段 3：独立 safety_checker 和 fallback_manager

目标：解决“看到了障碍但仍然执行危险动作”的问题。

必须验证：

```text
无安全候选轨迹时，不再继续追目标
near-field LiDAR 触发时，planner 输出被覆盖
```

---

### 阶段 4：加入 command_output 平滑和限幅

目标：解决路径抖动、速度突变、对角线方向不稳的问题。

验证：

```text
速度曲线是否连续
cmd_vel_ned 是否突变
轨迹是否明显左右抖动
```

---

### 阶段 5：加入动态障碍预测

目标：支持低速 / 中速动态障碍。

先用简单模型：

```text
p_obs(t) = p_obs(0) + v_obs * t
```

不要一开始上复杂 PointFlow 或深度学习预测。

---

### 阶段 6：加入多机扩展

目标：将其他无人机作为动态障碍处理。

先做：

```text
2 机不互撞
3 机不互撞
5 机基本可达
```

不要一开始追求复杂编队重构。

---

## 8. 实验验证顺序

建议每次修改后按以下顺序测试：

```text
1. 单机短路径 + 静态障碍
2. 单机对角线路径 + 静态障碍
3. 单机长路径 + 静态障碍
4. 单机低速动态障碍
5. 单机中速动态障碍
6. 双机交叉路径
7. 三机 / 五机集群路径
```

每个阶段记录：

```text
collision_count
min_clearance
emergency_trigger_count
success_rate
average_speed
planning_time
cmd_vel_smoothness
```

---

## 9. 当前阶段不建议做的事情

暂时不要加入：

```text
1. 完整 MPC
2. ORCA 重新接管最终控制
3. RL 决策
4. PointFlow / 深度学习预测
5. 复杂编队重构
6. 完整 EGO-Planner B-spline 优化移植
7. 完整 MADER/RMADER 优化框架
```

原因：当前优先级是稳定 ego-like 分层闭环，而不是继续引入新的控制源或复杂优化器。

---

## 10. 论文表述建议

可以将该方法描述为：

> 本文在 EGO-Planner / EGO-Swarm 局部重规划思想基础上，构建面向无人机集群动态避障的轻量化 EGO-like 分层局部规划框架。该框架将局部目标生成、前端候选轨迹生成、后端轨迹评价、安全可行性检查、失败恢复机制和控制输出平滑解耦，避免补丁式避障方法中多约束互相冲突的问题。针对多无人机场景，后续进一步参考 MADER/RMADER 的 committed trajectory 思想，将其他无人机的短时未来轨迹作为时空约束输入，实现去中心化集群避障。

---

## 11. 本分支近期任务清单

优先任务：

```text
[ ] 确认当前分支能复现 fix/lidar-risk-astar-safety 的单机局部避障结果
[ ] 梳理现有 planner 相关文件和节点
[ ] 标记 LiDAR 风险计算代码位置
[ ] 标记 A* / ego-like 局部规划代码位置
[ ] 标记 safety / emergency 相关代码位置
[ ] 标记 cmd_vel_ned 输出代码位置
[ ] 按模块边界拆分 planner_framework
[ ] 加入 local_goal_manager
[ ] 独立 safety_checker
[ ] 独立 fallback_manager
[ ] 加入 command_output 平滑与限幅
```

后续任务：

```text
[ ] 加入动态障碍简单预测
[ ] 加入其他无人机轨迹预测接口
[ ] 加入 multi_uav_collision_check
[ ] 加入简单让行机制
[ ] 加入 committed trajectory 机制
[ ] 完成多机集群测试
```
