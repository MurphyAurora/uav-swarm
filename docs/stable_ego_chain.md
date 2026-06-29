# Stable EGO-like planner execution chain

This branch should run one clear command path during normal experiments. The goal is to avoid mixed control from legacy ORCA, legacy motion primitives, direct mission velocity output, and the new EGO-like planner.

## Stable chain

```text
swarm_state_exchange
  -> obstacle_tracker
  -> trajectory_filter
  -> ego_like_planner_framework
       publishes /xtdrone2/x500_{id}/primitive_cmd_vel_ned
  -> lidar_ttc_safety_filter
       publishes /xtdrone2/x500_{id}/safe_cmd_vel_ned
  -> multi_waypoint2_arbiter
       publishes /xtdrone2/x500_{id}/cruise_cmd_vel_ned
  -> command_arbiter
       publishes /xtdrone2/x500_{id}/cmd_vel_ned
  -> PX4 / XTDrone2 communication
```

Only `command_arbiter` should publish the final `/cmd_vel_ned` velocity topic in this mode.

## Why this is more stable

The old branch had several possible command sources:

- `multi_waypoint2` could publish directly to `/cmd_vel_ned`.
- `motion_primitive_selector` could publish `/primitive_cmd_vel_ned`.
- `ego_like_planner_framework` could also publish `/primitive_cmd_vel_ned`.
- `lidar_ttc_safety_filter` could publish `/safe_cmd_vel_ned`.

That made it difficult to determine which command actually controlled the UAV. In the stable chain, each module has one role:

| Module | Role | Output |
|---|---|---|
| `ego_like_planner_framework` | Main local planner | `primitive_cmd_vel_ned` |
| `lidar_ttc_safety_filter` | Near-field TTC shield | `safe_cmd_vel_ned` |
| `multi_waypoint2_arbiter` | Low-priority mission/cruise proposal | `cruise_cmd_vel_ned` |
| `command_arbiter` | Single final command output | `cmd_vel_ned` |

## Recommended run command

From the workspace root after building:

```bash
chmod +x scripts/start_stable_ego_chain.sh
bash scripts/start_stable_ego_chain.sh 3
```

You can pass normal launch overrides after the drone count:

```bash
bash scripts/start_stable_ego_chain.sh 3 gz_gui:=true scene_freeze_dynamics:=1
```

## Expected parameters

The default `src/xtd2_mission/config/swarm_params.yaml` is set for the stable chain:

```yaml
avoid_source: "primitive"
motion_primitive_enable: false
ego_like_enable: true
ego_like_publish_cmd: true
lidar_ttc_enable: true
```

The startup script also sets:

```bash
export COMMAND_ARBITER_ENABLE=1
export EGO_LIKE_ENABLE=1
```

`COMMAND_ARBITER_ENABLE=1` makes `mission_controller` launch `multi_waypoint2_arbiter` instead of the direct-output `multi_waypoint2` process.

## Debug topics

Check these topics in order:

```bash
ros2 topic echo /xtdrone2/x500_1/primitive_cmd_vel_ned
ros2 topic echo /xtdrone2/x500_1/safe_cmd_vel_ned
ros2 topic echo /xtdrone2/x500_1/cruise_cmd_vel_ned
ros2 topic echo /xtdrone2/x500_1/cmd_vel_ned
ros2 topic echo /xtdrone2/x500_1/arbiter_state
```

A healthy run should show:

1. `ego_like_planner_framework` publishing primitive commands.
2. `lidar_ttc_safety_filter` publishing filtered safe commands.
3. `multi_waypoint2_arbiter` publishing cruise proposals, not final commands.
4. `command_arbiter` selecting `safe_primitive` when the planner chain is fresh.

## Important cleanup note

`multi_waypoint2.py` still contains several historical local-planner experiments and multiple definitions of `_unified_lidar_velocity_selector`. In Python, only the last definition is active. The stable chain avoids relying on that experimental selector as the main safety mechanism. Future cleanup should move mission execution and local planning into separate files, but the current stable chain already isolates final command ownership through `command_arbiter`.
