# XTDrone2 Mission Architecture (Frozen Interface)

## Control ownership

Only `command_arbiter` publishes final vehicle commands:

```
Mission layer
  multi_waypoint2 / mission_controller
          |
          v
  cruise_cmd_vel_ned

Local planner layer
  motion_primitive_selector
          |
          v
  primitive_cmd_vel_ned

Safety layer
  lidar_ttc_safety_filter
          |
          v
  safe_cmd_vel_ned

          v
  command_arbiter

          v
  cmd_vel_ned

          v
  PX4
```

## Rules

- Mission nodes must not publish `cmd_vel_ned`.
- Local planners publish command proposals only.
- Safety modules can override lower priority proposals.
- PX4 interface remains unchanged.
