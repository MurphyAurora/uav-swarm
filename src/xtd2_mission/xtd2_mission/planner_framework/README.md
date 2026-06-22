# Planner Framework Layout

This package keeps the planner pipeline split by responsibility so frontend
decision logic can be replaced without changing the ROS bridge or safety output.

Current pipeline:

```text
ego_like_planner_framework_node
  -> perception_interface
     -> lidar point conversion
     -> static/dynamic/swarm obstacle aggregation
     -> local_map_builder
  -> goal_manager
     -> final goal / waypoint tracking
     -> short-horizon local goal generation
  -> frontend_planner
     -> local_astar
     -> primitive/candidate trajectory generation
  -> safety_checker
     -> static collision check
     -> dynamic collision check
     -> multi-UAV collision check
     -> kinodynamic feasibility check
  -> backend_selector
     -> cost_goal
     -> cost_clearance
     -> cost_ttc
     -> cost_smoothness
     -> best trajectory selection
  -> fallback_manager
     -> hover / brake / back / lateral escape / climb
  -> command_output
     -> velocity smoothing
     -> acceleration limiting
     -> cmd_vel_ned publication by the ROS node
```

`frontend_mode=hybrid_astar` is the default for LiDAR forest validation. It
adds a stateful local A* frontend ahead of the primitive candidates, while
keeping the existing safety checker, backend selector, fallback manager, and
output filter in the loop.

Single-UAV static-loop state currently lives in `local_astar.py`:

```text
local occupancy map
  -> A* local path search
  -> committed path latch
  -> path validity recheck
  -> progress / stuck monitor
  -> short recovery escape
  -> lookahead velocity candidate
```

Future planners such as RRT, DWA, or an EGO-style optimizer should plug in at
the same frontend boundary and return `CandidateTrajectory` objects.
