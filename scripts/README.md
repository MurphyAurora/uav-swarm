# Scripts Layout

Common entry points stay in this directory:

```text
run_sando_forest3.sh        current SANDO forest3 launcher
bridge_x500_lidar_3d.sh     Gazebo LiDAR -> ROS2 bridge helper
```

Subdirectories:

```text
tools/          small debug and one-off maintenance tools
tests/          repository smoke tests for planner framework pieces
evaluation/     offline analysis and benchmark entry scripts
scenario/       offline scene generation and plotting scripts
legacy/         old start1-based XTDrone2 scripts kept for reference
difficulty_analysis/
ros_analysis/
scenes/
worlds/
gazebo_assets/
runs/
results/
```

Current planner code does not live in `scripts/`. It lives in:

```text
src/xtd2_mission/xtd2_mission/planner_framework/
```

Useful commands:

```bash
./scripts/run_sando_forest3.sh lidar3d
./scripts/bridge_x500_lidar_3d.sh 1
python3 scripts/tests/test_planner_framework_smoke.py
python3 scripts/tests/test_goal_manager_waypoints.py
python3 scripts/tools/debug_velocity_chain.py
./scripts/evaluation/run_benchmark_v1.sh
```
