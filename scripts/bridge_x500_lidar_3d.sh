#!/usr/bin/env bash
set -euo pipefail

# Bridge Gazebo 3D GPU LiDAR point clouds from x500_lidar_3d models to ROS2.
# Run this after the PX4/Gazebo simulation has started.
#
# Usage:
#   ./scripts/bridge_x500_lidar_3d.sh [num_drones]

NUM_DRONES="${1:-5}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-90}"
POLL_SEC="${POLL_SEC:-3}"
WORLD_NAME="${WORLD_NAME:-sando_forest3_walls_xtd2}"

set +u
source /opt/ros/jazzy/setup.bash
if [ -f "$HOME/ws_xtd2/install/setup.bash" ]; then
  source "$HOME/ws_xtd2/install/setup.bash"
fi
set -u

bridge_pids=()

cleanup() {
  for pid in "${bridge_pids[@]:-}"; do
    if [ -n "$pid" ]; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

start_tf_helpers() {
  echo "[OK] publishing clean per-UAV LiDAR TF from /xtdrone2/swarm/state_exchange"
  python3 "$HOME/ws_xtd2/scripts/lidar_tf_republisher.py" --num-drones "${NUM_DRONES}" &
  bridge_pids+=("$!")
}

find_lidar_topic() {
  local drone_id="$1"
  local topic_list="$2"
  local topic
  topic="$(printf '%s\n' "$topic_list" \
    | grep -E "(x500_${drone_id}|x500_lidar_3d_${drone_id})" \
    | grep -E "(lidar_3d|lidar)" \
    | grep -E "/points$" \
    | sort -r \
    | head -n 1 || true)"
  printf '%s\n' "$topic"
}

deadline=$((SECONDS + MAX_WAIT_SEC))
declare -a topics
while [ "$SECONDS" -le "$deadline" ]; do
  topics=()
  missing=0
  topic_list="$(gz topic -l || true)"

  for drone_id in $(seq 1 "$NUM_DRONES"); do
    topic="$(find_lidar_topic "$drone_id" "$topic_list")"
    topics[$drone_id]="$topic"
    if [ -z "$topic" ]; then
      missing=$((missing + 1))
    fi
  done

  if [ "$missing" -eq 0 ]; then
    break
  fi

  echo "[INFO] waiting for per-UAV Gazebo lidar topics: missing=${missing}/${NUM_DRONES} (${POLL_SEC}s)"
  sleep "$POLL_SEC"
done

start_tf_helpers

for drone_id in $(seq 1 "$NUM_DRONES"); do
  topic="${topics[$drone_id]:-}"
  if [ -z "$topic" ]; then
    echo "[WARN] x500_${drone_id}: no lidar_3d topic found before timeout"
    continue
  fi

  echo "[OK] x500_${drone_id}: bridging $topic"
  ros2 run ros_gz_bridge parameter_bridge \
    "${topic}@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked" \
    --ros-args -r "${topic}:=/x500_${drone_id}/lidar/points" &
  bridge_pids+=("$!")
done

if [ "${#bridge_pids[@]}" -le 1 ]; then
  echo "[ERROR] no LiDAR topics were bridged"
  echo "        Check that launch used: px4_sim_model:=gz_x500_lidar_3d"
  echo "        Restart the simulation after installing the x500_lidar_3d model."
  echo "        Debug with: gz topic -l | grep lidar"
  echo "        Expected Gazebo topics like: /world/.../model/x500_lidar_3d_1/.../scan/points"
  exit 1
fi

wait
