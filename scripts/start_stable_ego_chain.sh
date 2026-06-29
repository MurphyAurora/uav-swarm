#!/usr/bin/env bash
set -euo pipefail

# Stable execution chain for the EGO-like local planner:
#   ego_like_planner_framework -> /primitive_cmd_vel_ned
#   lidar_ttc_safety_filter    -> /safe_cmd_vel_ned
#   multi_waypoint2_arbiter    -> /cruise_cmd_vel_ned
#   command_arbiter            -> /cmd_vel_ned
#
# Usage:
#   bash scripts/start_stable_ego_chain.sh [num_drones] [extra ros2 launch args...]
# Example:
#   bash scripts/start_stable_ego_chain.sh 3 gz_gui:=true scene_freeze_dynamics:=1

NUM_DRONES="${1:-5}"
if [[ $# -gt 0 ]]; then
  shift || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${WS_ROOT}/logs"
mkdir -p "${LOG_DIR}"

source /opt/ros/jazzy/setup.bash
source "${WS_ROOT}/install/setup.bash"

export COMMAND_ARBITER_ENABLE=1
export EGO_LIKE_ENABLE=1
# In the stable chain, mission outputs cruise proposals instead of writing directly
# to /cmd_vel_ned. The arbiter is the only final command publisher.
export PYTHONUNBUFFERED=1

ARB_LOG="${LOG_DIR}/command_arbiter_stable.log"
LAUNCH_LOG="${LOG_DIR}/stable_ego_chain_launch.log"

cleanup() {
  if [[ -n "${ARB_PID:-}" ]] && kill -0 "${ARB_PID}" >/dev/null 2>&1; then
    kill "${ARB_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

ros2 run xtd2_mission command_arbiter \
  --num-drones "${NUM_DRONES}" \
  --max-age-sec 0.8 \
  --rate 20 \
  --escape-hold-sec 0.8 \
  --recovery-hold-sec 0.8 \
  >"${ARB_LOG}" 2>&1 &
ARB_PID=$!

echo "[stable-chain] command_arbiter started: pid=${ARB_PID}, log=${ARB_LOG}"
echo "[stable-chain] launching swarm_simulation_launch.py with num_drones=${NUM_DRONES}"

ros2 launch xtd2_mission swarm_simulation_launch.py \
  num_drones:="${NUM_DRONES}" \
  ego_like_enable:=1 \
  motion_primitive_enable:=0 \
  lidar_ttc_enable:=1 \
  avoid_source:=primitive \
  "$@" \
  2>&1 | tee "${LAUNCH_LOG}"
