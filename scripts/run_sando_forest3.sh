#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the local SANDO forest3-with-walls experiment.
#
# Usage:
#   ./scripts/run_sando_forest3.sh          # x500 with 3D GPU LiDAR
#   ./scripts/run_sando_forest3.sh normal   # normal x500 without LiDAR

MODE="${1:-lidar3d}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DYNAMIC_OBS_ENABLE="${DYNAMIC_OBS_ENABLE:-1}"
DYNAMIC_OBS_FREEZE="${DYNAMIC_OBS_FREEZE:-1}"
NUM_DRONES="${NUM_DRONES:-1}"
LEADER_ID="${LEADER_ID:-1}"
LOCAL_PLANNER_MODE="${LOCAL_PLANNER_MODE:-risk_astar}"
STOP_ON_STAGE_TIMEOUT="${STOP_ON_STAGE_TIMEOUT:-1}"
STATIC_LOCAL_PLANNER="${STATIC_LOCAL_PLANNER:-0}"

# Static-local-planner test mode is meant to quickly validate whether the new
# planner can command visible single-UAV motion.  Keep the mission window long,
# reduce the pre-mission wait, increase the visible cruise speed, and disable
# legacy advisory velocity nodes so the log is easier to read.
if [[ "${STATIC_LOCAL_PLANNER}" == "1" || "${STATIC_LOCAL_PLANNER}" == "true" || "${STATIC_LOCAL_PLANNER}" == "TRUE" ]]; then
  RUN_DURATION="${STATIC_PLANNER_DURATION:-180}"
  WARMUP_SEC="${WARMUP_SEC:-12}"
  POST_OFFBOARD_HOLD_SEC="${POST_OFFBOARD_HOLD_SEC:-1}"
  MOTION_PRIMITIVE_ENABLE="${MOTION_PRIMITIVE_ENABLE:-0}"
  LIDAR_TTC_ENABLE="${LIDAR_TTC_ENABLE:-0}"
  export STATIC_PLANNER_MAX_SPEED="${STATIC_PLANNER_MAX_SPEED:-0.55}"
  export STATIC_PLANNER_INFLATION="${STATIC_PLANNER_INFLATION:-0.60}"
  export STATIC_PLANNER_RESOLUTION="${STATIC_PLANNER_RESOLUTION:-0.25}"
else
  RUN_DURATION="${DURATION:-32}"
  WARMUP_SEC="${WARMUP_SEC:-25}"
  POST_OFFBOARD_HOLD_SEC="${POST_OFFBOARD_HOLD_SEC:-3}"
  MOTION_PRIMITIVE_ENABLE="${MOTION_PRIMITIVE_ENABLE:-1}"
  LIDAR_TTC_ENABLE="${LIDAR_TTC_ENABLE:-1}"
fi

export LOCAL_PLANNER_MODE
export STOP_ON_STAGE_TIMEOUT
export STATIC_LOCAL_PLANNER

case "${MODE}" in
  normal)
    PX4_SIM_MODEL="gz_x500"
    echo "[WARN] normal mode has no 3D LiDAR; unknown-obstacle avoidance cannot be validated reliably."
    echo "[WARN] Use: ./scripts/run_sando_forest3.sh lidar3d"
    ;;
  lidar3d)
    PX4_SIM_MODEL="gz_x500_lidar_3d"
    ;;
  *)
    echo "Usage: $0 [normal|lidar3d]" >&2
    exit 2
    ;;
esac

pkill -f "gz sim" 2>/dev/null || true
pkill -f "gz gui" 2>/dev/null || true
pkill -f "ruby.*gz" 2>/dev/null || true
pkill -f "PX4 -i" 2>/dev/null || true
pkill -f px4_sitl_default 2>/dev/null || true
pkill -f MicroXRCEAgent 2>/dev/null || true

cd "${WS_ROOT}"
"${SCRIPT_DIR}/worlds/install_sando_worlds.sh"

# Apply the controller-side LiDAR unknown-obstacle safety patch before launch.
# This keeps the experiment usable even when the working tree was pulled before
# the controller file was manually patched.
python3 "${SCRIPT_DIR}/apply_lidar_risk_astar_safety_patch.py"

set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

echo "[INFO] mode=${MODE}, px4_sim_model=${PX4_SIM_MODEL}"
echo "[INFO] num_drones=${NUM_DRONES}"
echo "[INFO] leader_id=${LEADER_ID}"
echo "[INFO] local_planner_mode=${LOCAL_PLANNER_MODE}"
echo "[INFO] static_local_planner=${STATIC_LOCAL_PLANNER}"
echo "[INFO] static_planner_max_speed=${STATIC_PLANNER_MAX_SPEED:-unset}"
echo "[INFO] static_planner_inflation=${STATIC_PLANNER_INFLATION:-unset}"
echo "[INFO] stop_on_stage_timeout=${STOP_ON_STAGE_TIMEOUT}"
echo "[INFO] warmup_sec=${WARMUP_SEC}, post_offboard_hold_sec=${POST_OFFBOARD_HOLD_SEC}"
echo "[INFO] run_duration=${RUN_DURATION}"
echo "[INFO] motion_primitive_enable=${MOTION_PRIMITIVE_ENABLE}, lidar_ttc_enable=${LIDAR_TTC_ENABLE}"
echo "[INFO] dynamic_obs_enable=${DYNAMIC_OBS_ENABLE}, scene_freeze_dynamics=${DYNAMIC_OBS_FREEZE}"
echo "[INFO] default keeps scene obstacles published and freezes moving obstacles for static-column validation"
echo "[INFO] set DYNAMIC_OBS_FREEZE=0 to re-enable moving scene obstacles"
if [ "${MODE}" = "lidar3d" ]; then
  echo "[INFO] After drones are spawned, open another terminal and run:"
  echo "       cd ${WS_ROOT} && ./scripts/bridge_x500_lidar_3d.sh ${NUM_DRONES}"
fi

exec ros2 launch xtd2_mission swarm_simulation_launch.py \
  num_drones:="${NUM_DRONES}" \
  leader_id:="${LEADER_ID}" \
  gz_world:=sando_forest3_walls_xtd2 \
  gz_gui:=0 \
  px4_sim_model:="${PX4_SIM_MODEL}" \
  px4_use_versioned_local_position:=1 \
  dynamic_obs_enable:="${DYNAMIC_OBS_ENABLE}" \
  dynamic_obs_mode:=scene \
  scene_freeze_dynamics:="${DYNAMIC_OBS_FREEZE}" \
  dynamic_obs_visualize_gz:=0 \
  spawned_formation_axis:=x \
  duration:="${RUN_DURATION}" \
  warmup_sec:="${WARMUP_SEC}" \
  post_offboard_hold_sec:="${POST_OFFBOARD_HOLD_SEC}" \
  motion_primitive_enable:="${MOTION_PRIMITIVE_ENABLE}" \
  lidar_ttc_enable:="${LIDAR_TTC_ENABLE}" \
  scene_config:="${WS_ROOT}/scripts/scenes/sando/forest3_walls_dynamic.yaml" \
  dynamic_obs_wall_x:=27.0 \
  start_wall_clearance:=8.0 \
  dynamic_obs_wall_y:=-14.0 \
  dynamic_obs_target_x:=19.0 \
  dynamic_obs_target_y_base:=30.0
