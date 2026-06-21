#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the local SANDO forest3-with-walls experiment.
#
# Usage:
#   ./scripts/run_sando_forest3.sh          # x500 with 3D GPU LiDAR
#   ./scripts/run_sando_forest3.sh normal   # normal x500 without LiDAR
#
# Static planner coordinate test:
#   STATIC_LOCAL_PLANNER=1 \
#   STATIC_PLANNER_START_X=19.0 STATIC_PLANNER_START_Y=-14.0 \
#   STATIC_PLANNER_TARGET_X=19.0 STATIC_PLANNER_TARGET_Y=30.0 STATIC_PLANNER_TARGET_Z=-3.0 \
#   ./scripts/run_sando_forest3.sh lidar3d
# Coordinates are NED. Only one UAV is spawned in this mode.

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
DYNAMIC_OBS_TARGET_X="${DYNAMIC_OBS_TARGET_X:-19.0}"
DYNAMIC_OBS_TARGET_Y_BASE="${DYNAMIC_OBS_TARGET_Y_BASE:-30.0}"
DYNAMIC_OBS_TARGET_Y_SPACING="${DYNAMIC_OBS_TARGET_Y_SPACING:-2.6}"
MISSION_Z="${MISSION_Z:--3.0}"
SPAWNED_FORMATION_AXIS="${SPAWNED_FORMATION_AXIS:-x}"
SPAWN_OVERRIDE_NED_X=""
SPAWN_OVERRIDE_NED_Y=""

# Static-local-planner test mode is meant to validate whether the local planner
# can command visible single-UAV motion.  Keep the mission window long, reduce
# the pre-mission wait, and disable legacy advisory velocity nodes so the log is
# easier to read.
if [[ "${STATIC_LOCAL_PLANNER}" == "1" || "${STATIC_LOCAL_PLANNER}" == "true" || "${STATIC_LOCAL_PLANNER}" == "TRUE" ]]; then
  RUN_DURATION="${STATIC_PLANNER_DURATION:-180}"
  WARMUP_SEC="${WARMUP_SEC:-12}"
  POST_OFFBOARD_HOLD_SEC="${POST_OFFBOARD_HOLD_SEC:-1}"
  MOTION_PRIMITIVE_ENABLE="${MOTION_PRIMITIVE_ENABLE:-0}"
  LIDAR_TTC_ENABLE="${LIDAR_TTC_ENABLE:-0}"
  export STATIC_PLANNER_MAX_SPEED="${STATIC_PLANNER_MAX_SPEED:-0.55}"
  export STATIC_PLANNER_INFLATION="${STATIC_PLANNER_INFLATION:-0.60}"
  export STATIC_PLANNER_RESOLUTION="${STATIC_PLANNER_RESOLUTION:-0.25}"
fi

if [[ "${STATIC_LOCAL_PLANNER}" == "1" || "${STATIC_LOCAL_PLANNER}" == "true" || "${STATIC_LOCAL_PLANNER}" == "TRUE" ]]; then
  # Deprecated: this old slot mode changed leader_id and made other modules
  # inconsistent.  Use explicit STATIC_PLANNER_START_X/Y instead.
  if [[ -n "${STATIC_PLANNER_SPAWN_ID:-}" ]]; then
    echo "[ERROR] STATIC_PLANNER_SPAWN_ID is deprecated. Use STATIC_PLANNER_START_X/Y and STATIC_PLANNER_TARGET_X/Y/Z instead." >&2
    exit 2
  fi

  if [[ -n "${STATIC_PLANNER_START_X:-}" || -n "${STATIC_PLANNER_START_Y:-}" ]]; then
    if [[ -z "${STATIC_PLANNER_START_X:-}" || -z "${STATIC_PLANNER_START_Y:-}" ]]; then
      echo "[ERROR] STATIC_PLANNER_START_X and STATIC_PLANNER_START_Y must be set together." >&2
      exit 2
    fi
    NUM_DRONES=1
    LEADER_ID=1
    export STATIC_PLANNER_DRONE_ID=1
    SPAWN_OVERRIDE_NED_X="${STATIC_PLANNER_START_X}"
    SPAWN_OVERRIDE_NED_Y="${STATIC_PLANNER_START_Y}"
  fi

  export STATIC_PLANNER_TARGET_X="${STATIC_PLANNER_TARGET_X:-${DYNAMIC_OBS_TARGET_X}}"
  export STATIC_PLANNER_TARGET_Y="${STATIC_PLANNER_TARGET_Y:-${DYNAMIC_OBS_TARGET_Y_BASE}}"
  export STATIC_PLANNER_TARGET_Z="${STATIC_PLANNER_TARGET_Z:-${MISSION_Z}}"
else
  RUN_DURATION="${DURATION:-32}"
  WARMUP_SEC="${WARMUP_SEC:-25}"
  POST_OFFBOARD_HOLD_SEC="${POST_OFFBOARD_HOLD_SEC:-3}"
  MOTION_PRIMITIVE_ENABLE="${MOTION_PRIMITIVE_ENABLE:-1}"
  LIDAR_TTC_ENABLE="${LIDAR_TTC_ENABLE:-1}"
fi

# Keep the target marker and auxiliary target table consistent with the static
# planner target in single-UAV coordinate tests.
if [[ "${STATIC_LOCAL_PLANNER}" == "1" || "${STATIC_LOCAL_PLANNER}" == "true" || "${STATIC_LOCAL_PLANNER}" == "TRUE" ]]; then
  DYNAMIC_OBS_TARGET_X="${STATIC_PLANNER_TARGET_X}"
  DYNAMIC_OBS_TARGET_Y_BASE="${STATIC_PLANNER_TARGET_Y}"
  MISSION_Z="${STATIC_PLANNER_TARGET_Z}"
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
echo "[INFO] static_planner_start=(${SPAWN_OVERRIDE_NED_X:-auto},${SPAWN_OVERRIDE_NED_Y:-auto})"
echo "[INFO] static_planner_drone_id=${STATIC_PLANNER_DRONE_ID:-unset}"
echo "[INFO] static_planner_target=(${STATIC_PLANNER_TARGET_X:-auto},${STATIC_PLANNER_TARGET_Y:-auto},${STATIC_PLANNER_TARGET_Z:-auto})"
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

LAUNCH_ARGS=(
  num_drones:="${NUM_DRONES}"
  leader_id:="${LEADER_ID}"
  gz_world:=sando_forest3_walls_xtd2
  gz_gui:=0
  px4_sim_model:="${PX4_SIM_MODEL}"
  px4_use_versioned_local_position:=1
  dynamic_obs_enable:="${DYNAMIC_OBS_ENABLE}"
  dynamic_obs_mode:=scene
  scene_freeze_dynamics:="${DYNAMIC_OBS_FREEZE}"
  dynamic_obs_visualize_gz:=0
  spawned_formation_axis:="${SPAWNED_FORMATION_AXIS}"
  duration:="${RUN_DURATION}"
  warmup_sec:="${WARMUP_SEC}"
  post_offboard_hold_sec:="${POST_OFFBOARD_HOLD_SEC}"
  motion_primitive_enable:="${MOTION_PRIMITIVE_ENABLE}"
  lidar_ttc_enable:="${LIDAR_TTC_ENABLE}"
  scene_config:="${WS_ROOT}/scripts/scenes/sando/forest3_walls_dynamic.yaml"
  dynamic_obs_wall_x:=27.0
  start_wall_clearance:=8.0
  dynamic_obs_wall_y:=-14.0
  dynamic_obs_target_x:="${DYNAMIC_OBS_TARGET_X}"
  dynamic_obs_target_y_base:="${DYNAMIC_OBS_TARGET_Y_BASE}"
  dynamic_obs_target_y_spacing:="${DYNAMIC_OBS_TARGET_Y_SPACING}"
  mission_z:="${MISSION_Z}"
)

# ROS2 treats an empty launch argument like "spawn_override_ned_x:=" as
# malformed.  Only pass explicit spawn overrides when both coordinates are set.
# Otherwise the launch file uses its declared empty defaults and auto-spawns.
if [[ -n "${SPAWN_OVERRIDE_NED_X}" && -n "${SPAWN_OVERRIDE_NED_Y}" ]]; then
  LAUNCH_ARGS+=(
    spawn_override_ned_x:="${SPAWN_OVERRIDE_NED_X}"
    spawn_override_ned_y:="${SPAWN_OVERRIDE_NED_Y}"
  )
fi

exec ros2 launch xtd2_mission swarm_simulation_launch.py "${LAUNCH_ARGS[@]}"
