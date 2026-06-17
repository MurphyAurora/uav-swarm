#!/usr/bin/env bash
set -euo pipefail

# One-command launcher for the local SANDO forest3-with-walls experiment.
#
# Usage:
#   ./scripts/run_sando_forest3.sh          # normal x500
#   ./scripts/run_sando_forest3.sh lidar3d  # x500 with 3D GPU LiDAR

MODE="${1:-normal}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

case "${MODE}" in
  normal)
    PX4_SIM_MODEL="gz_x500"
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

set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u

echo "[INFO] mode=${MODE}, px4_sim_model=${PX4_SIM_MODEL}"
if [ "${MODE}" = "lidar3d" ]; then
  echo "[INFO] After drones are spawned, open another terminal and run:"
  echo "       cd ${WS_ROOT} && ./scripts/bridge_x500_lidar_3d.sh 5"
fi

exec ros2 launch xtd2_mission swarm_simulation_launch.py \
  gz_world:=sando_forest3_walls_xtd2 \
  gz_gui:=0 \
  px4_sim_model:="${PX4_SIM_MODEL}" \
  px4_use_versioned_local_position:=1 \
  dynamic_obs_enable:=1 \
  dynamic_obs_mode:=scene \
  dynamic_obs_visualize_gz:=0 \
  spawned_formation_axis:=x \
  scene_config:="${WS_ROOT}/scripts/scenes/sando/forest3_walls_dynamic.yaml" \
  dynamic_obs_wall_x:=27.0 \
  start_wall_clearance:=8.0 \
  dynamic_obs_wall_y:=-14.0 \
  dynamic_obs_target_x:=19.0 \
  dynamic_obs_target_y_base:=30.0
