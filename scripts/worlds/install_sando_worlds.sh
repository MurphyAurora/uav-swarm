#!/usr/bin/env bash
# Install local SANDO-adapted SDF worlds and PX4 model overrides into PX4 Gazebo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC_DIR="${SCRIPT_DIR}/sando_adapted"
MODEL_SRC_DIR="${SCRIPT_DIR}/px4_models"
PX4_WORLDS_DIR="${PX4_WORLDS_DIR:-${WS_ROOT}/firmware/PX4-Autopilot/Tools/simulation/gz/worlds}"
PX4_MODELS_DIR="${PX4_MODELS_DIR:-${WS_ROOT}/firmware/PX4-Autopilot/Tools/simulation/gz/models}"

if [ ! -d "${SRC_DIR}" ]; then
  echo "[ERROR] source world dir not found: ${SRC_DIR}" >&2
  exit 1
fi

if [ ! -d "${PX4_WORLDS_DIR}" ]; then
  echo "[ERROR] PX4 worlds dir not found: ${PX4_WORLDS_DIR}" >&2
  echo "        Set PX4_WORLDS_DIR=/path/to/PX4-Autopilot/Tools/simulation/gz/worlds" >&2
  exit 1
fi

if [ ! -d "${PX4_MODELS_DIR}" ]; then
  echo "[ERROR] PX4 models dir not found: ${PX4_MODELS_DIR}" >&2
  echo "        Set PX4_MODELS_DIR=/path/to/PX4-Autopilot/Tools/simulation/gz/models" >&2
  exit 1
fi

for world in "${SRC_DIR}"/*.sdf; do
  [ -e "${world}" ] || continue
  cp "${world}" "${PX4_WORLDS_DIR}/$(basename "${world}")"
  echo "[OK] installed $(basename "${world}")"
done

echo "[DONE] PX4 worlds dir: ${PX4_WORLDS_DIR}"

if [ -d "${MODEL_SRC_DIR}" ]; then
  for model_dir in "${MODEL_SRC_DIR}"/*; do
    [ -d "${model_dir}" ] || continue
    model_name="$(basename "${model_dir}")"
    mkdir -p "${PX4_MODELS_DIR}/${model_name}"
    cp "${model_dir}"/* "${PX4_MODELS_DIR}/${model_name}/"
    echo "[OK] installed model ${model_name}"
  done
fi

echo "[DONE] PX4 models dir: ${PX4_MODELS_DIR}"
