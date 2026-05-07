#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${ISAACLAB_DIR:-}" ]]; then
  for candidate in \
    "${ROOT_DIR}/external/IsaacLab" \
    "${ROOT_DIR}/../external/IsaacLab" \
    "${HOME}/IsaacLab"; do
    if [[ -f "${candidate}/isaaclab.sh" ]]; then
      ISAACLAB_DIR="${candidate}"
      break
    fi
  done
fi
ISAACLAB_DIR="${ISAACLAB_DIR:-${ROOT_DIR}/external/IsaacLab}"

if [[ ! -f "${ISAACLAB_DIR}/isaaclab.sh" ]]; then
  echo "ERROR: IsaacLab not found at: ${ISAACLAB_DIR}"
  echo "Set ISAACLAB_DIR to your IsaacLab path."
  exit 1
fi

export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
export LIVESTREAM="${LIVESTREAM:-0}"
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM="xterm-256color"
fi
export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

cd "${ROOT_DIR}"

echo "[INFO] Collecting Isaac grasp dataset."
echo "[INFO] ISAACLAB_DIR=${ISAACLAB_DIR}"
echo "[INFO] ENABLE_CAMERAS=${ENABLE_CAMERAS}"
echo "[INFO] Per-episode timeout is controlled by --episode_timeout_sec (default: 180)."
echo "[INFO] Per-episode video recording is disabled unless --save_video_per_episode true is passed."

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
  "${ISAACLAB_DIR}/isaaclab.sh" -p "${ROOT_DIR}/src/vla_world_grasp/pipeline/collect_dataset.py" "$@"
