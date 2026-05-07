#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SIM_BACKEND="isaac"
ARGS=("$@")
PASSTHROUGH_ARGS=()
CANDIDATE_VIS_WARNING_SHOWN=0

# Parse --sim_backend from args, default to isaac.
for ((i=0; i<${#ARGS[@]}; i++)); do
  case "${ARGS[$i]}" in
    --visualize_candidates|--target_highlight)
      if [[ "${CANDIDATE_VIS_WARNING_SHOWN}" == "0" ]]; then
        echo "[vla_world_grasp] Candidate visualization is disabled in final demo mode."
        CANDIDATE_VIS_WARNING_SHOWN=1
      fi
      if [[ "${ARGS[$i]}" == "--visualize_candidates" ]]; then
        PASSTHROUGH_ARGS+=("--no_visualize_candidates")
      fi
      ;;
    --sim_backend)
      if [[ $((i + 1)) -lt ${#ARGS[@]} ]]; then
        SIM_BACKEND="${ARGS[$((i + 1))]}"
        PASSTHROUGH_ARGS+=("${ARGS[$i]}" "${ARGS[$((i + 1))]}")
        i=$((i + 1))
      fi
      ;;
    --frame_interval)
      if [[ $((i + 1)) -lt ${#ARGS[@]} ]]; then
        export VLA_WORLD_GRASP_FRAME_INTERVAL="${ARGS[$((i + 1))]}"
        PASSTHROUGH_ARGS+=("${ARGS[$i]}" "${ARGS[$((i + 1))]}")
        i=$((i + 1))
      fi
      ;;
    --max_steps)
      if [[ $((i + 1)) -lt ${#ARGS[@]} ]]; then
        export VLA_WORLD_GRASP_MAX_STEPS="${ARGS[$((i + 1))]}"
        PASSTHROUGH_ARGS+=("${ARGS[$i]}" "${ARGS[$((i + 1))]}")
        i=$((i + 1))
      fi
      ;;
    *)
      PASSTHROUGH_ARGS+=("${ARGS[$i]}")
      ;;
  esac
done

if [[ "$SIM_BACKEND" == "mock" ]]; then
  echo "[INFO] Running mock simulation backend."
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
  PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m vla_world_grasp.pipeline.run_episode "${PASSTHROUGH_ARGS[@]}"
  exit 0
fi

if [[ "$SIM_BACKEND" != "isaac" ]]; then
  echo "ERROR: Unsupported --sim_backend '$SIM_BACKEND'. Use 'isaac' or 'mock'."
  exit 1
fi

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
  echo
  echo "Set ISAACLAB_DIR to your IsaacLab path, for example:"
  echo "  export ISAACLAB_DIR=\$HOME/work/external/IsaacLab"
  echo
  echo "Or clone IsaacLab into:"
  echo "  ${ROOT_DIR}/external/IsaacLab"
  exit 1
fi

echo "[INFO] Running Isaac Sim / IsaacLab backend."
echo "[INFO] ISAACLAB_DIR=${ISAACLAB_DIR}"
export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
export LIVESTREAM="${LIVESTREAM:-0}"
export TERM="${TERM:-xterm-256color}"
if [[ "${TERM}" == "dumb" ]]; then
  export TERM="xterm-256color"
fi
echo "[INFO] ENABLE_CAMERAS=${ENABLE_CAMERAS}"
if [[ -n "${VLA_WORLD_GRASP_FRAME_INTERVAL:-}" ]]; then
  echo "[INFO] VLA_WORLD_GRASP_FRAME_INTERVAL=${VLA_WORLD_GRASP_FRAME_INTERVAL}"
fi
if [[ -n "${VLA_WORLD_GRASP_MAX_STEPS:-}" ]]; then
  echo "[INFO] VLA_WORLD_GRASP_MAX_STEPS=${VLA_WORLD_GRASP_MAX_STEPS}"
fi

cd "${ROOT_DIR}"

# Use IsaacLab's launcher so isaaclab and Isaac Sim extensions are importable.
PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
  "${ISAACLAB_DIR}/isaaclab.sh" -p "${ROOT_DIR}/src/vla_world_grasp/pipeline/run_episode.py" "${PASSTHROUGH_ARGS[@]}"
