#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
export ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
export LIVESTREAM="${LIVESTREAM:-0}"

HEADLESS_ARGS=()
VLA_BACKEND="mock"
SIM_BACKEND="isaac"
NUM_CANDIDATES="16"
SEED="0"
OUTPUT_DIR="${ROOT_DIR}/outputs/recordings"
VIDEOS_DIR="${ROOT_DIR}/outputs/videos"
FRAME_INTERVAL=""
MAX_STEPS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless)
      HEADLESS_ARGS=(--headless)
      shift
      ;;
    --no-headless)
      HEADLESS_ARGS=(--no-headless)
      shift
      ;;
    --vla_backend)
      VLA_BACKEND="$2"
      shift 2
      ;;
    --sim_backend)
      SIM_BACKEND="$2"
      shift 2
      ;;
    --num_candidates)
      NUM_CANDIDATES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --videos_dir)
      VIDEOS_DIR="$2"
      shift 2
      ;;
    --frame_interval)
      FRAME_INTERVAL="$2"
      shift 2
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'USAGE'
Usage: bash scripts/record_demo.sh [options]

Runs three Isaac demo episodes and saves result.json plus Isaac camera videos.

Options:
  --headless              Run in headless mode.
  --no-headless           Request non-headless mode.
  --vla_backend NAME      VLA backend name, default: mock.
  --sim_backend NAME      Simulation backend, default: isaac. Use mock only for debug.
  --num_candidates N      Number of grasp candidates, clamped by pipeline.
  --seed N                Base seed, default: 0.
  --output_dir DIR        Output root, default: outputs/recordings.
  --videos_dir DIR        MP4 output root, default: outputs/videos.
  --frame_interval N      Capture one frame every N simulation steps.
  --max_steps N           Accepted for run_demo compatibility.
  -h, --help              Show this help.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

INSTRUCTIONS=(
  "抓起红色方块"
  "抓起蓝色圆柱"
  "抓起中间的物体"
)

SLUGS=(
  "red_cube"
  "blue_cylinder"
  "middle_object"
)

for index in "${!INSTRUCTIONS[@]}"; do
  episode_dir="${OUTPUT_DIR}/$((index + 1))_${SLUGS[$index]}"
  echo "recording ${INSTRUCTIONS[$index]} -> ${episode_dir}"
  EXTRA_ARGS=()
  if [[ -n "${FRAME_INTERVAL}" ]]; then
    EXTRA_ARGS+=(--frame_interval "${FRAME_INTERVAL}")
  fi
  if [[ -n "${MAX_STEPS}" ]]; then
    EXTRA_ARGS+=(--max_steps "${MAX_STEPS}")
  fi
  bash "${ROOT_DIR}/scripts/run_demo.sh" \
    --instruction "${INSTRUCTIONS[$index]}" \
    --vla_backend "${VLA_BACKEND}" \
    --sim_backend "${SIM_BACKEND}" \
    --num_candidates "${NUM_CANDIDATES}" \
    --seed "$((SEED + index))" \
    --output_dir "${episode_dir}" \
    --videos_dir "${VIDEOS_DIR}" \
    --video_name "${SLUGS[$index]}_isaac_demo" \
    --save_video \
    "${EXTRA_ARGS[@]}" \
    "${HEADLESS_ARGS[@]}"
done

echo "recordings_dir=${OUTPUT_DIR}"
