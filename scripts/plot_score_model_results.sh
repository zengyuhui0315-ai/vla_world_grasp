#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

cd "${ROOT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/outputs/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

DATASET_CSV="data/grasp_success/processed/grasp_success_dataset.csv"
CHECKPOINT="outputs/checkpoints/score_model_best.pt"
ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_csv)
      DATASET_CSV="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m vla_world_grasp.world_model.eval \
  --dataset_csv "${DATASET_CSV}" \
  --checkpoint "${CHECKPOINT}" \
  "${ARGS[@]}"

