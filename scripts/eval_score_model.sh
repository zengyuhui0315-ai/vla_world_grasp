#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

cd "${ROOT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/outputs/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" \
  "${PYTHON_BIN}" -m vla_world_grasp.world_model.eval "$@"

