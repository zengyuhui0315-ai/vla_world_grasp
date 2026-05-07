#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${ROOT_DIR}/outputs/smoke_test"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" -m vla_world_grasp.pipeline.run_episode \
  --instruction "抓起红色方块" \
  --vla_backend mock \
  --sim_backend mock \
  --num_candidates 10 \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
