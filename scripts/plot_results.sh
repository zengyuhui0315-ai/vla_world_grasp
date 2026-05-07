#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" -m vla_world_grasp.pipeline.visualize "$@"
