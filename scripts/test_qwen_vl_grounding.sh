#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-${QWEN_VL_MODEL_PATH:-/home/ubuntu/models/Qwen2.5-VL-3B-Instruct}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --qwen_model_path)
      QWEN_MODEL_PATH="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done

cd "${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${QWEN_MODEL_PATH}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from vla_world_grasp.vla.manager import VLABackendManager
from vla_world_grasp.vla.qwen_vl_adapter import QwenVLAdapter
from vla_world_grasp.vla.schemas import SceneObject, VLAObservation

model_path = sys.argv[1]
print("[vla_world_grasp] qwen_vl_adapter import: ok")
print(f"[vla_world_grasp] QwenVLAdapter class: {QwenVLAdapter.__name__}")

if not model_path or not Path(model_path).exists():
    print("[vla_world_grasp] Qwen-VL model: not_available")
else:
    print(f"[vla_world_grasp] Qwen-VL model: available at {model_path}")

image_path = Path("outputs/debug/qwen_vl_blue_cylinder_demo_qwen_input.png")
if model_path and Path(model_path).exists() and image_path.exists():
    adapter = QwenVLAdapter(model_path=model_path, device="cuda")
    adapter._run_model = lambda observation, model_path: '{"target_object":"red cube","color":"red","shape":"cube","grasp_hint":"top-down grasp at object center"}'
    simplified_observation = VLAObservation(
        instruction="抓起红色方块",
        rgb_path=str(image_path),
        object_states=(
            SceneObject(
                object_id="red_cube",
                label="red cube",
                color="red",
                shape="cube",
                position=(0.45, -0.12, 0.045),
                size=(0.05, 0.05, 0.05),
            ),
        ),
        metadata={"qwen_model_path": model_path},
    )
    qwen_result = adapter.infer(simplified_observation)
    print(f"[vla_world_grasp] simplified_json backend: {qwen_result.backend}")
    print(f"[vla_world_grasp] simplified_json matched_target_name: {qwen_result.metadata.get('matched_target_name')}")
    print(f"[vla_world_grasp] simplified_json fallback_used: {bool(qwen_result.metadata.get('fallback_used', False))}")
    if qwen_result.backend != "qwen_vl" or qwen_result.metadata.get("matched_target_name") != "red_cube":
        raise SystemExit("simplified JSON qwen_vl parser test failed")

objects = (
    SceneObject(
        object_id="blue_cylinder",
        label="blue cylinder",
        color="blue",
        shape="cylinder",
        position=(0.52, 0.03, 0.07),
        size=(0.08, 0.08, 0.10),
    ),
)
observation = VLAObservation(
    instruction="抓起蓝色圆柱",
    object_states=objects,
    rgb_path="",
    metadata={"qwen_model_path": model_path},
)
manager = VLABackendManager.from_name(
    "qwen_vl",
    qwen_model_path=model_path or None,
    qwen_fallback_to_mock=True,
)
result = manager.infer(observation)
print(f"[vla_world_grasp] fallback mock_vla status: {result.status}")
print(f"[vla_world_grasp] fallback backend: {result.backend}")
print(f"[vla_world_grasp] fallback_used: {bool(result.metadata.get('fallback_used', False))}")
if result.status != "ok" or result.backend != "mock_vla" or not result.metadata.get("fallback_used", False):
    raise SystemExit("fallback mock_vla test failed")
PY
