#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
from vla_world_grasp.vla.manager import VLABackendManager
from vla_world_grasp.vla.mock_vla import MockVLA
from vla_world_grasp.vla.parsing import robust_json_parse
from vla_world_grasp.vla.schemas import SceneObject, VLAObservation, VLAResult

objects = (
    SceneObject("red_cube", "red cube", "red", "cube", (-0.12, 0.02, 0.035)),
    SceneObject("blue_cylinder", "blue cylinder", "blue", "cylinder", (0.13, -0.04, 0.045)),
    SceneObject("green_sphere", "green sphere", "green", "sphere", (0.01, 0.01, 0.04)),
)
cases = {
    "抓取红色方块": "red_cube",
    "抓取蓝色圆柱": "blue_cylinder",
    "抓取绿色球": "green_sphere",
    "抓取左边的物体": "red_cube",
    "抓取中间的物体": "green_sphere",
}

backend = MockVLA()
for prompt, expected_id in cases.items():
    result = backend.infer(VLAObservation(instruction=prompt, object_states=objects))
    assert isinstance(result, VLAResult)
    assert result.mode == "target_localization"
    assert result.status == "ok"
    assert result.target is not None
    assert result.target.object_id == expected_id, (prompt, result.target.object_id, expected_id)
    assert result.target.point_3d == next(item.position for item in objects if item.object_id == expected_id)
    assert result.backend == "mock_vla"
    assert 0.0 <= result.confidence <= 1.0

qwen_fallback = VLABackendManager("qwen_vl").infer(
    VLAObservation(instruction="抓取红色方块", object_states=objects)
)
assert isinstance(qwen_fallback, VLAResult)
assert qwen_fallback.backend == "mock_vla"
assert qwen_fallback.status == "ok"
assert qwen_fallback.target_id == "red_cube"
assert qwen_fallback.metadata["requested_backend"] == "qwen_vl"

smol_fallback = VLABackendManager("smolvla").infer(
    VLAObservation(instruction="抓取蓝色圆柱", object_states=objects)
)
assert isinstance(smol_fallback, VLAResult)
assert smol_fallback.backend == "mock_vla"
assert smol_fallback.status == "ok"
assert smol_fallback.target_id == "blue_cylinder"
assert smol_fallback.metadata["requested_backend"] == "smolvla"

assert robust_json_parse('{"a": 1}') == {"a": 1}
assert robust_json_parse('```json\n{"b": 2}\n```') == {"b": 2}

print("VLA interface tests passed")
PY
