#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import json

from vla_world_grasp.vla.mock_vla import MockVLA
from vla_world_grasp.vla.schemas import SceneObject, VLAObservation

objects = (
    SceneObject("red_cube", "red cube", "red", "cube", (0.45, -0.12, 0.045)),
    SceneObject("blue_cylinder", "blue cylinder", "blue", "cylinder", (0.52, 0.03, 0.07)),
    SceneObject("green_sphere", "green sphere", "green", "sphere", (0.40, 0.11, 0.05)),
)

table_metadata = {"table": {"position": (0.5, 0.0, 0.0)}}
cases = {
    "抓起红色方块": "red_cube",
    "请抓起蓝色圆柱": "blue_cylinder",
    "把绿色球拿起来": "green_sphere",
    "抓起最左边的物体": "green_sphere",
    "抓起中间的物体": "blue_cylinder",
    "夹起离机器人最近的物体": "green_sphere",
    "抓起蓝色物体": "blue_cylinder",
    "抓起圆柱": "blue_cylinder",
}

backend = MockVLA()
for instruction, expected in cases.items():
    result = backend.infer(
        VLAObservation(
            instruction=instruction,
            object_states=objects,
            metadata=table_metadata,
        )
    )
    assert result.target is not None
    assert result.target.object_id == expected, (instruction, result.target.object_id, expected)
    assert result.metadata["instruction_parse"]["raw_instruction"] == instruction
    assert result.metadata["selection_reason"]
    json.dumps(result.metadata, ensure_ascii=False)

print("Chinese mock_vla tests passed")
PY
