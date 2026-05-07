#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
from vla_world_grasp.vla.instruction_parser import parse_chinese_instruction

cases = [
    ("抓起红色方块", "red", "cube", None, "红色方块", "red cube"),
    ("请抓起蓝色圆柱", "blue", "cylinder", None, "蓝色圆柱", "blue cylinder"),
    ("把绿色球拿起来", "green", "sphere", None, "绿色球", "green sphere"),
    ("抓起最左边的物体", None, None, "left", "左边的物体", "left"),
    ("抓起中间的物体", None, None, "center", "中间的物体", "center"),
    ("夹起离机器人最近的物体", None, None, "nearest", "离机器人最近的物体", "nearest"),
    ("pick blue cylinder", "blue", "cylinder", None, "蓝色圆柱", "blue cylinder"),
    ("grasp the right object", None, None, "right", "右边的物体", "right"),
]

for instruction, color, shape, position, label_zh, label_en in cases:
    parsed = parse_chinese_instruction(instruction)
    assert parsed["raw_instruction"] == instruction
    assert parsed["language"] == "zh"
    assert parsed["intent"] == "pick"
    assert parsed["target_color"] == color, (instruction, parsed)
    assert parsed["target_type"] == shape, (instruction, parsed)
    assert parsed["target_position_hint"] == position, (instruction, parsed)
    assert parsed["normalized_target_label_zh"] == label_zh, (instruction, parsed)
    assert parsed["normalized_target_label_en"] == label_en, (instruction, parsed)
    assert 0.0 <= parsed["confidence"] <= 1.0

print("Chinese instruction parser tests passed")
PY
