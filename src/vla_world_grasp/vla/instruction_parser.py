"""Chinese instruction parser for tabletop grasp commands."""

from __future__ import annotations

from typing import Any


COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("红色", "红", "red"),
    "blue": ("蓝色", "蓝", "blue"),
    "green": ("绿色", "绿", "green"),
    "yellow": ("黄色", "黄", "yellow"),
    "white": ("白色", "白", "white"),
    "black": ("黑色", "黑", "black"),
}

COLOR_ZH: dict[str, str] = {
    "red": "红色",
    "blue": "蓝色",
    "green": "绿色",
    "yellow": "黄色",
    "white": "白色",
    "black": "黑色",
}

TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "cube": ("方块", "立方体", "cube", "box"),
    "cylinder": ("圆柱体", "圆柱", "cylinder"),
    "sphere": ("球体", "球", "sphere"),
}

TYPE_ZH: dict[str, str] = {
    "cube": "方块",
    "cylinder": "圆柱",
    "sphere": "球",
}

POSITION_ALIASES: dict[str, tuple[str, ...]] = {
    "left": ("最左边", "左边", "left"),
    "right": ("最右边", "右边", "right"),
    "center": ("中间", "中央", "middle", "center"),
    "nearest": ("离机器人最近", "最近", "nearest"),
    "farthest": ("最远", "farthest", "furthest"),
}

ACTION_ALIASES: tuple[str, ...] = (
    "抓起",
    "抓取",
    "拿起",
    "夹起",
    "pick",
    "grasp",
    "lift",
)


def parse_chinese_instruction(instruction: str) -> dict[str, Any]:
    """Parse a Chinese or mixed Chinese-English tabletop pick instruction."""

    raw = instruction or ""
    text = raw.strip().lower()
    target_color = _find_alias(text, COLOR_ALIASES)
    target_type = _find_alias(text, TYPE_ALIASES)
    position_hint = _find_alias(text, POSITION_ALIASES)
    has_action = any(alias in text for alias in ACTION_ALIASES)

    normalized_zh = _normalized_label(target_color, target_type, position_hint, zh=True)
    normalized_en = _normalized_label(target_color, target_type, position_hint, zh=False)
    confidence = _confidence(has_action, target_color, target_type, position_hint)

    return {
        "raw_instruction": raw,
        "language": "zh",
        "intent": "pick" if has_action else "pick",
        "target_color": target_color,
        "target_type": target_type,
        "target_position_hint": position_hint,
        "normalized_target_label_zh": normalized_zh,
        "normalized_target_label_en": normalized_en,
        "confidence": confidence,
    }


def _find_alias(text: str, aliases: dict[str, tuple[str, ...]]) -> str | None:
    for value, options in aliases.items():
        if any(option in text for option in options):
            return value
    return None


def _normalized_label(color: str | None, shape: str | None, position: str | None, zh: bool) -> str:
    if zh:
        parts = []
        if color:
            parts.append(COLOR_ZH[color])
        if shape:
            parts.append(TYPE_ZH[shape])
        if parts:
            return "".join(parts)
        return _position_zh(position) if position else "目标物体"

    parts = [part for part in (color, shape) if part]
    if parts:
        return " ".join(parts)
    return position or "target object"


def _position_zh(position: str | None) -> str:
    return {
        "left": "左边的物体",
        "right": "右边的物体",
        "center": "中间的物体",
        "nearest": "离机器人最近的物体",
        "farthest": "最远的物体",
    }.get(position or "", "目标物体")


def _confidence(
    has_action: bool,
    color: str | None,
    shape: str | None,
    position: str | None,
) -> float:
    score = 0.45 if has_action else 0.25
    if color:
        score += 0.20
    if shape:
        score += 0.20
    if position:
        score += 0.15
    return min(1.0, score)
