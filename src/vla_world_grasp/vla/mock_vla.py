"""Deterministic mock VLA backend for tests and fallback runs."""

from __future__ import annotations

from vla_world_grasp.vla.base import BaseVLABackend
from vla_world_grasp.vla.instruction_parser import parse_chinese_instruction
from vla_world_grasp.vla.schemas import (
    SceneObject,
    VLAObservation,
    VLAResult,
    VLATarget,
)


DEFAULT_OBJECT_STATES: tuple[SceneObject, ...] = (
    SceneObject(
        object_id="red_cube",
        label="red cube",
        color="red",
        shape="cube",
        position=(-0.12, 0.02, 0.035),
        size=(0.05, 0.05, 0.05),
        bbox_xyxy=(90, 120, 145, 175),
    ),
    SceneObject(
        object_id="blue_cylinder",
        label="blue cylinder",
        color="blue",
        shape="cylinder",
        position=(0.13, -0.04, 0.045),
        size=(0.045, 0.045, 0.09),
        bbox_xyxy=(220, 130, 265, 190),
    ),
    SceneObject(
        object_id="green_sphere",
        label="green sphere",
        color="green",
        shape="sphere",
        position=(0.01, 0.01, 0.04),
        size=(0.055, 0.055, 0.055),
        bbox_xyxy=(155, 115, 205, 165),
    ),
)

DEFAULT_OBJECTS = DEFAULT_OBJECT_STATES


class MockVLA(BaseVLABackend):
    """Rule-based target selector using Chinese instructions and object state."""

    name = "mock_vla"

    def infer(self, observation: VLAObservation) -> VLAResult:
        objects = observation.objects or DEFAULT_OBJECT_STATES
        parsed = parse_chinese_instruction(observation.text)
        target_object, selection_reason = self._select_object(parsed, objects, observation)
        target = VLATarget(
            object_id=target_object.object_id,
            label=target_object.label,
            point_3d=target_object.position,
            confidence=0.95,
            bbox_xyxy=target_object.bbox_xyxy,
            point_2d=_bbox_center(target_object.bbox_xyxy),
            metadata={
                "color": target_object.color,
                "shape": target_object.shape,
                "size": target_object.size,
                "source": "object_states",
                "instruction_parse": parsed,
                "selection_reason": selection_reason,
            },
        )
        return VLAResult(
            mode="target_localization",
            backend=self.name,
            status="ok",
            target=target,
            message="target localized by deterministic mock_vla",
            metadata={
                "instruction": observation.text,
                "instruction_parse": parsed,
                "selection": "deterministic_mock",
                "selection_reason": selection_reason,
            },
        )

    def _select_object(
        self,
        parsed_instruction: dict[str, object],
        objects: tuple[SceneObject, ...],
        observation: VLAObservation,
    ) -> tuple[SceneObject, str]:
        if not objects:
            raise ValueError("mock_vla requires at least one object state")

        color = _optional_str(parsed_instruction.get("target_color"))
        shape = _optional_str(parsed_instruction.get("target_type"))
        position_hint = _optional_str(parsed_instruction.get("target_position_hint"))

        if color and shape:
            target = _find_first(objects, color=color, shape=shape)
            if target is not None:
                return target, f"matched color+type: {color} {shape}"

        if color:
            target = _find_first(objects, color=color)
            if target is not None:
                return target, f"matched color: {color}"

        if shape:
            target = _find_first(objects, shape=shape)
            if target is not None:
                return target, f"matched type: {shape}"

        if position_hint:
            return _select_by_position_hint(objects, position_hint, observation), f"matched position_hint: {position_hint}"

        return _select_center(objects, observation), "fallback: nearest table center"


def _find_first(
    objects: tuple[SceneObject, ...],
    color: str | None = None,
    shape: str | None = None,
) -> SceneObject | None:
    for item in objects:
        if color is not None and item.color != color:
            continue
        if shape is not None and item.shape != shape:
            continue
        return item
    return None


def _select_by_position_hint(
    objects: tuple[SceneObject, ...],
    position_hint: str,
    observation: VLAObservation,
) -> SceneObject:
    if position_hint == "left":
        return min(objects, key=lambda item: item.position[0])
    if position_hint == "right":
        return max(objects, key=lambda item: item.position[0])
    if position_hint == "center":
        return _select_center(objects, observation)
    if position_hint == "nearest":
        return min(objects, key=lambda item: _xy_distance(item.position, (0.0, 0.0)))
    if position_hint == "farthest":
        return max(objects, key=lambda item: _xy_distance(item.position, (0.0, 0.0)))
    return _select_center(objects, observation)


def _select_center(objects: tuple[SceneObject, ...], observation: VLAObservation) -> SceneObject:
    center = _table_center_xy(observation)
    return min(objects, key=lambda item: _xy_distance(item.position, center))


def _table_center_xy(observation: VLAObservation) -> tuple[float, float]:
    table = observation.metadata.get("table", {})
    position = table.get("position") if isinstance(table, dict) else None
    if position is not None:
        return (float(position[0]), float(position[1]))
    return (0.0, 0.0)


def _xy_distance(position: tuple[float, float, float], xy: tuple[float, float]) -> float:
    return ((position[0] - xy[0]) ** 2 + (position[1] - xy[1]) ** 2) ** 0.5


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None


def _bbox_center(
    bbox_xyxy: tuple[int, int, int, int] | None,
) -> tuple[float, float] | None:
    if bbox_xyxy is None:
        return None
    x_min, y_min, x_max, y_max = bbox_xyxy
    return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)
