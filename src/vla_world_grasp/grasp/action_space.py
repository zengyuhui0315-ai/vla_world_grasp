"""Simple top-down grasp action space for the MVP."""

from __future__ import annotations

import math

from vla_world_grasp.vla.schemas import Vector3


DEFAULT_APPROACH_HEIGHT = 0.15
DEFAULT_GRASP_DEPTH = 0.035
TOP_DOWN_APPROACH: Vector3 = (0.0, 0.0, -1.0)
XY_OFFSET_SAMPLES: tuple[float, ...] = (-0.03, 0.0, 0.03)
YAW_SAMPLES_RAD: tuple[float, ...] = (
    0.0,
    math.radians(45.0),
    math.radians(90.0),
    math.radians(135.0),
)
XY_OFFSETS: tuple[tuple[float, float], ...] = tuple(
    (x, y) for x in XY_OFFSET_SAMPLES for y in XY_OFFSET_SAMPLES
)

GRIPPER_WIDTH_BY_TYPE: dict[str, float] = {
    "cube": 0.035,
    "cylinder": 0.025,
    "sphere": 0.025,
}


def estimate_gripper_width(
    object_size: Vector3 | None,
    object_type: str | None = None,
    min_width: float = 0.025,
    max_width: float = 0.085,
    margin: float = 0.015,
) -> float:
    """Estimate a reasonable gripper width from object size."""

    if object_type in GRIPPER_WIDTH_BY_TYPE:
        return GRIPPER_WIDTH_BY_TYPE[object_type]
    if object_size is None:
        return 0.06
    width = max(float(object_size[0]), float(object_size[1])) + margin
    return max(min_width, min(max_width, width))


def default_collision_risk(xy_offset: tuple[float, float], yaw: float) -> float:
    """Deterministic placeholder risk for non-Isaac scoring."""

    offset_mag = (xy_offset[0] ** 2 + xy_offset[1] ** 2) ** 0.5
    offset_risk = min(0.5, offset_mag / 0.05)
    yaw_risk = 0.05 if yaw in (0.0, math.radians(90.0)) else 0.1
    return min(1.0, offset_risk + yaw_risk)
