"""Small geometry helpers for non-Isaac grasp planning tests."""

from __future__ import annotations

import math

from vla_world_grasp.grasp.action_space import TOP_DOWN_APPROACH
from vla_world_grasp.vla.schemas import Vector3


def add_xyz(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def xy_distance(a: Vector3, b: Vector3) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def distance_score(distance: float, max_distance: float = 0.08) -> float:
    return clamp01(1.0 - float(distance) / max_distance)


def table_height(object_states: list[dict] | tuple[dict, ...] | None = None) -> float:
    return 0.02


def height_score(z: float, table_top_z: float = 0.02, max_above_table: float = 0.18) -> float:
    if z < table_top_z:
        return 0.0
    return clamp01(1.0 - max(0.0, z - (table_top_z + max_above_table)) / max_above_table)


def is_in_workspace(
    position: Vector3,
    x_limits: tuple[float, float] = (0.20, 0.75),
    y_limits: tuple[float, float] = (-0.35, 0.35),
    z_limits: tuple[float, float] = (0.0, 0.75),
) -> bool:
    return (
        x_limits[0] <= position[0] <= x_limits[1]
        and y_limits[0] <= position[1] <= y_limits[1]
        and z_limits[0] <= position[2] <= z_limits[1]
    )


def reachability_score(
    position: Vector3,
    x_limits: tuple[float, float] = (0.20, 0.75),
    y_limits: tuple[float, float] = (-0.35, 0.35),
    z_limits: tuple[float, float] = (0.0, 0.75),
) -> float:
    if not is_in_workspace(position, x_limits=x_limits, y_limits=y_limits, z_limits=z_limits):
        return 0.0
    center = (
        (x_limits[0] + x_limits[1]) / 2.0,
        (y_limits[0] + y_limits[1]) / 2.0,
        (z_limits[0] + z_limits[1]) / 2.0,
    )
    normalized = (
        abs(position[0] - center[0]) / ((x_limits[1] - x_limits[0]) / 2.0)
        + abs(position[1] - center[1]) / ((y_limits[1] - y_limits[0]) / 2.0)
        + abs(position[2] - center[2]) / ((z_limits[1] - z_limits[0]) / 2.0)
    ) / 3.0
    return clamp01(1.0 - 0.35 * normalized)


def yaw_degrees(yaw_rad: float) -> float:
    return math.degrees(yaw_rad)


def ring_offsets(radius: float, count: int) -> tuple[Vector3, ...]:
    """Compatibility helper retained for earlier callers."""

    return tuple(
        (
            radius * math.cos(2.0 * math.pi * idx / count),
            radius * math.sin(2.0 * math.pi * idx / count),
            0.0,
        )
        for idx in range(count)
    )


def yaw_from_offset(offset: Vector3) -> float:
    """Compatibility helper retained for earlier callers."""

    if offset[0] == 0.0 and offset[1] == 0.0:
        return 0.0
    return math.atan2(offset[1], offset[0])
