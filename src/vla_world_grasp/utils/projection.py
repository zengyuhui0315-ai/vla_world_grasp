"""Small camera projection helpers for VLA grounding."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np


DEFAULT_CAMERA_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 672.0,
    "fy": 672.0,
    "cx": 320.0,
    "cy": 240.0,
}
DEFAULT_CAMERA_POSE = {
    "eye": (1.25, -1.15, 0.82),
    "target": (0.48, 0.0, 0.08),
    "up": (0.0, 0.0, 1.0),
}


def default_camera_intrinsics() -> dict[str, float | int]:
    return dict(DEFAULT_CAMERA_INTRINSICS)


def default_camera_pose() -> dict[str, tuple[float, float, float]]:
    return dict(DEFAULT_CAMERA_POSE)


def bbox_center(bbox_2d: Iterable[float] | None) -> tuple[float, float] | None:
    if bbox_2d is None:
        return None
    values = [float(value) for value in bbox_2d]
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_contains_point(bbox_2d: Iterable[float] | None, point_2d: Iterable[float] | None) -> bool | None:
    if bbox_2d is None or point_2d is None:
        return None
    x1, y1, x2, y2 = [float(value) for value in bbox_2d]
    u, v = [float(value) for value in point_2d]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return left <= u <= right and top <= v <= bottom


def project_point(
    point_3d: Iterable[float],
    intrinsics: dict[str, Any] | None = None,
    camera_pose: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    k = _intrinsics(intrinsics)
    rotation, eye = _world_to_camera(camera_pose)
    point = np.asarray([float(value) for value in point_3d], dtype=np.float64)
    cam = rotation @ (point - eye)
    if cam[2] <= 1e-6:
        return None
    u = k["fx"] * (cam[0] / cam[2]) + k["cx"]
    v = k["cy"] - k["fy"] * (cam[1] / cam[2])
    return (float(u), float(v))


def backproject_pixel(
    point_2d: Iterable[float],
    depth: float,
    intrinsics: dict[str, Any] | None = None,
    camera_pose: dict[str, Any] | None = None,
) -> tuple[float, float, float] | None:
    if not math.isfinite(float(depth)) or float(depth) <= 0.0:
        return None
    k = _intrinsics(intrinsics)
    u, v = [float(value) for value in point_2d]
    x = (u - k["cx"]) / k["fx"] * float(depth)
    y = (k["cy"] - v) / k["fy"] * float(depth)
    z = float(depth)
    rotation, eye = _world_to_camera(camera_pose)
    world = rotation.T @ np.asarray([x, y, z], dtype=np.float64) + eye
    return (float(world[0]), float(world[1]), float(world[2]))


def nearest_object_by_projected_center(
    point_2d: Iterable[float],
    object_states: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    intrinsics: dict[str, Any] | None = None,
    camera_pose: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, float | None]:
    point = np.asarray([float(value) for value in point_2d], dtype=np.float64)
    best_obj = None
    best_dist = None
    for obj in object_states:
        position = obj.get("position") or obj.get("point_3d")
        if position is None:
            continue
        projected = project_point(position, intrinsics=intrinsics, camera_pose=camera_pose)
        if projected is None:
            continue
        dist = float(np.linalg.norm(np.asarray(projected, dtype=np.float64) - point))
        if best_dist is None or dist < best_dist:
            best_obj = obj
            best_dist = dist
    return best_obj, best_dist


def nearest_object_by_3d_point(
    point_3d: Iterable[float],
    object_states: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any] | None, float | None]:
    point = np.asarray([float(value) for value in point_3d], dtype=np.float64)
    best_obj = None
    best_dist = None
    for obj in object_states:
        position = obj.get("position") or obj.get("point_3d")
        if position is None:
            continue
        dist = float(np.linalg.norm(np.asarray(position, dtype=np.float64) - point))
        if best_dist is None or dist < best_dist:
            best_obj = obj
            best_dist = dist
    return best_obj, best_dist


def depth_at_pixel(depth_image: Any, point_2d: Iterable[float]) -> float | None:
    if depth_image is None:
        return None
    arr = np.asarray(depth_image)
    if arr.ndim < 2:
        return None
    u, v = [int(round(float(value))) for value in point_2d]
    if v < 0 or v >= arr.shape[0] or u < 0 or u >= arr.shape[1]:
        return None
    value = float(arr[v, u])
    return value if math.isfinite(value) and value > 0.0 else None


def _intrinsics(intrinsics: dict[str, Any] | None) -> dict[str, float]:
    merged = dict(DEFAULT_CAMERA_INTRINSICS)
    if intrinsics:
        merged.update({key: value for key, value in intrinsics.items() if value is not None})
    return {key: float(merged[key]) for key in ("fx", "fy", "cx", "cy", "width", "height")}


def _world_to_camera(camera_pose: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    pose = dict(DEFAULT_CAMERA_POSE)
    if camera_pose:
        pose.update({key: value for key, value in camera_pose.items() if value is not None})
    eye = np.asarray(pose["eye"], dtype=np.float64)
    target = np.asarray(pose["target"], dtype=np.float64)
    up = np.asarray(pose.get("up", (0.0, 0.0, 1.0)), dtype=np.float64)
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-8)
    true_up = np.cross(right, forward)
    rotation = np.stack([right, true_up, forward], axis=0)
    return rotation, eye
