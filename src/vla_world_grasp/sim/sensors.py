"""Sensor and observation helpers for the grasp demo."""

from __future__ import annotations

from typing import Any

from vla_world_grasp.sim.scene import GraspScene
from vla_world_grasp.vla.schemas import VLAObservation


def capture_observation(
    scene: GraspScene,
    instruction: str,
    image: Any | None = None,
    depth: Any | None = None,
    rgb_path: str | None = None,
    depth_path: str | None = None,
    camera_intrinsics: dict[str, Any] | None = None,
    camera_pose: dict[str, Any] | None = None,
) -> VLAObservation:
    """Build the object-state observation consumed by VLA backends."""

    return VLAObservation(
        instruction=instruction,
        rgb_path=rgb_path,
        depth_path=depth_path,
        camera_intrinsics=camera_intrinsics or {},
        camera_pose=camera_pose or {},
        object_states=scene.get_object_states(),
        image=image,
        depth=depth,
        camera_info={
            "frame": "world",
            "mode": "object_states",
            "headless": scene.headless,
            "intrinsics": camera_intrinsics or {},
            "pose": camera_pose or {},
        },
        metadata={
            "sim_backend": scene.backend,
            "table": scene.table_state,
        },
    )
