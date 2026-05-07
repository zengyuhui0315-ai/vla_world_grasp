"""Placeholder OpenVLA adapter."""

from __future__ import annotations

from vla_world_grasp.vla.base import BaseVLABackend
from vla_world_grasp.vla.schemas import VLAObservation, VLAResult


class OpenVLAAdapter(BaseVLABackend):
    name = "openvla"

    def infer(self, observation: VLAObservation) -> VLAResult:
        return VLAResult(
            mode="target_localization",
            backend=self.name,
            status="not_enabled",
            message="openvla adapter is not enabled",
            metadata={"instruction": observation.text},
        )
