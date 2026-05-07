"""Base interface for VLA backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from vla_world_grasp.vla.schemas import VLAObservation, VLAResult


class BaseVLABackend(ABC):
    """Perception-only backend.

    Implementations return VLAResult and must not call robot control, IK, or
    simulator action APIs.
    """

    name: str

    @abstractmethod
    def infer(self, observation: VLAObservation) -> VLAResult:
        """Run VLA inference and return a structured result."""

    def locate(self, observation: VLAObservation) -> VLAResult:
        """Compatibility alias for earlier phase code."""

        return self.infer(observation)


VLABackend = BaseVLABackend
