"""VLA backend manager with mandatory mock fallback."""

from __future__ import annotations

from vla_world_grasp.vla.base import BaseVLABackend
from vla_world_grasp.vla.mock_vla import MockVLA
from vla_world_grasp.vla.openvla_adapter import OpenVLAAdapter
from vla_world_grasp.vla.qwen_vl_adapter import QwenVLAdapter
from vla_world_grasp.vla.schemas import VLAObservation, VLAResult
from vla_world_grasp.vla.smolvla_adapter import SmolVLAAdapter


class VLABackendManager:
    """Selects a VLA backend and falls back to mock_vla on any failure."""

    def __init__(
        self,
        backend_name: str = "mock_vla",
        qwen_model_path: str | None = None,
        qwen_device: str = "cuda",
        qwen_fallback_to_mock: bool = True,
    ) -> None:
        self.backend_name = backend_name
        self.qwen_fallback_to_mock = bool(qwen_fallback_to_mock)
        self.fallback = MockVLA()
        self.backends: dict[str, BaseVLABackend] = {
            "mock_vla": self.fallback,
            "qwen_vl": QwenVLAdapter(model_path=qwen_model_path, device=qwen_device),
            "qwen": QwenVLAdapter(model_path=qwen_model_path, device=qwen_device),
            "smolvla": SmolVLAAdapter(),
            "openvla": OpenVLAAdapter(),
        }

    @classmethod
    def from_name(
        cls,
        backend_name: str,
        qwen_model_path: str | None = None,
        qwen_device: str = "cuda",
        qwen_fallback_to_mock: bool = True,
    ) -> "VLABackendManager":
        return cls(
            backend_name=backend_name,
            qwen_model_path=qwen_model_path,
            qwen_device=qwen_device,
            qwen_fallback_to_mock=qwen_fallback_to_mock,
        )

    def infer(self, observation: VLAObservation) -> VLAResult:
        backend = self.backends.get(self.backend_name)
        if backend is None:
            return self._fallback(
                observation,
                requested_backend=self.backend_name,
                reason="backend_not_found",
            )

        try:
            result = backend.infer(observation)
        except Exception as exc:
            return self._fallback(
                observation,
                requested_backend=backend.name,
                reason=f"exception: {exc}",
            )

        if result.status in {"failed", "not_enabled"} or not result.ok:
            if backend.name == "qwen_vl" and not self.qwen_fallback_to_mock:
                return result
            return self._fallback(
                observation,
                requested_backend=backend.name,
                reason=result.message or result.status,
                failed_metadata=result.metadata,
            )

        return result

    def locate(self, observation: VLAObservation) -> VLAResult:
        """Compatibility alias for earlier phase code."""

        return self.infer(observation)

    def _fallback(
        self,
        observation: VLAObservation,
        requested_backend: str,
        reason: str,
        failed_metadata: dict[str, object] | None = None,
    ) -> VLAResult:
        if requested_backend in {"qwen_vl", "qwen"}:
            print("[vla_world_grasp] Qwen-VL not available, fallback to mock_vla.", flush=True)
        try:
            fallback_result = self.fallback.infer(observation)
        except Exception as exc:
            return VLAResult(
                mode="target_localization",
                backend="mock_vla",
                status="failed",
                message=f"mock_vla fallback failed: {exc}",
                metadata={
                    "requested_backend": requested_backend,
                    "fallback_reason": reason,
                    "fallback_used": True,
                },
            )

        metadata = dict(fallback_result.metadata)
        metadata["requested_backend"] = requested_backend
        metadata["fallback_reason"] = reason
        metadata["fallback_used"] = True
        if failed_metadata:
            metadata["model_path"] = failed_metadata.get("model_path", "")
            metadata["raw_model_output"] = failed_metadata.get("raw_model_output", "")
            metadata["target_label_zh"] = failed_metadata.get("target_label_zh", "")
            metadata["target_label_en"] = failed_metadata.get("target_label_en", "")
            metadata["color"] = failed_metadata.get("color", "")
            metadata["shape"] = failed_metadata.get("shape", "")
            metadata["reason_zh"] = failed_metadata.get("reason_zh", "")
            metadata["parsed_qwen_result"] = failed_metadata.get("parsed_qwen_result", {})
            for key in (
                "semantic_match_used",
                "bbox_point_used_for_target",
                "qwen_bbox_valid",
                "qwen_point_valid",
                "qwen_point_to_matched_object_px",
                "matched_object_projected_point",
                "bbox_validation_warning",
            ):
                if key in failed_metadata:
                    metadata[key] = failed_metadata[key]
        return VLAResult(
            mode=fallback_result.mode,
            backend=fallback_result.backend,
            status=fallback_result.status,
            target=fallback_result.target,
            action_proposal=fallback_result.action_proposal,
            message=f"fallback to mock_vla: {reason}",
            metadata=metadata,
        )


VLAManager = VLABackendManager
