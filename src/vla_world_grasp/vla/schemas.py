"""VLA schemas shared by mock, optional adapters, and grasp conversion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Vector3 = tuple[float, float, float]
BoundingBox2D = tuple[int, int, int, int]


@dataclass(frozen=True)
class SceneObject:
    """Small object-state record for non-Isaac tests."""

    object_id: str
    label: str
    color: str
    shape: str
    position: Vector3
    size: Vector3 = (0.05, 0.05, 0.05)
    bbox_xyxy: BoundingBox2D | None = None


@dataclass(frozen=True)
class VLAObservation:
    """Input to every VLA backend."""

    instruction: str = ""
    rgb_path: str | None = None
    depth_path: str | None = None
    camera_intrinsics: dict[str, Any] = field(default_factory=dict)
    camera_pose: dict[str, Any] = field(default_factory=dict)
    object_states: tuple[SceneObject, ...] | list[SceneObject] | tuple[dict[str, Any], ...] | list[dict[str, Any]] = ()
    image: Any | None = None
    depth: Any | None = None
    camera_info: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None

    @property
    def text(self) -> str:
        return self.instruction or self.prompt or ""

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        return normalize_object_states(self.object_states)


@dataclass(frozen=True)
class VLATarget:
    """Localized target selected by a VLA backend."""

    object_id: str
    label: str
    point_3d: Vector3
    confidence: float = 1.0
    bbox_xyxy: BoundingBox2D | None = None
    point_2d: tuple[float, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLAActionProposal:
    """Non-executable action hint.

    VLA backends may describe intent here, but this is not a robot command and
    must not be sent directly to Franka.
    """

    action_type: str
    target: VLATarget | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLAResult:
    """The only allowed output type from VLA backends."""

    mode: str
    backend: str
    status: str = "ok"
    target: VLATarget | None = None
    action_proposal: VLAActionProposal | None = None
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.target is not None

    @property
    def target_id(self) -> str:
        return self.target.object_id if self.target else ""

    @property
    def target_label(self) -> str:
        return self.target.label if self.target else ""

    @property
    def world_position(self) -> Vector3:
        if self.target is None:
            raise ValueError("VLAResult has no target point_3d")
        return self.target.point_3d

    @property
    def confidence(self) -> float:
        return self.target.confidence if self.target else 0.0

    @property
    def bbox_xyxy(self) -> BoundingBox2D | None:
        return self.target.bbox_xyxy if self.target else None

    @property
    def image_point_xy(self) -> tuple[float, float] | None:
        return self.target.point_2d if self.target else None

    @property
    def prompt(self) -> str:
        return str(self.metadata.get("instruction", ""))


@dataclass(frozen=True)
class GraspCandidate:
    """Top-down grasp candidate derived from a VLAResult."""

    candidate_id: str
    position: Vector3
    approach: Vector3
    yaw: float
    gripper_width: float
    pregrasp_height: float
    grasp_depth: float
    source: VLAResult
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScore:
    """Score assigned to a GraspCandidate."""

    candidate: GraspCandidate
    score: float
    components: dict[str, float] = field(default_factory=dict)


def normalize_object_states(
    object_states: tuple[SceneObject, ...] | list[SceneObject] | tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> tuple[SceneObject, ...]:
    normalized = []
    for idx, item in enumerate(object_states):
        if isinstance(item, SceneObject):
            normalized.append(item)
            continue
        normalized.append(
            SceneObject(
                object_id=str(item.get("object_id") or item.get("id") or f"object_{idx}"),
                label=str(item.get("label") or item.get("name") or f"object_{idx}"),
                color=str(item.get("color") or ""),
                shape=str(item.get("shape") or ""),
                position=_vector3(item.get("position") or item.get("point_3d") or item.get("xyz")),
                size=_vector3(item.get("size", (0.05, 0.05, 0.05))),
                bbox_xyxy=item.get("bbox_xyxy"),
            )
        )
    return tuple(normalized)


def _vector3(value: Any) -> Vector3:
    if value is None:
        return (0.0, 0.0, 0.0)
    return (float(value[0]), float(value[1]), float(value[2]))
