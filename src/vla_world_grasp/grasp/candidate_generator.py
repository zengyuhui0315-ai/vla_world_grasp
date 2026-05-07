"""Generate, score, and serialize top-down grasp candidates."""

from __future__ import annotations

import math
from random import Random
from typing import Any

from vla_world_grasp.grasp.action_space import (
    DEFAULT_APPROACH_HEIGHT,
    DEFAULT_GRASP_DEPTH,
    GRIPPER_WIDTH_BY_TYPE,
    TOP_DOWN_APPROACH,
    XY_OFFSETS,
    YAW_SAMPLES_RAD,
    default_collision_risk,
    estimate_gripper_width,
)
from vla_world_grasp.grasp.geometry import (
    clamp01,
    distance_score,
    height_score,
    is_in_workspace,
    reachability_score,
    table_height,
    xy_distance,
)

DEBIASED_XY_OFFSET_SAMPLES: tuple[float, ...] = (-0.06, -0.03, 0.0, 0.03, 0.06)
Z_SAMPLES_BY_TYPE: dict[str, tuple[float, ...]] = {
    "cube": (0.035, 0.045, 0.055, 0.070),
    "cylinder": (0.055, 0.065, 0.075, 0.090),
    "sphere": (0.050, 0.060, 0.075, 0.090),
}
GRIPPER_WIDTH_SAMPLES_BY_TYPE: dict[str, tuple[float, ...]] = {
    "cube": (0.025, 0.030, 0.035, 0.040, 0.050),
    "cylinder": (0.020, 0.025, 0.030, 0.035, 0.045, 0.055),
    "sphere": (0.020, 0.025, 0.030, 0.040, 0.050),
}
from vla_world_grasp.vla.schemas import (
    CandidateScore,
    GraspCandidate,
    VLAResult,
    VLATarget,
    Vector3,
)


def generate_top_down_candidates(
    result: VLAResult,
    count: int | None = None,
    approach_height: float = DEFAULT_APPROACH_HEIGHT,
    grasp_depth: float = DEFAULT_GRASP_DEPTH,
) -> list[GraspCandidate]:
    """Generate 12-24 geometry candidates around the target center."""

    target = _require_target(result)
    object_size = _target_size(target)
    object_type = _target_type(target)
    candidate_limit = _candidate_count(count)
    table_top_z = float(result.metadata.get("table_top_z", 0.02))
    randomize_offsets = bool(result.metadata.get("randomize_candidate_offsets", True))
    randomize_width = bool(result.metadata.get("randomize_gripper_width", True))
    seed = int(result.metadata.get("scene_seed", 0)) + int(result.metadata.get("candidate_seed_offset", 1009))
    rng = Random(seed)
    candidates: list[GraspCandidate] = []

    if randomize_offsets:
        offset_triplets = _debiased_offset_triplets(object_type, rng, candidate_limit)
    else:
        offset_triplets = [(xy[0], xy[1], max(target.point_3d[2], table_top_z + 0.005)) for xy in XY_OFFSETS]

    max_attempts = max(candidate_limit * 20, candidate_limit)
    for idx, (dx, dy, z_value) in enumerate(offset_triplets[:max_attempts]):
        if len(candidates) >= candidate_limit:
            break
        yaw = YAW_SAMPLES_RAD[idx % len(YAW_SAMPLES_RAD)]
        if randomize_offsets:
            yaw = rng.choice(YAW_SAMPLES_RAD)
        gripper_width = _sample_gripper_width(object_size, object_type, rng, randomize_width)
        position = (
            target.point_3d[0] + dx,
            target.point_3d[1] + dy,
            max(z_value, table_top_z + 0.005),
        )
        collision_risk = default_collision_risk((dx, dy), yaw)
        candidate = GraspCandidate(
            candidate_id=idx,
            position=position,
            approach=TOP_DOWN_APPROACH,
            yaw=yaw,
            gripper_width=gripper_width,
            pregrasp_height=approach_height,
            grasp_depth=grasp_depth,
            source=result,
            metadata={
                "source": "geometry",
                "generator": "top_down_debiased_grid" if randomize_offsets else "top_down_grid",
                "xy_offset": (dx, dy),
                "z_offset": z_value,
                "collision_risk": collision_risk,
                "target_size": object_size,
                "target_type": object_type,
                "approach_height": approach_height,
            },
        )
        candidates.append(candidate)

    if len(candidates) < candidate_limit:
        _append_fallback_candidates(
            candidates=candidates,
            result=result,
            target=target,
            object_size=object_size,
            object_type=object_type,
            table_top_z=table_top_z,
            approach_height=approach_height,
            grasp_depth=grasp_depth,
            candidate_limit=candidate_limit,
        )

    if not 12 <= len(candidates) <= 24:
        raise ValueError(f"generated {len(candidates)} candidates, expected 12-24")
    return candidates


def score_candidate(
    candidate: GraspCandidate,
    target_state: dict[str, Any],
    object_states: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> CandidateScore:
    """Score one candidate using simple geometry and tabletop heuristics."""

    target_position = _vector3(target_state["position"])
    target_size = _vector3(target_state.get("size", (0.05, 0.05, 0.05)))
    target_type = str(target_state.get("type") or target_state.get("shape") or "")
    distance = xy_distance(candidate.position, target_position)
    dz_to_target = float(candidate.position[2] - target_position[2])
    reachability = reachability_score(candidate.position)
    table_top_z = table_height(object_states)
    table_clearance = float(candidate.position[2] - table_top_z)
    height_valid = height_score(candidate.position[2], table_top_z=table_top_z)
    collision_risk = _collision_risk(candidate, target_state, object_states)
    collision_score = 1.0 - collision_risk
    neighbor_clearance = _neighbor_clearance(candidate, target_state, object_states)
    width_match = _width_match_score(candidate.gripper_width, target_size, target_type)
    yaw_score = _yaw_score(candidate.yaw, target_type)
    center_score = distance_score(distance, max_distance=0.08)

    features = {
        "center_score": center_score,
        "reachability_score": reachability,
        "collision_score": collision_score,
        "height_score": height_valid,
        "gripper_width_score": width_match,
        "yaw_score": yaw_score,
    }
    score = (
        0.30 * center_score
        + 0.20 * reachability
        + 0.20 * collision_score
        + 0.15 * height_valid
        + 0.10 * width_match
        + 0.05 * yaw_score
    )
    candidate.metadata["features"] = features
    candidate.metadata["score"] = score
    candidate.metadata["dx_to_target"] = float(candidate.position[0] - target_position[0])
    candidate.metadata["dy_to_target"] = float(candidate.position[1] - target_position[1])
    candidate.metadata["dz_to_target"] = dz_to_target
    candidate.metadata["distance_to_target_center"] = float(
        (
            (candidate.position[0] - target_position[0]) ** 2
            + (candidate.position[1] - target_position[1]) ** 2
            + dz_to_target**2
        )
        ** 0.5
    )
    candidate.metadata["neighbor_clearance"] = neighbor_clearance
    candidate.metadata["table_clearance"] = table_clearance
    candidate.metadata["explanation"] = _candidate_explanation(candidate, target_type, distance, features)
    return CandidateScore(candidate=candidate, score=score, components=features)


def score_and_sort_candidates(
    candidates: list[GraspCandidate],
    target_state: dict[str, Any],
    object_states: list[dict[str, Any]],
) -> list[CandidateScore]:
    scored = [score_candidate(candidate, target_state, object_states) for candidate in candidates]
    return sorted(scored, key=lambda item: item.score, reverse=True)


def baseline_score(candidate: GraspCandidate) -> CandidateScore:
    """Compatibility wrapper for older callers."""

    target_state = {
        "name": candidate.source.target_id,
        "type": candidate.metadata.get("target_type", ""),
        "position": candidate.source.world_position,
        "size": candidate.metadata.get("target_size", (0.05, 0.05, 0.05)),
    }
    return score_candidate(candidate, target_state, [target_state])


def select_best_candidate(candidates: list[GraspCandidate]) -> CandidateScore:
    if not candidates:
        raise ValueError("cannot select a best candidate from an empty list")
    target_state = {
        "name": candidates[0].source.target_id,
        "type": candidates[0].metadata.get("target_type", ""),
        "position": candidates[0].source.world_position,
        "size": candidates[0].metadata.get("target_size", (0.05, 0.05, 0.05)),
    }
    return score_and_sort_candidates(candidates, target_state, [target_state])[0]


def candidate_to_json(candidate: GraspCandidate, score: CandidateScore | None = None) -> dict[str, Any]:
    terms = dict(candidate.metadata.get("features", {}))
    value = score.score if score is not None else float(candidate.metadata.get("score", 0.0))
    explanation = str(candidate.metadata.get("explanation") or "")
    return {
        "candidate_id": int(candidate.candidate_id),
        "position": [float(v) for v in candidate.position],
        "dx_to_target": float(candidate.metadata.get("dx_to_target", 0.0)),
        "dy_to_target": float(candidate.metadata.get("dy_to_target", 0.0)),
        "dz_to_target": float(candidate.metadata.get("dz_to_target", 0.0)),
        "distance_to_target_center": float(candidate.metadata.get("distance_to_target_center", 0.0)),
        "neighbor_clearance": float(candidate.metadata.get("neighbor_clearance", 0.0)),
        "table_clearance": float(candidate.metadata.get("table_clearance", 0.0)),
        "yaw": float(candidate.yaw),
        "gripper_width": float(candidate.gripper_width),
        "approach_height": float(candidate.pregrasp_height),
        "source": "geometry",
        "score": float(value),
        "score_terms": {key: float(terms.get(key, 0.0)) for key in _score_term_names()},
        "explanation": explanation,
    }


def _require_target(result: VLAResult) -> VLATarget:
    if result.target is None:
        raise ValueError("VLAResult must contain target for candidate generation")
    return result.target


def _target_size(target: VLATarget) -> Vector3 | None:
    value = target.metadata.get("size")
    if value is None:
        return None
    return _vector3(value)


def _target_type(target: VLATarget) -> str:
    explicit = str(target.metadata.get("shape") or target.metadata.get("type") or "").strip().lower()
    if explicit:
        return explicit
    return _infer_type_from_text(f"{target.object_id} {target.label}")


def _infer_type_from_text(text: str) -> str:
    tokens = [token for token in text.strip().lower().replace("-", "_").split("_") if token]
    for shape in ("cube", "cylinder", "sphere"):
        if shape in tokens or text.strip().lower().endswith(shape):
            return shape
    return ""


def _candidate_count(count: int | None) -> int:
    if count is None:
        return 16
    return max(12, min(24, int(count)))


def _debiased_offset_triplets(object_type: str, rng: Random, candidate_limit: int) -> list[tuple[float, float, float]]:
    z_values = Z_SAMPLES_BY_TYPE.get(object_type, Z_SAMPLES_BY_TYPE["cube"])
    anchors = [
        (0.0, 0.0, z_values[min(1, len(z_values) - 1)]),
        (-0.06, 0.0, z_values[min(1, len(z_values) - 1)]),
        (0.06, 0.0, z_values[min(1, len(z_values) - 1)]),
        (0.0, -0.06, z_values[min(1, len(z_values) - 1)]),
        (0.0, 0.06, z_values[min(1, len(z_values) - 1)]),
        (0.0, 0.0, z_values[0]),
        (0.0, 0.0, z_values[-1]),
    ]
    all_triplets = [
        (dx, dy, z)
        for dx in DEBIASED_XY_OFFSET_SAMPLES
        for dy in DEBIASED_XY_OFFSET_SAMPLES
        for z in z_values
    ]
    rng.shuffle(all_triplets)
    selected: list[tuple[float, float, float]] = []
    for triplet in anchors + all_triplets:
        if triplet not in selected:
            selected.append(triplet)
        if len(selected) >= candidate_limit:
            break
    return selected


def _sample_gripper_width(
    object_size: Vector3 | None,
    object_type: str,
    rng: Random,
    randomize_width: bool,
) -> float:
    if not randomize_width:
        return estimate_gripper_width(object_size, object_type=object_type)
    samples = GRIPPER_WIDTH_SAMPLES_BY_TYPE.get(object_type)
    if not samples:
        return estimate_gripper_width(object_size, object_type=object_type)
    return float(rng.choice(samples))


def _append_fallback_candidates(
    candidates: list[GraspCandidate],
    result: VLAResult,
    target: VLATarget,
    object_size: Vector3 | None,
    object_type: str,
    table_top_z: float,
    approach_height: float,
    grasp_depth: float,
    candidate_limit: int,
) -> None:
    fallback_offsets = [(0.0, 0.0), (-0.03, 0.0), (0.03, 0.0), (0.0, -0.03), (0.0, 0.03)] + list(XY_OFFSETS)
    fallback_z = Z_SAMPLES_BY_TYPE.get(object_type, Z_SAMPLES_BY_TYPE["cube"])[0]
    while len(candidates) < candidate_limit:
        idx = len(candidates)
        dx, dy = fallback_offsets[idx % len(fallback_offsets)]
        yaw = YAW_SAMPLES_RAD[idx % len(YAW_SAMPLES_RAD)]
        width = estimate_gripper_width(object_size, object_type=object_type)
        position = (
            target.point_3d[0] + dx,
            target.point_3d[1] + dy,
            max(fallback_z, table_top_z + 0.005),
        )
        candidates.append(
            GraspCandidate(
                candidate_id=idx,
                position=position,
                approach=TOP_DOWN_APPROACH,
                yaw=yaw,
                gripper_width=width,
                pregrasp_height=approach_height,
                grasp_depth=grasp_depth,
                source=result,
                metadata={
                    "source": "geometry",
                    "generator": "deterministic_fallback",
                    "xy_offset": (dx, dy),
                    "z_offset": fallback_z,
                    "collision_risk": default_collision_risk((dx, dy), yaw),
                    "target_size": object_size,
                    "target_type": object_type,
                    "approach_height": approach_height,
                },
            )
        )


def _vector3(value: Any) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _collision_risk(
    candidate: GraspCandidate,
    target_state: dict[str, Any],
    object_states: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> float:
    base = float(candidate.metadata.get("collision_risk", 0.2))
    target_name = target_state.get("name") or target_state.get("object_id")
    nearby_penalty = 0.0
    for obj in object_states:
        name = obj.get("name") or obj.get("object_id")
        if name == target_name:
            continue
        distance = xy_distance(candidate.position, _vector3(obj["position"]))
        neighbor_radius = _object_xy_radius(obj)
        clearance = distance - neighbor_radius
        if clearance < 0.10:
            nearby_penalty = max(nearby_penalty, (0.10 - clearance) / 0.10 * 0.45)
    return clamp01(base + nearby_penalty)


def _neighbor_clearance_score(
    candidate: GraspCandidate,
    target_state: dict[str, Any],
    object_states: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> float:
    target_name = target_state.get("name") or target_state.get("object_id")
    nearest_clearance: float | None = None
    for obj in object_states:
        name = obj.get("name") or obj.get("object_id")
        if name == target_name:
            continue
        distance = xy_distance(candidate.position, _vector3(obj["position"]))
        clearance = distance - _object_xy_radius(obj)
        nearest_clearance = clearance if nearest_clearance is None else min(nearest_clearance, clearance)
    if nearest_clearance is None:
        return 1.0
    return clamp01((nearest_clearance - 0.03) / 0.12)


def _neighbor_clearance(
    candidate: GraspCandidate,
    target_state: dict[str, Any],
    object_states: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> float:
    target_name = target_state.get("name") or target_state.get("object_id")
    nearest_clearance: float | None = None
    for obj in object_states:
        name = obj.get("name") or obj.get("object_id")
        if name == target_name:
            continue
        distance = xy_distance(candidate.position, _vector3(obj["position"]))
        clearance = distance - _object_xy_radius(obj)
        nearest_clearance = clearance if nearest_clearance is None else min(nearest_clearance, clearance)
    return 1.0 if nearest_clearance is None else float(nearest_clearance)


def _object_xy_radius(obj: dict[str, Any]) -> float:
    size = _vector3(obj.get("size", (0.05, 0.05, 0.05)))
    return max(size[0], size[1]) / 2.0


def _width_match_score(width: float, target_size: Vector3, target_type: str) -> float:
    expected = GRIPPER_WIDTH_BY_TYPE.get(target_type)
    if expected is None:
        expected = max(target_size[0], target_size[1])
    return clamp01(1.0 - abs(width - expected) / 0.05)


def _yaw_score(yaw: float, target_type: str) -> float:
    if target_type in {"cylinder", "sphere"}:
        return 1.0
    yaw_mod = abs((yaw % math.pi) - math.pi / 2.0)
    axis_aligned = min(abs(yaw % math.pi), yaw_mod)
    return clamp01(1.0 - axis_aligned / (math.pi / 4.0))


def _score_term_names() -> tuple[str, ...]:
    return (
        "center_score",
        "reachability_score",
        "collision_score",
        "height_score",
        "gripper_width_score",
        "yaw_score",
    )


def _candidate_explanation(
    candidate: GraspCandidate,
    target_type: str,
    distance_to_center: float,
    terms: dict[str, float],
) -> str:
    yaw_deg = math.degrees(candidate.yaw)
    return (
        f"候选 {candidate.candidate_id} 距离目标中心 {distance_to_center:.3f} 米，"
        f"center_score={terms['center_score']:.3f}；"
        f"Franka 工作空间可达性 reachability_score={terms['reachability_score']:.3f}；"
        f"与桌面和其他物体的碰撞风险较低程度 collision_score={terms['collision_score']:.3f}；"
        f"抓取高度合理性 height_score={terms['height_score']:.3f}；"
        f"夹爪宽度 {candidate.gripper_width:.3f} 米对 {target_type or '目标物体'} 的匹配分 "
        f"gripper_width_score={terms['gripper_width_score']:.3f}；"
        f"yaw={yaw_deg:.1f} 度的方向适配分 yaw_score={terms['yaw_score']:.3f}。"
    )
