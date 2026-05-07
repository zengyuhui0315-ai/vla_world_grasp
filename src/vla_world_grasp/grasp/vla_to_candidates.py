"""Conversion from VLAResult modes to GraspCandidate objects."""

from __future__ import annotations

from vla_world_grasp.grasp.candidate_generator import generate_top_down_candidates
from vla_world_grasp.vla.schemas import GraspCandidate, VLAResult


def vla_result_to_candidates(
    result: VLAResult,
    candidate_count: int | None = None,
) -> list[GraspCandidate]:
    """Convert supported VLAResult modes into top-down grasp candidates."""

    if result.status != "ok":
        raise ValueError(f"cannot convert non-ok VLAResult: {result.status}")

    if result.mode == "target_localization":
        return generate_top_down_candidates(result, count=candidate_count)

    if result.mode == "action_proposal":
        if result.action_proposal is None or result.action_proposal.target is None:
            raise ValueError("action_proposal mode requires action_proposal.target")
        proposal_result = VLAResult(
            mode="target_localization",
            backend=result.backend,
            status=result.status,
            target=result.action_proposal.target,
            message=result.message,
            metadata=result.metadata,
        )
        return generate_top_down_candidates(proposal_result, count=candidate_count)

    if result.mode == "grasp_candidates":
        candidates = result.metadata.get("grasp_candidates")
        if not isinstance(candidates, list):
            raise ValueError("grasp_candidates mode requires metadata['grasp_candidates']")
        if not all(isinstance(candidate, GraspCandidate) for candidate in candidates):
            raise TypeError("metadata['grasp_candidates'] must contain GraspCandidate objects")
        return candidates

    if result.mode == "mixed":
        mixed_candidates = []
        raw_candidates = result.metadata.get("grasp_candidates", [])
        if raw_candidates:
            if not all(isinstance(candidate, GraspCandidate) for candidate in raw_candidates):
                raise TypeError("mixed grasp_candidates must contain GraspCandidate objects")
            mixed_candidates.extend(raw_candidates)
        if result.target is not None:
            mixed_candidates.extend(generate_top_down_candidates(result, count=candidate_count))
        if not mixed_candidates:
            raise ValueError("mixed mode needs target or grasp_candidates")
        return mixed_candidates

    raise ValueError(f"unsupported VLAResult mode: {result.mode}")
