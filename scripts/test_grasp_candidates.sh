#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:src"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" - <<'PY'
import math

from vla_world_grasp.grasp.action_space import (
    DEFAULT_APPROACH_HEIGHT,
    TOP_DOWN_APPROACH,
)
from vla_world_grasp.grasp.candidate_generator import (
    baseline_score,
    select_best_candidate,
)
from vla_world_grasp.grasp.vla_to_candidates import vla_result_to_candidates
from vla_world_grasp.vla.schemas import (
    GraspCandidate,
    VLAActionProposal,
    VLAResult,
    VLATarget,
)

target = VLATarget(
    object_id="red_cube",
    label="red cube",
    point_3d=(0.10, -0.02, 0.04),
    confidence=0.9,
    metadata={"size": (0.05, 0.05, 0.05)},
)
result = VLAResult(
    mode="target_localization",
    backend="mock_vla",
    status="ok",
    target=target,
    metadata={"instruction": "抓取红色方块"},
)

candidates = vla_result_to_candidates(result)
assert 10 <= len(candidates) <= 16
assert len(candidates) == 12
assert all(isinstance(candidate, GraspCandidate) for candidate in candidates)
assert all(candidate.source.mode == "target_localization" for candidate in candidates)
assert all(candidate.approach == TOP_DOWN_APPROACH for candidate in candidates)
assert all(candidate.pregrasp_height == DEFAULT_APPROACH_HEIGHT for candidate in candidates)
assert all(0.025 <= candidate.gripper_width <= 0.085 for candidate in candidates)

yaws = sorted({round(math.degrees(candidate.yaw)) for candidate in candidates})
assert yaws == [0, 45, 90, 135], yaws

xy_offsets = {candidate.metadata["xy_offset"] for candidate in candidates}
assert xy_offsets == {(0.0, 0.0), (0.012, 0.0), (-0.012, 0.0)}
assert all("collision_risk" in candidate.metadata for candidate in candidates)

scores = [baseline_score(candidate) for candidate in candidates]
best = select_best_candidate(candidates)
assert best.score == max(score.score for score in scores)
assert best.candidate in candidates
assert best.candidate.position == target.point_3d
assert best.components["distance"] == 1.0

proposal_result = VLAResult(
    mode="action_proposal",
    backend="mock_vla",
    status="ok",
    action_proposal=VLAActionProposal(action_type="top_down_grasp", target=target),
)
proposal_candidates = vla_result_to_candidates(proposal_result)
assert len(proposal_candidates) == 12
assert proposal_candidates[0].source.target == target

candidate_result = VLAResult(
    mode="grasp_candidates",
    backend="mock_vla",
    status="ok",
    metadata={"grasp_candidates": candidates[:2]},
)
assert vla_result_to_candidates(candidate_result) == candidates[:2]

mixed_result = VLAResult(
    mode="mixed",
    backend="mock_vla",
    status="ok",
    target=target,
    metadata={"grasp_candidates": candidates[:1]},
)
mixed_candidates = vla_result_to_candidates(mixed_result)
assert len(mixed_candidates) == 13
assert mixed_candidates[0] == candidates[0]

print("Grasp candidate tests passed")
PY
