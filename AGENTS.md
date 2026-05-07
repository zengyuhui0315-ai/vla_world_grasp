# AGENTS.md

## Project

`vla_world_model_grasp`

This project builds a desktop object grasping MVP with Isaac Sim 4.5, IsaacLab, and a Franka Panda robot.

The MVP should:

1. Run an Isaac Sim / IsaacLab desktop manipulation scene.
2. Use `mock_vla` or Qwen-VL to locate the target object.
3. Convert RGB-D observations or object state into `GraspCandidate` actions.
4. Score candidates with `baseline_score` or a lightweight `world_model_score`.
5. Select the best candidate.
6. Execute a top-down grasp with Franka Panda.

Prioritize a working demo over architectural completeness.

## Reference Repositories

Use these repositories as references only:

1. `external/IsaacLab`
   - Franka Lift Cube examples
   - Franka IK examples
   - camera examples
   - scene and environment wrappers

2. `external/GenReal_CogRob`
   - IsaacLab + Franka data collection patterns
   - task organization patterns

Do not directly modify code under `external/`.

## Core Pipeline

The required control flow is:

```text
RGB-D / object state
        |
        v
VLA backend: mock_vla or Qwen-VL
        |
        v
VLAResult
        |
        v
GraspCandidate generation
        |
        v
baseline_score or world_model_score
        |
        v
best_candidate
        |
        v
Franka top-down grasp execution
```

## Hard Rules

1. The VLA backend must not directly control Franka.
2. The VLA backend may only output `VLAResult`.
3. Every `VLAResult` must be converted into one or more `GraspCandidate` objects before execution.
4. Every `GraspCandidate` must be scored by `baseline_score` or `world_model_score`.
5. Franka may only execute the final `best_candidate`.
6. If Qwen-VL or any model backend is unavailable, the system must fall back to `mock_vla`.
7. Server execution defaults to headless mode.
8. All Isaac Sim scripts must support `--headless`.
9. All scripts must support `--help`.
10. Keep the MVP simple and runnable. Do not over-design.

## Design Boundaries

### VLA Backend

The VLA backend is a perception and target-localization component, not a robot controller.

Allowed:

- Read RGB images, depth images, camera metadata, text prompts, and object state.
- Return a structured `VLAResult`.
- Use `mock_vla` when model inference is disabled, unavailable, or fails.

Not allowed:

- Sending joint targets.
- Sending end-effector targets.
- Calling IK.
- Calling IsaacLab action APIs.
- Choosing or executing the final robot action.

### VLAResult

`VLAResult` should be the only output type from VLA code.

It should contain target-localization information such as:

- target label or prompt match
- image-space point or bounding box
- confidence
- optional depth or 3D hint
- backend name, for example `mock_vla` or `qwen_vl`
- diagnostic metadata useful for debugging

Keep this structure small and explicit.

### GraspCandidate

`GraspCandidate` is the bridge between perception and control.

It should represent a possible top-down grasp, for example:

- world-frame position
- approach direction
- yaw angle
- gripper width
- pre-grasp height
- grasp depth or final z offset
- source `VLAResult`
- candidate metadata

Candidate generation may use:

- RGB-D projection
- object state from Isaac Sim
- simple geometric heuristics
- fixed top-down grasp assumptions for the MVP

### Scoring

Every candidate must pass through a scoring function before execution.

Use `baseline_score` first unless the task explicitly requires a learned or lightweight world model scorer.

Good MVP scoring signals include:

- distance to target center
- valid depth
- reachable workspace
- collision-free or approximately safe approach
- top-down alignment
- confidence from `VLAResult`

`world_model_score` may be added later, but it must preserve the same boundary: it scores candidates, it does not directly control Franka.

### Franka Execution

Franka execution code should consume only `best_candidate`.

For the MVP, prefer a simple top-down grasp sequence:

1. Move above candidate pose.
2. Open gripper.
3. Descend along the top-down approach axis.
4. Close gripper.
5. Lift.

Use IsaacLab Franka IK and Lift Cube examples as references.

## Runtime Requirements

All runnable scripts must:

- support `--help`
- support `--headless` if they launch Isaac Sim or IsaacLab
- default to headless on servers
- provide clear CLI flags for backend selection where relevant
- fall back to `mock_vla` when model loading or inference fails

Recommended CLI pattern:

```bash
python scripts/run_grasp_demo.py --headless --vla-backend mock_vla
```

Model-enabled runs should still be safe:

```bash
python scripts/run_grasp_demo.py --headless --vla-backend qwen_vl
```

If Qwen-VL is unavailable, the run should log the failure and continue with `mock_vla`.

## Implementation Priorities

Build in this order:

1. Minimal IsaacLab scene with Franka, table, camera, and one graspable object.
2. `mock_vla` returning a deterministic `VLAResult`.
3. RGB-D or object-state conversion from `VLAResult` to `GraspCandidate`.
4. `baseline_score` and best-candidate selection.
5. Franka top-down grasp execution.
6. Optional Qwen-VL backend behind the same `VLAResult` interface.
7. Optional lightweight `world_model_score`.

Do not block the MVP on Qwen-VL, learned scoring, complex scene randomization, or large abstractions.

## Code Organization Guidance

Prefer small modules with clear ownership:

```text
vla_world_model_grasp/
  vla/
    mock_vla.py
    qwen_vl.py
    types.py
  grasp/
    candidates.py
    scoring.py
  sim/
    scene.py
    franka_controller.py
  scripts/
    run_grasp_demo.py
```

This layout is guidance, not a strict requirement. Follow existing project structure once it exists.

## Testing and Validation

At minimum, keep validation focused on demo health:

- `--help` works for every script.
- Isaac Sim scripts accept `--headless`.
- `mock_vla` path runs without external model dependencies.
- VLA backend returns `VLAResult`, not robot actions.
- Candidate generation produces at least one `GraspCandidate`.
- Scoring selects exactly one `best_candidate`.
- Franka execution receives only `best_candidate`.

Prefer simple smoke tests and deterministic mock data early.

## Development Notes

- Do not edit `external/` code.
- Copy or adapt patterns from reference repositories into project-owned modules when needed.
- Keep logs readable and explicit around backend fallback.
- Avoid adding distributed systems, training infrastructure, or heavyweight configuration frameworks unless required for the MVP.
- When in doubt, choose the simplest path that demonstrates target localization, candidate generation, scoring, and Franka execution end to end.
