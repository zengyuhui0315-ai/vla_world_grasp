# VLA Backend Comparison

## mock_vla

`mock_vla` is the deterministic backend used for smoke tests, fallback behavior, and report demos. It reads the instruction and object states, then selects a target using simple color, shape, and spatial rules. It is fast, reproducible, and safe for CI.

## Qwen-VL

Qwen-VL is now wired as a Chinese visual-language grounding backend. In a full Isaac run, `run_episode.py` saves the current Isaac camera RGB image to `outputs/debug/<video_name>_qwen_input.png`, sends that image plus the Chinese instruction to Qwen2.5-VL, and expects JSON containing:

- `target_label_zh`
- `target_label_en`
- `bbox_2d`
- `point_2d`
- `confidence`
- `reason_zh`

The backend returns a `VLAResult(mode="grounding")`. If `point_2d` is available, the pipeline matches it to the nearest projected object-state center, using depth backprojection when depth is available and projected-center matching otherwise. If only `bbox_2d` is available, the bbox center becomes `point_2d`.

Qwen-VL never outputs Franka actions. The selected target object is still passed to the existing candidate generation, candidate scoring, and Franka controller.

If optional dependencies, model weights, GPU memory, JSON parsing, or inference fail, the manager prints:

```text
[vla_world_grasp] Qwen-VL not available, fallback to mock_vla.
```

The run then continues with `mock_vla`, and `outputs/logs/<video_name>_vla_result.json` records `fallback_used=true` plus the fallback reason and raw model output when available.

## SmolVLA Reserved

SmolVLA is reserved as a smaller model option. It is useful for future experiments where runtime cost, local deployment, or fast iteration matters more than maximum perception quality. It should still return the same `VLAResult` structure.

## OpenVLA Reserved

OpenVLA is reserved for experiments with action-oriented VLA models. In this project, even an action-capable model should be adapted into the same perception/action-proposal boundary before robot execution.

## Why The VLA Does Not Directly Control The Robot

The VLA backend is intentionally limited to perception and structured target output. It does not send joint targets, end-effector poses, IK commands, or IsaacLab actions.

This boundary improves safety and debuggability:

- perception failures can be inspected as `VLAResult`;
- candidate generation is deterministic and constrained by robot geometry;
- every executable action is represented as a `GraspCandidate`;
- every candidate is scored before execution;
- Franka only executes the final selected candidate.

This design keeps language/vision reasoning flexible while keeping robot control explicit, auditable, and replaceable.
