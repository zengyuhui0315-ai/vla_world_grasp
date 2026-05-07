# Final Report Material

## Project Background

This project builds a minimal desktop manipulation pipeline around Isaac Sim 4.5, IsaacLab, and a Franka Panda robot. The motivating problem is to connect language-and-vision target selection with robot grasp execution while keeping the control boundary explicit and testable.

## Modified Task Objective

The original broad goal is narrowed to a reliable MVP: given a natural-language instruction, choose one of three tabletop objects, generate top-down grasp candidates, score them, execute the best candidate, and record the episode result. The current object set is `red_cube`, `blue_cylinder`, and `green_sphere`.

## System Architecture

The pipeline is:

```text
scene reset
  -> object states / optional RGB-D
  -> VLABackendManager
  -> VLAResult
  -> GraspCandidate list
  -> baseline_score
  -> best candidate
  -> Franka top-down execution
  -> result.json / frames / figures
```

The implementation keeps simulation, VLA inference, candidate generation, scoring, control, and reporting in separate modules. This makes it possible to run a deterministic mock demo without Isaac, while preserving the Isaac-facing entry points.

## VLAResult / GraspCandidate Interface

`VLAResult` is the only allowed output from a VLA backend. It contains the selected target, confidence, optional image-space data, 3D point, backend name, and metadata.

`GraspCandidate` is the bridge from perception to control. It stores a world-frame grasp position, top-down approach direction, yaw, gripper width, pregrasp height, grasp depth, source `VLAResult`, and generator metadata.

This split is important because target localization is not the same as executable robot control.

## Candidate Action Scoring

The baseline scorer ranks candidates using:

- distance to the target center;
- approximate collision risk;
- gripper width reasonableness;
- workspace validity.

The best score is selected before Franka execution. Later work can replace or augment this with a learned world model scorer, but it should still score candidates rather than bypass the action interface.

## Isaac Sim / Franka Execution

The scene module creates a tabletop setup with Franka Panda, a table, and three simple objects. The control sequence is intentionally minimal:

1. move above the selected grasp point;
2. open the gripper;
3. descend toward the target;
4. close the gripper;
5. lift.

Success is measured by whether the target object's final z height exceeds a lift threshold. On systems without Isaac modules available, the same sequence runs through a deterministic state update for smoke testing and report generation.

## Experimental Results

The recording script runs three instructions:

- 抓起红色方块;
- 抓起蓝色圆柱;
- 抓起中间的物体.

Each run writes `result.json` and visual frames. `scripts/plot_results.sh` aggregates the recorded results and writes `outputs/figures/success_rate_bar.png`.

In the deterministic mock setting, the VLA target selector resolves the requested object from object state and the top-down grasp sequence succeeds when the target is lifted above the configured threshold.

## Limitations

- The current controller is an MVP top-down sequence, not a full physical grasp policy.
- The mock path updates object height deterministically and does not model contact failures.
- Video generation uses rendered summary frames when direct Isaac camera capture is unavailable.
- Qwen-VL, SmolVLA, and OpenVLA adapters are interface placeholders unless their model runtimes are installed.
- Candidate scoring uses hand-written heuristics rather than learned affordance prediction.

## Future Work

- Connect real Isaac camera RGB-D capture to the observation path.
- Replace deterministic lift updates with full Isaac physics contact validation.
- Add Qwen-VL target localization from rendered images.
- Add a lightweight world model scorer for candidate success prediction.
- Expand object geometry, randomized layouts, and failure-case evaluation.
- Record native Isaac Sim camera videos when the workstation rendering stack is available.
