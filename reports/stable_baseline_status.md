# Stable Baseline Status

## Baseline Command

Run from the project root:

```bash
bash scripts/run_demo.sh \
  --instruction "抓起蓝色圆柱" \
  --vla_backend mock \
  --sim_backend isaac \
  --headless \
  --save_video \
  --video_name stable_baseline_demo \
  --max_steps 500 \
  --frame_interval 5 \
  --no_visualize_candidates \
  --debug_attach_on_grasp
```

## Validation Status

- The command is accepted by the current CLI and enters the Isaac Sim / IsaacLab backend.
- `scripts/run_demo.sh` resolves IsaacLab at `/home/ubuntu/work/external/IsaacLab` and launches through `isaaclab.sh`; the Isaac startup path was not changed.
- On this execution host, the run could not complete because Isaac Sim reported no usable CUDA/Vulkan GPU and read-only cache/config writes under the Isaac Python environment.
- No mock renderer fallback was used for this baseline validation.

## Currently Supported Features

- Creates the Isaac Sim tabletop Franka scene with a Franka Panda robot, table, camera, and default objects.
- Supports the deterministic mock VLA backend for target selection from natural-language instructions.
- Generates and scores grasp candidates, then executes the selected top-down grasp.
- Supports hidden candidate markers via `--no_visualize_candidates`.
- Supports Isaac camera video recording with `--save_video`, `--video_name`, and `--frame_interval`.
- Supports the current Isaac attach-on-grasp behavior used by the stable grasp path.

## Do Not Modify Casually

These files are part of the stable baseline surface and should not be changed without an explicit module-level reason and a baseline re-run:

- `scripts/run_demo.sh`
- `src/vla_world_grasp/pipeline/run_episode.py`
- `src/vla_world_grasp/sim/controller.py`
- `src/vla_world_grasp/sim/scene.py`
- `src/vla_world_grasp/sim/robot.py`
- `src/vla_world_grasp/sim/objects.py`
- `src/vla_world_grasp/sim/sensors.py`
- `src/vla_world_grasp/sim/recorder.py`

## Baseline Protection Rule

Future modules must not break this baseline command or silently fall back from Isaac Sim to the mock renderer. Changes that touch the stable files above must preserve:

- Isaac Sim / IsaacLab launch through `scripts/run_demo.sh`;
- Franka/table/object creation in Isaac;
- successful mock-VLA blue-cylinder target selection;
- hidden candidate visualization when `--no_visualize_candidates` is set;
- top-down grasp execution and attach-on-grasp behavior;
- Isaac video output behavior when `--save_video` is set.
