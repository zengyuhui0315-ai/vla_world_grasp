# Demo Script

## Goal

Show the complete MVP loop: natural-language instruction, VLA target localization, grasp candidate generation, baseline scoring, Franka top-down execution, saved result, and recorded visual material.

## Setup

Run from the project root:

```bash
bash scripts/record_demo.sh --headless
bash scripts/plot_results.sh
```

The recording script runs three instructions:

- 抓起红色方块
- 抓起蓝色圆柱
- 抓起中间的物体

Each episode writes `result.json`, RGB frames, and a `video_manifest.json`. If `ffmpeg` is installed, the same folder also contains `demo.mp4`.

## Walkthrough

1. Open with the task: a Franka Panda should pick one object from a tabletop scene using a VLA-style perception interface.
2. Show the object state list for `red_cube`, `blue_cylinder`, and `green_sphere`.
3. Run the first command and point out the selected `VLAResult.target`.
4. Show the generated top-down candidates and the selected candidate score.
5. Explain the execution sequence: move above, open gripper, move down, close gripper, lift.
6. Show the success criterion: target object final z height must exceed the threshold.
7. Open `outputs/figures/success_rate_bar.png` as the compact experiment summary.

## Notes

On a workstation with Isaac Sim 4.5 and IsaacLab configured, the same CLI remains the entry point. On lightweight/headless CI, the deterministic mock path still produces reproducible results and report assets.
