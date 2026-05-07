# Grasp Success Dataset

## Goal

This dataset trains and evaluates candidate grasp-action scoring models. Each CSV row represents one generated candidate grasp action for one Isaac Sim episode. The debiased collector is designed so target color, object type, gripper width, and grasp success are not trivially coupled.

## Debiased Collection Command

```bash
bash scripts/collect_dataset.sh \
  --num_episodes 100 \
  --candidates_per_episode 16 \
  --headless \
  --output_dir data/grasp_success_debiased \
  --attach_mode mixed \
  --randomize_color_shape \
  --randomize_gripper_width \
  --randomize_candidate_offsets \
  --min_object_distance 0.10
```

The collector uses Isaac Sim / IsaacLab through `scripts/collect_dataset.sh`. It does not use the mock renderer and does not create candidate markers in the Isaac scene.

## Randomization

Each episode samples three tabletop objects with dynamic names:

```text
<color>_<shape>_<index>
```

Supported colors are `red`, `blue`, `green`, `yellow`, `white`, and `black`. Supported shapes are `cube`, `cylinder`, and `sphere`. Object states save `name`, `color`, `type`, `position`, `size`, `mass`, and `friction`.

Objects are placed in the configurable tabletop range `x=[0.35, 0.60]`, `y=[-0.18, 0.18]` with `min_object_distance=0.10` by default. `scene_seed` is saved so the scene can be reproduced.

## Candidate Generation

The candidate generator now includes center, shifted, high, and low candidates:

- `x/y` offsets: `-0.06`, `-0.03`, `0.0`, `0.03`, `0.06`
- yaw samples: `0`, `45`, `90`, `135` degrees
- type-aware `z` samples for cubes, cylinders, and spheres
- randomized type-aware `gripper_width` samples

Each candidate saves geometry diagnostics such as `dx_to_target`, `dy_to_target`, `dz_to_target`, `distance_to_target_center`, `neighbor_clearance`, `table_clearance`, `baseline_score`, and `score_terms`.

## Attach Modes

`--attach_mode mixed` alternates probabilistically between:

- `debug_attach_on_grasp=True`: stable positive trajectories
- `debug_attach_on_grasp=False`: natural physics success/failure examples

CSV fields include `debug_attach_on_grasp`, `attach_mode`, `attached`, and `natural_physics_success`.

## Label Policy

Only the selected candidate is executed in the current collector:

- selected candidate: `selected=True`, `executed=True`, `label_available=True`, `label_source=executed`
- non-selected candidate: `selected=False`, `executed=False`, `label_available=False`, `label_source=not_executed`

Future candidate-level evaluation can set `label_source=candidate_eval` after resetting and executing each candidate independently.

## Training Split

The score model defaults to episode-level splitting:

```bash
bash scripts/train_score_model.sh \
  --dataset_csv data/grasp_success_debiased/processed/grasp_success_dataset.csv \
  --epochs 50 \
  --batch_size 16 \
  --split_by_episode true \
  --include_attach_data true \
  --include_no_attach_data true \
  --use_label_available_only true
```

`outputs/logs/score_model_splits.json` records `train_episode_ids`, `val_episode_ids`, `test_episode_ids`, and row counts for each split. This prevents candidates from the same episode leaking across train/validation/test.

## Analysis

Use:

```bash
bash scripts/inspect_dataset.sh \
  --dataset_csv data/grasp_success_debiased/processed/grasp_success_dataset.csv

bash scripts/analyze_score_dataset.sh \
  --dataset_csv data/grasp_success_debiased/processed/grasp_success_dataset.csv
```

The analysis scripts report color-shape coupling, label availability, attach/no_attach balance, gripper-width distribution, action-offset distribution, and generate dataset plots under `outputs/figures/`.
