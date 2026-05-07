# vla_world_model_grasp

Desktop object grasping MVP based on Isaac Sim 4.5, IsaacLab, and Franka Panda.

The project goal is to connect a VLA target-localization backend (`mock_vla` first, Qwen-VL later), RGB-D or object-state based grasp candidate generation, simple candidate scoring, and final Franka top-down grasp execution.

Stage 0 only creates the project skeleton and basic scripts. Isaac Sim logic is intentionally not implemented yet.

## Planned Pipeline

```text
RGB-D / object state
  -> VLAResult
  -> GraspCandidate
  -> baseline_score or world_model_score
  -> best_candidate
  -> Franka top-down grasp
```

## Demo Commands

Run a single demo:

```bash
bash scripts/run_demo.sh
```

Run the terminal Chinese interactive demo:

```bash
bash scripts/run_interactive_demo.sh
```

The script prompts:

```text
请输入中文抓取指令，例如：抓起红色方块 / 抓起蓝色圆柱 / 抓起绿色球
```

Press Enter directly to use `抓起红色方块`. Example inputs:

- `抓起红色方块` -> `outputs/videos/interactive_red_cube_demo.mp4`
- `抓起蓝色圆柱` -> `outputs/videos/interactive_blue_cylinder_demo.mp4`
- `抓起绿色球` -> `outputs/videos/interactive_green_sphere_demo.mp4`

Interactive runs use the `mock` VLA backend for Chinese instruction parsing and target selection, keep candidate markers disabled in the Isaac Sim scene, and still save candidate JSON, Chinese explanations, and score plots.

Run the simulation smoke-test placeholder:

```bash
bash scripts/run_sim_smoke_test.sh
```

Record a demo placeholder:

```bash
bash scripts/record_demo.sh
```

Check VLA interface placeholders:

```bash
bash scripts/test_vla_interfaces.sh
```

Check Qwen2.5-VL grounding availability and fallback behavior:

```bash
bash scripts/test_qwen_vl_grounding.sh
```

Download Qwen2.5-VL-3B weights with low-concurrency retries:

```bash
bash scripts/download_qwen_vl.sh
```

If Hugging Face network downloads are unstable or SSL connections break, retry with the mirror endpoint:

```bash
bash scripts/download_qwen_vl.sh --use_mirror
```

Check an existing local model directory:

```bash
bash scripts/check_qwen_vl_model.sh \
  --model_dir /home/ubuntu/models/Qwen2.5-VL-3B-Instruct
```

Run the Qwen-VL grounding demo. Qwen-VL receives the Isaac Sim camera RGB image and the Chinese instruction, returns a target bbox/point when available, and the existing candidate generation plus Franka controller execute the grasp. If Qwen-VL dependencies or weights are unavailable, the run falls back to `mock_vla` and records `fallback_used=true`.

```bash
timeout 1200 bash scripts/run_qwen_vl_demo.sh
```

Direct command:

```bash
timeout 900 bash scripts/run_demo.sh \
  --instruction "抓起蓝色圆柱" \
  --vla_backend qwen_vl \
  --sim_backend isaac \
  --headless \
  --save_video \
  --video_name qwen_vl_blue_cylinder_demo \
  --no_visualize_candidates \
  --debug_attach_on_grasp \
  --qwen_fallback_to_mock true
```

Check grasp candidate placeholders:

```bash
bash scripts/test_grasp_candidates.sh
```

## Notes

- Server runs should default to headless mode.
- All future Isaac Sim scripts must support `--headless`.
- All runnable scripts should support `--help`.
- Model failures must fall back to `mock_vla`.
- Qwen-VL is a visual-language grounding module only. It does not output robot actions or directly control Franka.
