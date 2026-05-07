#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT_DIR}/scripts/run_demo.sh" \
  --instruction "抓起红色方块" \
  --vla_backend qwen_vl \
  --qwen_model_path /home/ubuntu/models/Qwen2.5-VL-3B-Instruct \
  --qwen_device cuda \
  --sim_backend isaac \
  --headless \
  --save_video \
  --video_name qwen_vl_real_red_cube_demo \
  --no_visualize_candidates \
  --debug_attach_on_grasp \
  --score_mode hybrid \
  --score_model_checkpoint outputs/checkpoints/score_model_best.pt \
  --qwen_fallback_to_mock true \
  "$@"
