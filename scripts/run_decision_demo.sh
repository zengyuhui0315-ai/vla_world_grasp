#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

bash scripts/run_demo.sh \
  --instruction "${INSTRUCTION:-抓起蓝色圆柱}" \
  --vla_backend "${VLA_BACKEND:-mock}" \
  --sim_backend "${SIM_BACKEND:-isaac}" \
  --headless \
  --save_video \
  --video_name "${VIDEO_NAME:-decision_demo}" \
  --max_steps "${MAX_STEPS:-600}" \
  --frame_interval "${FRAME_INTERVAL:-5}" \
  --no_visualize_candidates \
  --explain_scores \
  --save_candidate_scores_plot \
  --candidate_top_k "${CANDIDATE_TOP_K:-5}" \
  --debug_attach_on_grasp
