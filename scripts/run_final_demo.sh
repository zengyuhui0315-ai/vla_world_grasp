#!/usr/bin/env bash
set -euo pipefail

export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

instruction="${1:-抓起蓝色圆柱}"
video_name="final_zh_demo"

if [[ "${instruction}" == *"红色方块"* || "${instruction}" == *"红方块"* || "${instruction}" == *"red cube"* ]]; then
  video_name="final_red_cube_demo"
elif [[ "${instruction}" == *"蓝色圆柱"* || "${instruction}" == *"蓝圆柱"* || "${instruction}" == *"blue cylinder"* ]]; then
  video_name="final_blue_cylinder_demo"
elif [[ "${instruction}" == *"绿色球"* || "${instruction}" == *"绿球"* || "${instruction}" == *"green sphere"* || "${instruction}" == *"green ball"* ]]; then
  video_name="final_green_sphere_demo"
fi

echo "[vla_world_grasp] Final demo instruction: ${instruction}"
echo "[vla_world_grasp] video_name=${video_name}"
echo "[vla_world_grasp] outputs/videos/${video_name}.mp4"
echo "[vla_world_grasp] outputs/logs/${video_name}_candidates.json"
echo "[vla_world_grasp] outputs/logs/${video_name}_explanation.txt"
echo "[vla_world_grasp] outputs/figures/${video_name}_candidate_scores.png"

bash scripts/run_demo.sh \
  --instruction "${instruction}" \
  --vla_backend mock \
  --sim_backend isaac \
  --headless \
  --save_video \
  --video_name "${video_name}" \
  --max_steps 700 \
  --frame_interval 5 \
  --no_visualize_candidates \
  --explain_scores \
  --save_candidate_scores_plot \
  --debug_attach_on_grasp
