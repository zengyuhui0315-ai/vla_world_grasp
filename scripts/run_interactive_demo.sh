#!/usr/bin/env bash
set -euo pipefail

export LANG="${LANG:-en_US.UTF-8}"
export LC_ALL="${LC_ALL:-en_US.UTF-8}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

echo "请输入中文抓取指令，例如：抓起红色方块 / 抓起蓝色圆柱 / 抓起绿色球"
read -r instruction

if [[ -z "${instruction}" ]]; then
  instruction="抓起红色方块"
fi

video_name="interactive_zh_demo"
if [[ "${instruction}" == *"红色方块"* || "${instruction}" == *"红方块"* || "${instruction}" == *"red cube"* ]]; then
  video_name="interactive_red_cube_demo"
elif [[ "${instruction}" == *"蓝色圆柱"* || "${instruction}" == *"蓝圆柱"* || "${instruction}" == *"blue cylinder"* ]]; then
  video_name="interactive_blue_cylinder_demo"
elif [[ "${instruction}" == *"绿色球"* || "${instruction}" == *"绿球"* || "${instruction}" == *"green sphere"* || "${instruction}" == *"green ball"* ]]; then
  video_name="interactive_green_sphere_demo"
fi

echo "[vla_world_grasp] 中文指令: ${instruction}"
echo "[vla_world_grasp] video_name=${video_name}"

bash scripts/run_demo.sh \
  --instruction "${instruction}" \
  --vla_backend mock \
  --sim_backend isaac \
  --headless \
  --save_video \
  --video_name "${video_name}" \
  --no_visualize_candidates \
  --explain_scores \
  --save_candidate_scores_plot \
  --debug_attach_on_grasp
