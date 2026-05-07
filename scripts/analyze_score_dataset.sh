#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python)}"
DATASET_CSV="data/grasp_success/processed/grasp_success_dataset.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_csv)
      DATASET_CSV="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done

cd "${ROOT_DIR}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT_DIR}/outputs/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}" outputs/figures

PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${DATASET_CSV}" <<'PY'
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vla_world_grasp.world_model.dataset import parse_bool

csv_path = Path(sys.argv[1])
rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
fig_dir = Path("outputs/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

def is_labeled(row):
    if "label_available" in row:
        return parse_bool(row.get("label_available", "")) is True
    return parse_bool(row.get("selected", "")) is True and parse_bool(row.get("success", "")) is not None

def is_success(row):
    return parse_bool(row.get("success", "")) is True

def count_bar(counter, title, xlabel, output_name):
    items = sorted(counter.items())
    labels = [item[0] or "unknown" for item in items]
    values = [item[1] for item in items]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / output_name, dpi=160)
    plt.close()

def success_rate_bar(group_key, title, output_name):
    stats = defaultdict(lambda: [0, 0])
    for row in rows:
        if not is_labeled(row):
            continue
        key = group_key(row)
        stats[key][1] += 1
        if is_success(row):
            stats[key][0] += 1
    labels = sorted(stats)
    rates = [stats[label][0] / stats[label][1] if stats[label][1] else 0.0 for label in labels]
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, rates)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("Success Rate")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / output_name, dpi=160)
    plt.close()

def numeric(row, key):
    try:
        return float(row.get(key, ""))
    except ValueError:
        return None

count_bar(Counter(row.get("target_type", "") for row in rows), "Target Type Distribution", "target_type", "dataset_target_type_distribution.png")
count_bar(Counter(row.get("target_color", "") for row in rows), "Target Color Distribution", "target_color", "dataset_target_color_distribution.png")

colors = sorted({row.get("target_color", "") for row in rows if row.get("target_color", "")})
shapes = sorted({row.get("target_type", "") for row in rows if row.get("target_type", "")})
heat = np.zeros((len(colors), len(shapes)))
for row in rows:
    color = row.get("target_color", "")
    shape = row.get("target_type", "")
    if color in colors and shape in shapes:
        heat[colors.index(color), shapes.index(shape)] += 1
plt.figure(figsize=(6.5, 5.0))
plt.imshow(heat, cmap="YlGnBu")
plt.colorbar(label="Rows")
plt.xticks(range(len(shapes)), shapes)
plt.yticks(range(len(colors)), colors)
for y in range(len(colors)):
    for x in range(len(shapes)):
        plt.text(x, y, int(heat[y, x]), ha="center", va="center")
plt.title("Color-Shape Heatmap")
plt.tight_layout()
plt.savefig(fig_dir / "dataset_color_shape_heatmap.png", dpi=160)
plt.close()

success_rate_bar(lambda row: row.get("target_type", "unknown"), "Success Rate by Target Type", "success_rate_by_target_type.png")
success_rate_bar(lambda row: row.get("target_color", "unknown"), "Success Rate by Target Color", "success_rate_by_target_color.png")
success_rate_bar(lambda row: f"{row.get('target_color', '')}-{row.get('target_type', '')}", "Success Rate by Color-Shape", "success_rate_by_color_shape.png")

labeled = [row for row in rows if is_labeled(row)]
success = [row for row in labeled if is_success(row)]
failure = [row for row in labeled if not is_success(row)]

def hist_success_failure(key, title, output_name, bins=20):
    succ_vals = [numeric(row, key) for row in success]
    fail_vals = [numeric(row, key) for row in failure]
    succ_vals = [value for value in succ_vals if value is not None]
    fail_vals = [value for value in fail_vals if value is not None]
    plt.figure(figsize=(7, 4.6))
    if fail_vals:
        plt.hist(fail_vals, bins=bins, alpha=0.6, label="failure")
    if succ_vals:
        plt.hist(succ_vals, bins=bins, alpha=0.6, label="success")
    plt.title(title)
    plt.xlabel(key)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / output_name, dpi=160)
    plt.close()

hist_success_failure("gripper_width", "Gripper Width Success/Failure Distribution", "gripper_width_success_failure_distribution.png", bins=12)
hist_success_failure("distance_to_target_center", "Action Offset Success/Failure Distribution", "action_offset_success_failure_distribution.png", bins=16)
hist_success_failure("baseline_score", "Baseline Score Success/Failure Distribution", "baseline_score_success_failure_distribution.png", bins=16)

features = [
    "action_x", "action_y", "action_z", "dx_to_target", "dy_to_target", "dz_to_target",
    "distance_to_target_center", "neighbor_clearance", "table_clearance", "yaw",
    "gripper_width", "center_score", "reachability_score", "collision_score",
    "height_score", "gripper_width_score", "yaw_score", "baseline_score",
]
matrix = []
valid_names = []
for name in features:
    vals = [numeric(row, name) for row in rows]
    vals = [value for value in vals if value is not None]
    if vals:
        valid_names.append(name)

for row in rows:
    values = [numeric(row, name) for name in valid_names]
    if all(value is not None for value in values):
        matrix.append(values)

if len(matrix) >= 2 and len(valid_names) >= 2:
    arr = np.asarray(matrix, dtype=float)
    corr = np.corrcoef(arr, rowvar=False)
    corr = np.nan_to_num(corr)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(valid_names)), valid_names, rotation=70, ha="right", fontsize=8)
    plt.yticks(range(len(valid_names)), valid_names, fontsize=8)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_correlation_heatmap.png", dpi=160)
    plt.close()
else:
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.5, "Not enough numeric rows", ha="center", va="center")
    plt.axis("off")
    plt.savefig(fig_dir / "feature_correlation_heatmap.png", dpi=160)
    plt.close()

for path in sorted(fig_dir.glob("dataset_*distribution.png")) + sorted(fig_dir.glob("*heatmap.png")):
    pass
print(f"Saved dataset analysis figures to {fig_dir}")
PY
