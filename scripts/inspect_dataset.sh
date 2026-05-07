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
PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${DATASET_CSV}" <<'PY'
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from vla_world_grasp.world_model.dataset import parse_bool

SHAPES = ("cube", "cylinder", "sphere")


def truthy(row, key):
    return parse_bool(row.get(key, "")) is True


def labeled(row):
    if "label_available" in row:
        return truthy(row, "label_available")
    return truthy(row, "selected") and parse_bool(row.get("success", "")) is not None


def normalize_type(row):
    value = str(row.get("target_type", "")).strip().lower()
    if value in SHAPES:
        return value
    name = str(row.get("target_name", "")).strip().lower()
    tokens = [token for token in name.replace("-", "_").split("_") if token]
    for shape in SHAPES:
        if shape in tokens or name.endswith(shape):
            return shape
    return value


def normalize_color(row):
    return str(row.get("target_color", "")).strip().lower()


def success_value(row):
    return parse_bool(row.get("success", ""))


def add_rate(bucket, key, row):
    bucket[key][1] += 1
    if success_value(row) is True:
        bucket[key][0] += 1


def print_counter(title, counter):
    print(f"{title}:")
    if not counter:
        print("  <empty>: 0")
        return
    for key, value in sorted(counter.items(), key=lambda item: str(item[0])):
        label = key if key != "" else "<empty>"
        print(f"  {label}: {value}")


def print_rates(title, rates, numeric_keys=False):
    print(f"{title}:")
    if not rates:
        print("  <empty>: 0.000 (0/0)")
        return
    def sort_key(item):
        key = item[0]
        if numeric_keys:
            try:
                return (0, int(key))
            except (TypeError, ValueError):
                return (1, str(key))
        return (0, str(key))
    for key, (success, total) in sorted(rates.items(), key=sort_key):
        label = key if key != "" else "<empty>"
        rate = success / total if total else 0.0
        print(f"  {label}: {rate:.3f} ({success}/{total})")


csv_path = Path(sys.argv[1])
if not csv_path.exists():
    raise SystemExit(f"dataset csv not found: {csv_path}")

with csv_path.open("r", encoding="utf-8", newline="") as stream:
    rows = list(csv.DictReader(stream))

labeled_rows = [row for row in rows if labeled(row)]
executed_rows = [row for row in rows if truthy(row, "executed")]
selected_rows = [row for row in rows if truthy(row, "selected")]
success_rows = [row for row in labeled_rows if success_value(row) is True]
failure_rows = [row for row in labeled_rows if success_value(row) is False]
episodes = sorted({row.get("episode_id", "") for row in rows if row.get("episode_id")})

label_source_counts = Counter(row.get("label_source", "") or "<empty>" for row in rows)
target_color_counts = Counter(normalize_color(row) for row in rows if normalize_color(row))
target_type_counts = Counter(normalize_type(row) for row in rows if normalize_type(row))
combo_counts = Counter((normalize_color(row), normalize_type(row)) for row in rows if normalize_color(row) and normalize_type(row))
attach_counts = Counter("attach" if truthy(row, "debug_attach_on_grasp") else "no_attach" for row in labeled_rows)
width_counts = Counter(row.get("gripper_width", "") for row in rows if row.get("gripper_width", ""))

print(f"总 episode 数: {len(episodes)}")
print(f"总 candidate rows: {len(rows)}")
print(f"label_available=True 数量: {len(labeled_rows)}")
print(f"executed=True 数量: {len(executed_rows)}")
print(f"selected=True 数量: {len(selected_rows)}")
print_counter("label_source 分布", label_source_counts)
print(f"success 数量: {len(success_rows)}")
print(f"failure 数量: {len(failure_rows)}")
if labeled_rows:
    print(f"label_available 正负样本比例: positive={len(success_rows)} negative={len(failure_rows)}")
print_counter("attach/no_attach 数量", attach_counts)
print_counter("target_color 分布", target_color_counts)
print_counter("target_type 分布", target_type_counts)
print_counter("color-shape 组合分布", Counter(f"{color}-{shape}" for (color, shape), count in combo_counts.items() for _ in range(count)))

success_by_combo = defaultdict(lambda: [0, 0])
success_by_type = defaultdict(lambda: [0, 0])
success_by_color = defaultdict(lambda: [0, 0])
success_by_rank = defaultdict(lambda: [0, 0])
for row in labeled_rows:
    color = normalize_color(row)
    shape = normalize_type(row)
    add_rate(success_by_combo, f"{color}-{shape}", row)
    add_rate(success_by_type, shape, row)
    add_rate(success_by_color, color, row)
    rank = row.get("candidate_rank", "")
    if rank:
        add_rate(success_by_rank, rank, row)

print_rates("success rate by target_type", success_by_type)
print_rates("success rate by target_color", success_by_color)
print_rates("success rate by color-shape", success_by_combo)
print_rates("success rate by candidate rank", success_by_rank, numeric_keys=True)

for shape in SHAPES:
    success, total = success_by_type.get(shape, [0, 0])
    rate = success / total if total else 0.0
    print(f"{shape} success rate: {rate:.3f} ({success}/{total})")
    if total == 0 and any(shape in str(row.get("target_name", "")).lower() for row in rows):
        print(f"[WARN] {shape} appears in dynamic target_name but was not counted; check target_type parsing.")

print_counter("gripper_width 分布", width_counts)


def numeric_values(name):
    values = []
    for row in rows:
        try:
            if row.get(name, "") != "":
                values.append(float(row[name]))
        except ValueError:
            pass
    return values


for name in ("dx_to_target", "dy_to_target", "dz_to_target", "distance_to_target_center"):
    values = numeric_values(name)
    if values:
        print(f"{name} 分布: min={min(values):.4f} max={max(values):.4f} unique={len(set(round(v, 4) for v in values))}")
    else:
        print(f"{name} 分布: 无")

color_to_shapes = defaultdict(set)
for color, shape in combo_counts:
    color_to_shapes[color].add(shape)
strong_pairs = {"red": "cube", "blue": "cylinder", "green": "sphere"}
strongly_coupled = all(color_to_shapes.get(color) == {shape} for color, shape in strong_pairs.items() if color in color_to_shapes)
if strongly_coupled:
    print("[WARN] Color and shape are strongly coupled. Dataset may be biased.")
else:
    print("颜色和形状强绑定检查: 通过")

if target_type_counts:
    max_count = max(target_type_counts.values())
    min_count = min(target_type_counts.values())
    if min_count == 0 or max_count / max(min_count, 1) > 3.0:
        print("[WARN] target_type 类别严重不平衡。")
    else:
        print("类别平衡检查: 通过")

if labeled_rows and (not success_rows or not failure_rows):
    print("[WARN] labeled samples contain only one class.")
if len(labeled_rows) < 20:
    print("[WARN] label_available=True samples are very few.")

print(f"episode-level split 是否可用: {'通过' if len(episodes) >= 3 else 'episode 太少'}")
PY
