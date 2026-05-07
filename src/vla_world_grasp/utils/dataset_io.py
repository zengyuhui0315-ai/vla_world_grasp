"""UTF-8 dataset IO helpers for grasp success collection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS: tuple[str, ...] = (
    "episode_id",
    "scene_seed",
    "candidate_id",
    "selected",
    "executed",
    "label_available",
    "label_source",
    "success",
    "debug_attach_on_grasp",
    "attach_mode",
    "attached",
    "target_name",
    "target_type",
    "target_color",
    "instruction_zh",
    "action_x",
    "action_y",
    "action_z",
    "dx_to_target",
    "dy_to_target",
    "dz_to_target",
    "distance_to_target_center",
    "neighbor_clearance",
    "table_clearance",
    "yaw",
    "gripper_width",
    "baseline_score",
    "center_score",
    "reachability_score",
    "collision_score",
    "height_score",
    "gripper_width_score",
    "yaw_score",
    "approach_height",
    "candidate_rank",
    "parsed_instruction_json",
    "normalized_target_label_zh",
    "normalized_target_label_en",
    "target_position_x",
    "target_position_y",
    "target_position_z",
    "natural_physics_success",
    "rgb_path",
    "depth_path",
    "object_states_path",
    "candidate_json_path",
    "error_msg",
)


def ensure_dataset_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    raw = root / "raw"
    processed = root / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return {"root": root, "raw": raw, "processed": processed}


def episode_dir(raw_dir: str | Path, episode_index: int) -> Path:
    path = Path(raw_dir) / f"episode_{episode_index:06d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return output


def write_text(path: str | Path, text: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output


def write_dataset_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    return output


def initialize_dataset_csv(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
    return output


def append_dataset_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not output.exists() or output.stat().st_size == 0
    with output.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        stream.flush()
    return output


def read_dataset_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
