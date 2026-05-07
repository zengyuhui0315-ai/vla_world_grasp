"""CSV dataset utilities for candidate grasp scoring."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


NUMERIC_FEATURES = [
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
    "approach_height",
    "center_score",
    "reachability_score",
    "collision_score",
    "height_score",
    "gripper_width_score",
    "yaw_score",
    "baseline_score",
]
CATEGORICAL_FEATURES = ["target_type", "target_color"]
LABEL_COLUMN = "success"
SELECTED_COLUMN = "selected"


@dataclass
class GraspScoreDataset:
    x: np.ndarray
    y: np.ndarray
    baseline_score: np.ndarray
    rows: List[Dict[str, str]]
    feature_names: List[str]
    category_maps: Dict[str, List[str]]


def parse_bool(value: object) -> Optional[bool]:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_category_maps(rows: Sequence[Dict[str, str]], columns: Iterable[str]) -> Dict[str, List[str]]:
    maps: Dict[str, List[str]] = {}
    for column in columns:
        values = sorted({row.get(column, "").strip() for row in rows if row.get(column, "").strip()})
        maps[column] = values
    return maps


def row_to_features(row: Dict[str, str], category_maps: Dict[str, List[str]]) -> List[float]:
    values: List[float] = []
    for column in NUMERIC_FEATURES:
        raw = row.get(column, "")
        values.append(float(raw) if str(raw).strip() else 0.0)
    for column in CATEGORICAL_FEATURES:
        raw = row.get(column, "").strip()
        values.extend([1.0 if raw == category else 0.0 for category in category_maps.get(column, [])])
    return values


def feature_names_from_category_maps(category_maps: Dict[str, List[str]]) -> List[str]:
    names = list(NUMERIC_FEATURES)
    for column in CATEGORICAL_FEATURES:
        names.extend([f"{column}={category}" for category in category_maps.get(column, [])])
    return names


def load_grasp_score_dataset(
    dataset_csv: str,
    use_selected_only: bool = True,
    use_label_available_only: bool = True,
    include_attach_data: bool = True,
    include_no_attach_data: bool = True,
    geometry_only: bool = False,
    category_maps: Optional[Dict[str, List[str]]] = None,
) -> GraspScoreDataset:
    path = Path(dataset_csv)
    rows = read_csv_rows(path)
    filtered: List[Dict[str, str]] = []
    has_label_available = bool(rows) and "label_available" in rows[0]
    for row in rows:
        label = parse_bool(row.get(LABEL_COLUMN, ""))
        selected = parse_bool(row.get(SELECTED_COLUMN, ""))
        label_available = parse_bool(row.get("label_available", ""))
        debug_attach = parse_bool(row.get("debug_attach_on_grasp", ""))
        if label is None:
            continue
        if use_label_available_only and has_label_available:
            if label_available is not True:
                continue
        elif use_selected_only and selected is not True:
            continue
        if debug_attach is True and not include_attach_data:
            continue
        if debug_attach is not True and not include_no_attach_data:
            continue
        filtered.append(row)

    if len(filtered) < 20:
        print(f"WARNING: only {len(filtered)} labeled samples available after filtering; training may be unstable.")
    if len(filtered) == 0:
        print("WARNING: no labeled samples available after filtering.")

    maps = {} if geometry_only else (category_maps or infer_category_maps(filtered, CATEGORICAL_FEATURES))
    feature_names = feature_names_from_category_maps(maps)
    x = np.asarray([row_to_features(row, maps) for row in filtered], dtype=np.float64)
    if x.size == 0:
        x = np.zeros((0, len(feature_names)), dtype=np.float64)
    y = np.asarray([1.0 if parse_bool(row.get(LABEL_COLUMN, "")) else 0.0 for row in filtered], dtype=np.float64)
    baseline = np.asarray(
        [float(row.get("baseline_score", "0") or 0.0) for row in filtered],
        dtype=np.float64,
    )
    return GraspScoreDataset(
        x=x,
        y=y,
        baseline_score=baseline,
        rows=filtered,
        feature_names=feature_names,
        category_maps=maps,
    )


def split_indices(num_samples: int, seed: int = 7) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    train_end = int(round(num_samples * 0.70))
    val_end = train_end + int(round(num_samples * 0.15))
    if num_samples >= 3:
        train_end = min(max(train_end, 1), num_samples - 2)
        val_end = min(max(val_end, train_end + 1), num_samples - 1)
    elif num_samples == 2:
        train_end, val_end = 1, 1
    elif num_samples == 1:
        train_end, val_end = 1, 1
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def split_indices_by_episode(rows: Sequence[Dict[str, str]], seed: int = 7) -> Dict[str, np.ndarray]:
    episode_to_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        episode_id = row.get("episode_id", "").strip() or f"__missing_episode_{index}"
        episode_to_indices.setdefault(episode_id, []).append(index)

    episode_ids = sorted(episode_to_indices)
    episode_splits = split_indices(len(episode_ids), seed=seed)
    result: Dict[str, np.ndarray] = {}
    for split_name, episode_idx in episode_splits.items():
        split_episode_ids = [episode_ids[index] for index in episode_idx]
        split_indices_list: List[int] = []
        for episode_id in split_episode_ids:
            split_indices_list.extend(episode_to_indices[episode_id])
        result[split_name] = np.asarray(sorted(split_indices_list), dtype=np.int64)
    return result


def describe_splits(rows: Sequence[Dict[str, str]], splits: Dict[str, np.ndarray]) -> Dict[str, Dict[str, object]]:
    description: Dict[str, Dict[str, object]] = {}
    for split_name, indices in splits.items():
        episode_ids = sorted(
            {
                rows[int(index)].get("episode_id", "").strip() or f"__missing_episode_{int(index)}"
                for index in indices
            }
        )
        description[split_name] = {
            "num_episodes": len(episode_ids),
            "num_samples": int(len(indices)),
            "episode_ids": episode_ids,
            "row_indices": [int(index) for index in indices],
        }
    return description


def iter_batches(indices: np.ndarray, batch_size: int, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        yield shuffled[start : start + batch_size]
