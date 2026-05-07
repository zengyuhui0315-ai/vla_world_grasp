"""Inference helpers and CLI for the grasp score model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from vla_world_grasp.world_model.dataset import load_grasp_score_dataset, row_to_features
from vla_world_grasp.world_model.model import load_checkpoint


def predict_rows(rows: List[Dict[str, str]], checkpoint: str) -> np.ndarray:
    model, scaler, metadata = load_checkpoint(Path(checkpoint))
    category_maps = metadata.get("category_maps", {})
    x = np.asarray([row_to_features(row, category_maps) for row in rows], dtype=np.float64)
    return model.predict_proba(scaler.transform(x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate grasp score model inference.")
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--use_selected_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    _, _, metadata = load_checkpoint(Path(args.checkpoint))
    dataset = load_grasp_score_dataset(
        args.dataset_csv,
        use_selected_only=args.use_selected_only,
        category_maps=metadata.get("category_maps", {}),
    )
    model, scaler, _ = load_checkpoint(Path(args.checkpoint))
    scores = model.predict_proba(scaler.transform(dataset.x)) if len(dataset.x) else np.asarray([])
    records = []
    for row, score in zip(dataset.rows, scores):
        records.append(
            {
                "episode_id": row.get("episode_id", ""),
                "candidate_id": row.get("candidate_id", ""),
                "model_score": float(score),
                "baseline_score": float(row.get("baseline_score", "0") or 0.0),
            }
        )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(records[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
