"""CLI for random-search score weight optimization."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_world_grasp.world_model.score_weight_optimizer import optimize_score_weights


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize interpretable grasp score weights with random search.")
    parser.add_argument("--dataset_csv", default="data/grasp_score_quick/processed/grasp_success_dataset.csv")
    parser.add_argument("--num_trials", type=int, default=5000)
    parser.add_argument("--selection_metric", default="f1", choices=("accuracy", "precision", "recall", "f1", "auc"))
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--split_by_episode", nargs="?", const=True, default=True, type=_parse_bool)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = optimize_score_weights(
        dataset_csv=args.dataset_csv,
        num_trials=args.num_trials,
        selection_metric=args.selection_metric,
        output_dir=args.output_dir,
        split_by_episode=args.split_by_episode,
    )
    print("[vla_world_grasp] Score weight optimization complete.")
    print("[vla_world_grasp] optimized_weights=")
    for key, value in result["optimized_weights"].items():
        print(f"  {key}: {value:.4f}")
    print(f"[vla_world_grasp] best_threshold={result['best_threshold']:.4f}")
    output_dir = Path(args.output_dir)
    print(f"[vla_world_grasp] {output_dir / 'logs' / 'score_weight_optimization.json'}")
    print(f"[vla_world_grasp] {output_dir / 'figures' / 'score_weights_original_vs_optimized.png'}")
    print(f"[vla_world_grasp] {output_dir / 'figures' / 'score_weight_metrics_comparison.png'}")
    print("[vla_world_grasp] reports/score_weight_optimization.md")


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    main()
