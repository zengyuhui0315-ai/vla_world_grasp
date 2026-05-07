"""Evaluation and visualization for the grasp score model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vla_world_grasp.world_model.dataset import GraspScoreDataset, load_grasp_score_dataset
from vla_world_grasp.world_model.metrics import binary_classification_metrics, class_counts, roc_curve_points
from vla_world_grasp.world_model.model import load_checkpoint, sigmoid


def ensure_output_dirs(output_dir: str) -> Dict[str, Path]:
    root = Path(output_dir)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "logs": root / "logs",
        "figures": root / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def evaluate_scores(dataset: GraspScoreDataset, model_score: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    model_metrics = binary_classification_metrics(dataset.y, model_score, threshold=threshold, warn_prefix="model")
    model_metrics_0p5 = binary_classification_metrics(dataset.y, model_score, threshold=0.5, warn_prefix="model_threshold_0p5")
    baseline_metrics = binary_classification_metrics(
        dataset.y,
        dataset.baseline_score,
        threshold=0.5,
        warn_prefix="baseline",
    )
    return {
        "model": model_metrics,
        "model_threshold_0p5": model_metrics_0p5,
        "baseline": baseline_metrics,
        "baseline_metrics": baseline_metrics,
        "model_metrics": model_metrics,
        "accuracy": model_metrics["accuracy"],
        "precision": model_metrics["precision"],
        "recall": model_metrics["recall"],
        "f1": model_metrics["f1"],
        "auc": model_metrics["auc"],
        "comparison": {
            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_f1": baseline_metrics["f1"],
            "baseline_auc": baseline_metrics["auc"],
            "model_accuracy": model_metrics["accuracy"],
            "model_f1": model_metrics["f1"],
            "model_auc": model_metrics["auc"],
        },
    }


def save_metrics(path: Path, metrics: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_training_curve(history: Dict[str, object], output_path: Path) -> None:
    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    plt.figure(figsize=(7, 4.5))
    if epochs and train_loss:
        plt.plot(epochs, train_loss, label="train_loss", marker="o", markersize=2)
    if epochs and val_loss:
        plt.plot(epochs, val_loss, label="val_loss", marker="o", markersize=2)
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Score Model Training Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_validation_metrics_curve(history: Dict[str, object], output_path: Path) -> None:
    epochs = history.get("epoch", [])
    val_acc = history.get("val_accuracy", [])
    val_f1 = history.get("val_f1", [])
    plt.figure(figsize=(7, 4.5))
    if epochs and val_acc:
        plt.plot(epochs, val_acc, label="val_acc", marker="o", markersize=2)
    if epochs and val_f1:
        plt.plot(epochs, val_f1, label="val_f1", marker="o", markersize=2)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(-0.02, 1.02)
    plt.title("Validation Metrics Curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(matrix, output_path: Path) -> None:
    arr = np.asarray(matrix, dtype=np.int64)
    plt.figure(figsize=(4.8, 4.2))
    plt.imshow(arr, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for y in range(2):
        for x in range(2):
            plt.text(x, y, str(arr[y, x]), ha="center", va="center", color="black")
    plt.title("Model Confusion Matrix")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, model_score: np.ndarray, baseline_score: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(5.8, 4.6))
    model_points = roc_curve_points(y_true, model_score)
    baseline_points = roc_curve_points(y_true, baseline_score)
    if model_points is None and baseline_points is None:
        print("WARNING: AUC skipped because only one class is present.")
        plt.text(0.5, 0.5, "ROC unavailable: single label class", ha="center", va="center")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        if baseline_points is not None:
            plt.plot(baseline_points[0], baseline_points[1], label="baseline")
        if model_points is not None:
            plt.plot(model_points[0], model_points[1], label="model")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_score_distribution(dataset: GraspScoreDataset, model_score: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 16)
    plt.hist(dataset.baseline_score, bins=bins, alpha=0.55, label="baseline_score")
    plt.hist(model_score, bins=bins, alpha=0.55, label="model_score")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Baseline vs Model Score Distribution")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def permutation_feature_importance(
    model,
    scaler,
    dataset: GraspScoreDataset,
    seed: int = 7,
) -> Dict[str, float]:
    if len(dataset.y) < 2:
        print("WARNING: permutation feature importance skipped because data is too small.")
        return {name: 0.0 for name in dataset.feature_names}
    rng = np.random.default_rng(seed)
    x_scaled = scaler.transform(dataset.x)
    base_score = model.predict_proba(x_scaled)
    base_f1 = binary_classification_metrics(dataset.y, base_score)["f1"]
    values: Dict[str, float] = {}
    for index, name in enumerate(dataset.feature_names):
        permuted = np.array(x_scaled, copy=True)
        permuted[:, index] = rng.permutation(permuted[:, index])
        score = model.predict_proba(permuted)
        f1 = binary_classification_metrics(dataset.y, score)["f1"]
        values[name] = float(base_f1 - f1)
    return values


def plot_feature_importance(importances: Dict[str, float], output_path: Path, top_k: int = 20) -> None:
    items = sorted(importances.items(), key=lambda item: item[1], reverse=True)[:top_k]
    names = [item[0] for item in items][::-1]
    values = [item[1] for item in items][::-1]
    plt.figure(figsize=(8, max(4.5, 0.28 * len(names))))
    plt.barh(names, values)
    plt.xlabel("F1 drop after permutation")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_evaluation(
    dataset_csv: str,
    checkpoint: str,
    output_dir: str = "outputs",
    use_selected_only: bool = True,
    use_label_available_only: bool = True,
    include_attach_data: bool = True,
    include_no_attach_data: bool = True,
    geometry_only: bool = False,
    split_by_episode: bool = True,
    training_history: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    paths = ensure_output_dirs(output_dir)
    metrics_path = paths["logs"] / "score_model_metrics.json"
    previous_metrics: Dict[str, object] = {}
    if metrics_path.exists():
        try:
            previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_metrics = {}
    model, scaler, metadata = load_checkpoint(Path(checkpoint))
    best_threshold = float(metadata.get("best_threshold", previous_metrics.get("best_threshold", 0.5)))
    dataset = load_grasp_score_dataset(
        dataset_csv,
        use_selected_only=use_selected_only,
        use_label_available_only=use_label_available_only,
        include_attach_data=include_attach_data,
        include_no_attach_data=include_no_attach_data,
        geometry_only=geometry_only,
        category_maps=metadata.get("category_maps", {}),
    )
    if len(dataset.x) == 0:
        empty_metric = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "threshold": best_threshold,
            "num_samples": 0,
        }
        metrics = {
            "warning": "no labeled samples available",
            "best_threshold": best_threshold,
            "positive_count": 0,
            "negative_count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": None,
            "baseline": empty_metric,
            "model": empty_metric,
            "baseline_metrics": empty_metric,
            "model_metrics": empty_metric,
            "comparison": {},
        }
        save_metrics(paths["logs"] / "score_model_metrics.json", metrics)
        return metrics
    model_logits = model.predict_logits(scaler.transform(dataset.x))
    model_score = sigmoid(model_logits)
    metrics = evaluate_scores(dataset, model_score, threshold=best_threshold)
    split_metrics = _metrics_from_checkpoint_splits(dataset, model_score, metadata, best_threshold)
    metrics.update(split_metrics)
    counts = class_counts(dataset.y)
    metrics["best_threshold"] = best_threshold
    metrics["positive_count"] = counts["positive"]
    metrics["negative_count"] = counts["negative"]
    for key in (
        "best_epoch",
        "selection_metric",
        "train_positive_count",
        "train_negative_count",
        "val_positive_count",
        "val_negative_count",
        "test_positive_count",
        "test_negative_count",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
    ):
        if key in metrics and metrics[key] is not None:
            continue
        if key in previous_metrics and previous_metrics[key] is not None:
            metrics[key] = previous_metrics[key]
        elif key in metadata:
            metrics[key] = metadata[key]
    metrics["dataset"] = {
        "dataset_csv": dataset_csv,
        "num_samples": int(len(dataset.y)),
        "use_selected_only": bool(use_selected_only),
        "use_label_available_only": bool(use_label_available_only),
        "include_attach_data": bool(include_attach_data),
        "include_no_attach_data": bool(include_no_attach_data),
        "geometry_only": bool(geometry_only),
        "split_by_episode": bool(split_by_episode),
        "feature_names": dataset.feature_names,
    }
    if training_history is not None:
        metrics["training_history"] = training_history
    elif "training_history" in previous_metrics:
        metrics["training_history"] = previous_metrics["training_history"]
    for key in ("test", "test_baseline"):
        if key in previous_metrics:
            metrics[key] = previous_metrics[key]

    save_metrics(metrics_path, metrics)
    curve_history = metrics.get("training_history")
    if curve_history is not None:
        plot_training_curve(curve_history, paths["figures"] / "score_model_training_curve.png")
        plot_validation_metrics_curve(curve_history, paths["figures"] / "score_model_validation_metrics_curve.png")
    plot_confusion_matrix(metrics["model_threshold_0p5"]["confusion_matrix"], paths["figures"] / "score_model_confusion_matrix_threshold_0p5.png")
    plot_confusion_matrix(metrics["model"]["confusion_matrix"], paths["figures"] / "score_model_confusion_matrix_best_threshold.png")
    plot_confusion_matrix(metrics["model"]["confusion_matrix"], paths["figures"] / "score_model_confusion_matrix.png")
    plot_roc_curve(dataset.y, model_score, dataset.baseline_score, paths["figures"] / "score_model_roc_curve.png")
    plot_score_distribution(dataset, model_score, paths["figures"] / "baseline_vs_model_score_distribution.png")
    importances = permutation_feature_importance(model, scaler, dataset)
    plot_feature_importance(importances, paths["figures"] / "feature_importance_permutation.png")
    metrics["feature_importance_permutation"] = importances
    save_metrics(metrics_path, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the grasp score model.")
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--use_selected_only", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--split_by_episode", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--include_attach_data", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--include_no_attach_data", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--use_label_available_only", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--geometry_only", nargs="?", const=True, default=False, type=_parse_bool)
    args = parser.parse_args()
    metrics = run_evaluation(
        dataset_csv=args.dataset_csv,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        use_selected_only=args.use_selected_only,
        use_label_available_only=args.use_label_available_only,
        include_attach_data=args.include_attach_data,
        include_no_attach_data=args.include_no_attach_data,
        geometry_only=args.geometry_only,
        split_by_episode=args.split_by_episode,
    )
    print(json.dumps(metrics.get("comparison", {}), ensure_ascii=False, indent=2))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _metrics_from_checkpoint_splits(
    dataset: GraspScoreDataset,
    model_score: np.ndarray,
    metadata: Dict[str, object],
    threshold: float,
) -> Dict[str, object]:
    result: Dict[str, object] = {}
    splits = metadata.get("splits", {})
    if not isinstance(splits, dict):
        return result
    for split_name in ("train", "val", "test"):
        split_info = splits.get(split_name, {})
        if not isinstance(split_info, dict):
            continue
        row_indices = split_info.get("row_indices", [])
        if not isinstance(row_indices, list):
            continue
        indices = np.asarray([int(index) for index in row_indices if int(index) < len(dataset.y)], dtype=np.int64)
        counts = class_counts(dataset.y[indices]) if len(indices) else {"positive": 0, "negative": 0}
        result[f"{split_name}_positive_count"] = counts["positive"]
        result[f"{split_name}_negative_count"] = counts["negative"]
        if split_name in {"val", "test"}:
            split_metrics = (
                binary_classification_metrics(dataset.y[indices], model_score[indices], threshold=threshold, warn_prefix=f"{split_name}_model")
                if len(indices)
                else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": None, "confusion_matrix": [[0, 0], [0, 0]]}
            )
            result[f"{split_name}_accuracy"] = split_metrics["accuracy"]
            result[f"{split_name}_precision"] = split_metrics["precision"]
            result[f"{split_name}_recall"] = split_metrics["recall"]
            result[f"{split_name}_f1"] = split_metrics["f1"]
            result[f"{split_name}_auc"] = split_metrics["auc"]
            result[f"{split_name}_metrics"] = split_metrics
    return result


if __name__ == "__main__":
    main()
