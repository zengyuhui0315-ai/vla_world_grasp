"""Train the first-version MLP grasp candidate scoring model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from vla_world_grasp.world_model.dataset import (
    describe_splits,
    iter_batches,
    load_grasp_score_dataset,
    split_indices,
    split_indices_by_episode,
)
from vla_world_grasp.world_model.eval import ensure_output_dirs, run_evaluation
from vla_world_grasp.world_model.metrics import (
    best_threshold_by_f1,
    binary_auc,
    binary_classification_metrics,
    class_counts,
)
from vla_world_grasp.world_model.model import MLPBinaryClassifier, StandardScaler, save_checkpoint, sigmoid


def bce_loss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    eps = 1e-8
    return float(-np.mean(y_true * np.log(y_score + eps) + (1.0 - y_true) * np.log(1.0 - y_score + eps)))


def bce_with_logits_loss(y_true: np.ndarray, logits: np.ndarray, pos_weight: float | None = None) -> float:
    if len(y_true) == 0:
        return 0.0
    y = y_true.reshape(-1)
    z = logits.reshape(-1)
    weight = 1.0 if pos_weight is None else float(pos_weight)
    positive_loss = np.logaddexp(0.0, -z) * y * weight
    negative_loss = np.logaddexp(0.0, z) * (1.0 - y)
    return float(np.mean(positive_loss + negative_loss))


def evaluate_loss(model: MLPBinaryClassifier, x: np.ndarray, y: np.ndarray, pos_weight: float | None = None) -> float:
    if len(y) == 0:
        return 0.0
    return bce_with_logits_loss(y, model.predict_logits(x), pos_weight=pos_weight)


def train_score_model(args: argparse.Namespace) -> Dict[str, object]:
    paths = ensure_output_dirs(args.output_dir)
    dataset = load_grasp_score_dataset(
        args.dataset_csv,
        use_selected_only=args.use_selected_only,
        use_label_available_only=args.use_label_available_only,
        include_attach_data=args.include_attach_data,
        include_no_attach_data=args.include_no_attach_data,
        geometry_only=args.geometry_only,
    )
    if len(dataset.x) == 0:
        checkpoint = paths["checkpoints"] / "score_model_best.pt"
        scaler = StandardScaler(mean=np.zeros((len(dataset.feature_names),)), std=np.ones((len(dataset.feature_names),)))
        model = MLPBinaryClassifier(input_dim=len(dataset.feature_names))
        save_checkpoint(
            checkpoint,
            model,
            scaler,
            {
                "feature_names": dataset.feature_names,
                "category_maps": dataset.category_maps,
                "warning": "saved untrained model because no labeled samples were available",
            },
        )
        empty_counts = {
            "train_positive_count": 0,
            "train_negative_count": 0,
            "val_positive_count": 0,
            "val_negative_count": 0,
            "test_positive_count": 0,
            "test_negative_count": 0,
        }
        empty_metric = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": None,
            "confusion_matrix": [[0, 0], [0, 0]],
            "threshold": 0.5,
            "num_samples": 0,
        }
        metrics = {
            "warning": "no labeled samples available",
            "best_epoch": 0,
            "best_threshold": 0.5,
            **empty_counts,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": None,
            "baseline_metrics": empty_metric,
            "model_metrics": empty_metric,
            "comparison": {},
            "training_history": {},
        }
        (paths["logs"] / "score_model_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metrics

    splits = split_indices_by_episode(dataset.rows, seed=7) if args.split_by_episode else split_indices(len(dataset.y), seed=7)
    split_description = describe_splits(dataset.rows, splits)
    print("[vla_world_grasp] Split by episode_id enabled." if args.split_by_episode else "[vla_world_grasp] Split by row enabled.")
    split_payload = {
        "dataset_csv": args.dataset_csv,
        "use_selected_only": bool(args.use_selected_only),
        "use_label_available_only": bool(args.use_label_available_only),
        "include_attach_data": bool(args.include_attach_data),
        "include_no_attach_data": bool(args.include_no_attach_data),
        "geometry_only": bool(args.geometry_only),
        "split_by_episode": bool(args.split_by_episode),
        "train_episode_ids": split_description["train"]["episode_ids"],
        "val_episode_ids": split_description["val"]["episode_ids"],
        "test_episode_ids": split_description["test"]["episode_ids"],
        "train_num_rows": split_description["train"]["num_samples"],
        "val_num_rows": split_description["val"]["num_samples"],
        "test_num_rows": split_description["test"]["num_samples"],
        "splits": split_description,
    }
    split_path = paths["logs"] / "score_model_splits.json"
    split_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for split_name in ("train", "val", "test"):
        info = split_description[split_name]
        print(f"[vla_world_grasp] {split_name} episodes={info['num_episodes']}, rows={info['num_samples']}")
    print(f"Saved splits: {split_path}")

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    split_counts: Dict[str, Dict[str, int]] = {}
    for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        counts = class_counts(dataset.y[indices])
        split_counts[split_name] = counts
        print(f"[vla_world_grasp] {split_name} positives={counts['positive']}, negatives={counts['negative']}")
    if len(train_idx) < 10:
        print(f"WARNING: train split has only {len(train_idx)} samples; model quality may be poor.")
    train_positive_count = split_counts["train"]["positive"]
    train_negative_count = split_counts["train"]["negative"]
    if not args.use_pos_weight:
        pos_weight = None
        print("[vla_world_grasp] use_pos_weight=False; using ordinary BCEWithLogits loss.")
    elif train_positive_count == 0 or train_negative_count == 0:
        print("WARNING: train split contains a single class; falling back to ordinary BCEWithLogits loss.")
        pos_weight = None
    else:
        pos_weight = train_negative_count / max(train_positive_count, 1)
        print(f"[vla_world_grasp] pos_weight={pos_weight:.4f}")

    scaler = StandardScaler.fit(dataset.x[train_idx])
    x_scaled = scaler.transform(dataset.x)
    model = MLPBinaryClassifier(input_dim=x_scaled.shape[1], hidden_dims=(64, 32), seed=7)
    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    best_selection_key = None
    best_val_metrics: Dict[str, object] = {}
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auc": [],
        "best_threshold": [],
    }

    for epoch in range(1, args.epochs + 1):
        batch_losses = []
        for batch_idx in iter_batches(train_idx, args.batch_size, seed=epoch):
            loss = model.train_batch(x_scaled[batch_idx], dataset.y[batch_idx], lr=args.lr, pos_weight=pos_weight)
            batch_losses.append(loss)
        train_loss = float(np.mean(batch_losses)) if batch_losses else evaluate_loss(model, x_scaled[train_idx], dataset.y[train_idx], pos_weight=pos_weight)
        val_logits = model.predict_logits(x_scaled[val_idx]) if len(val_idx) else np.asarray([])
        val_scores = sigmoid(val_logits) if len(val_idx) else np.asarray([])
        val_loss = bce_with_logits_loss(dataset.y[val_idx], val_logits, pos_weight=None)
        if len(val_idx):
            if args.search_best_threshold:
                epoch_threshold, val_metrics = best_threshold_by_f1(dataset.y[val_idx], val_scores)
            else:
                epoch_threshold = 0.5
                val_metrics = binary_classification_metrics(dataset.y[val_idx], val_scores, threshold=0.5, compute_auc=False)
            val_auc = binary_auc(dataset.y[val_idx], val_scores)
            if val_auc is None:
                print("WARNING: val: AUC skipped because only one class is present.")
            val_metrics["auc"] = val_auc
        else:
            epoch_threshold = 0.5
            val_auc = None
            val_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": None, "confusion_matrix": [[0, 0], [0, 0]]}
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_precision"].append(float(val_metrics["precision"]))
        history["val_recall"].append(float(val_metrics["recall"]))
        history["val_f1"].append(float(val_metrics["f1"]))
        history["val_auc"].append(None if val_auc is None else float(val_auc))
        history["best_threshold"].append(float(epoch_threshold))

        selection_key = _selection_key(
            args.selection_metric,
            float(val_metrics["f1"]),
            None if val_auc is None else float(val_auc),
            float(val_loss),
            float(val_metrics["accuracy"]),
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            best_state = {name: value.copy() for name, value in model.params.items()}
            best_epoch = epoch
            best_threshold = float(epoch_threshold)
            best_val_metrics = dict(val_metrics)

        if epoch == 1 or epoch == args.epochs or epoch % max(args.epochs // 10, 1) == 0:
            val_auc_text = "nan" if val_auc is None else f"{float(val_auc):.3f}"
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_metrics['accuracy']:.3f} "
                f"val_f1={val_metrics['f1']:.3f} val_auc={val_auc_text} "
                f"best_threshold={epoch_threshold:.2f}"
            )

    if best_state is not None:
        model.params = best_state
    checkpoint = paths["checkpoints"] / "score_model_best.pt"
    save_checkpoint(
        checkpoint,
        model,
        scaler,
        {
            "feature_names": dataset.feature_names,
            "category_maps": dataset.category_maps,
            "splits": split_description,
            "split_by_episode": bool(args.split_by_episode),
            "use_label_available_only": bool(args.use_label_available_only),
            "include_attach_data": bool(args.include_attach_data),
            "include_no_attach_data": bool(args.include_no_attach_data),
            "geometry_only": bool(args.geometry_only),
            "selection_metric": args.selection_metric,
            "use_pos_weight": bool(args.use_pos_weight),
            "search_best_threshold": bool(args.search_best_threshold),
            "best_epoch": int(best_epoch),
            "best_threshold": float(best_threshold),
            "pos_weight": pos_weight,
            "train_positive_count": split_counts["train"]["positive"],
            "train_negative_count": split_counts["train"]["negative"],
            "val_positive_count": split_counts["val"]["positive"],
            "val_negative_count": split_counts["val"]["negative"],
            "test_positive_count": split_counts["test"]["positive"],
            "test_negative_count": split_counts["test"]["negative"],
            "input_features": dataset.feature_names,
            "label": "success",
            "model_type": "numpy_mlp_binary_classifier",
        },
    )
    metrics = run_evaluation(
        dataset_csv=args.dataset_csv,
        checkpoint=str(checkpoint),
        output_dir=args.output_dir,
        use_selected_only=args.use_selected_only,
        use_label_available_only=args.use_label_available_only,
        include_attach_data=args.include_attach_data,
        include_no_attach_data=args.include_no_attach_data,
        geometry_only=args.geometry_only,
        training_history=history,
    )
    val_scores_best = sigmoid(model.predict_logits(x_scaled[val_idx])) if len(val_idx) else np.asarray([])
    test_scores = sigmoid(model.predict_logits(x_scaled[test_idx])) if len(test_idx) else np.asarray([])
    val_metrics_best = (
        binary_classification_metrics(dataset.y[val_idx], val_scores_best, threshold=best_threshold, warn_prefix="val_model")
        if len(val_idx)
        else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": None, "confusion_matrix": [[0, 0], [0, 0]]}
    )
    test_metrics = (
        binary_classification_metrics(dataset.y[test_idx], test_scores, threshold=best_threshold, warn_prefix="test_model")
        if len(test_idx)
        else {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": None, "confusion_matrix": [[0, 0], [0, 0]]}
    )
    test_baseline = (
        binary_classification_metrics(dataset.y[test_idx], dataset.baseline_score[test_idx], threshold=0.5, warn_prefix="test_baseline")
        if len(test_idx)
        else {}
    )
    metrics.update(
        {
            "best_epoch": int(best_epoch),
            "best_threshold": float(best_threshold),
            "selection_metric": args.selection_metric,
            "train_positive_count": split_counts["train"]["positive"],
            "train_negative_count": split_counts["train"]["negative"],
            "val_positive_count": split_counts["val"]["positive"],
            "val_negative_count": split_counts["val"]["negative"],
            "test_positive_count": split_counts["test"]["positive"],
            "test_negative_count": split_counts["test"]["negative"],
            "val_accuracy": val_metrics_best["accuracy"],
            "val_precision": val_metrics_best["precision"],
            "val_recall": val_metrics_best["recall"],
            "val_f1": val_metrics_best["f1"],
            "val_auc": val_metrics_best["auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["auc"],
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "auc": test_metrics["auc"],
            "val_metrics": val_metrics_best,
            "test": test_metrics,
            "test_baseline": test_baseline,
            "best_val_metrics_at_checkpoint": best_val_metrics,
        }
    )
    (paths["logs"] / "score_model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint: {checkpoint}")
    print(f"Saved metrics: {paths['logs'] / 'score_model_metrics.json'}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP grasp candidate score model.")
    parser.add_argument("--dataset_csv", default="data/grasp_success/processed/grasp_success_dataset.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--use_selected_only", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--split_by_episode", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--include_attach_data", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--include_no_attach_data", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--use_label_available_only", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--geometry_only", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--selection_metric", choices=("f1", "auc", "loss"), default="f1")
    parser.add_argument("--use_pos_weight", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--search_best_threshold", nargs="?", const=True, default=True, type=_parse_bool)
    args = parser.parse_args()
    train_score_model(args)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _selection_key(selection_metric: str, val_f1: float, val_auc: float | None, val_loss: float, val_acc: float) -> tuple[float, ...]:
    if selection_metric == "loss":
        return (-val_loss,)
    if selection_metric == "auc":
        if val_auc is None:
            return (0.0, -val_loss)
        return (1.0, val_auc, val_f1, val_acc)
    if val_f1 > 0.0:
        return (2.0, val_f1, val_acc)
    if val_auc is not None:
        return (1.0, val_auc, -val_loss)
    return (0.0, -val_loss)


if __name__ == "__main__":
    main()
