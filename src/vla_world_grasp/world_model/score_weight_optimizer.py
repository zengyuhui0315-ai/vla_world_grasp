"""Random-search optimizer for interpretable grasp score weights."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vla_world_grasp.world_model.dataset import parse_bool
from vla_world_grasp.world_model.metrics import binary_auc, binary_classification_metrics, class_counts


SCORE_TERMS: tuple[str, ...] = (
    "center_score",
    "reachability_score",
    "collision_score",
    "height_score",
    "gripper_width_score",
    "yaw_score",
)

SHORT_TERM_LABELS: tuple[str, ...] = (
    "center",
    "reachability",
    "collision",
    "height",
    "width",
    "yaw",
)

ORIGINAL_WEIGHTS: dict[str, float] = {
    "center_score": 0.30,
    "reachability_score": 0.20,
    "collision_score": 0.20,
    "height_score": 0.15,
    "gripper_width_score": 0.10,
    "yaw_score": 0.05,
}


@dataclass(frozen=True)
class WeightOptimizationDataset:
    x: np.ndarray
    y: np.ndarray
    rows: list[dict[str, str]]
    baseline_score: np.ndarray | None


def optimize_score_weights(
    dataset_csv: str | Path = "data/grasp_score_quick/processed/grasp_success_dataset.csv",
    num_trials: int = 5000,
    selection_metric: str = "f1",
    output_dir: str | Path = "outputs",
    split_by_episode: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    output_paths = ensure_output_dirs(output_dir)
    dataset = load_weight_optimization_dataset(dataset_csv)
    splits = split_indices(dataset.rows, split_by_episode=split_by_episode, seed=seed)
    _warn_for_single_class_splits(dataset.y, splits)

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    split_counts = {
        name: class_counts(dataset.y[indices])
        for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx))
    }

    original_weight_array = weights_dict_to_array(ORIGINAL_WEIGHTS)
    original_val_score = score_candidates(dataset.x[val_idx], original_weight_array)
    original_threshold, _ = best_threshold_for_scores(dataset.y[val_idx], original_val_score)

    optimized_weight_array, optimized_threshold, best_val_metrics = random_search_weights(
        x_val=dataset.x[val_idx],
        y_val=dataset.y[val_idx],
        num_trials=num_trials,
        selection_metric=selection_metric,
        seed=seed,
    )

    original_scores = score_candidates(dataset.x[test_idx], original_weight_array)
    optimized_scores = score_candidates(dataset.x[test_idx], optimized_weight_array)
    original_metrics = evaluate_named_scores(dataset.y[test_idx], original_scores, original_threshold, "original_score")
    optimized_metrics = evaluate_named_scores(dataset.y[test_idx], optimized_scores, optimized_threshold, "optimized_score")
    baseline_metrics = None
    if dataset.baseline_score is not None:
        baseline_val_scores = dataset.baseline_score[val_idx]
        baseline_threshold, _ = best_threshold_for_scores(dataset.y[val_idx], baseline_val_scores)
        baseline_metrics = evaluate_named_scores(
            dataset.y[test_idx],
            dataset.baseline_score[test_idx],
            baseline_threshold,
            "baseline_score",
        )

    payload: dict[str, Any] = {
        "method": "random_search",
        "num_trials": int(num_trials),
        "selection_metric": selection_metric,
        "dataset_csv": str(dataset_csv),
        "num_samples_used": int(len(dataset.y)),
        "score_terms": list(SCORE_TERMS),
        "original_weights": dict(ORIGINAL_WEIGHTS),
        "optimized_weights": weights_array_to_dict(optimized_weight_array),
        "best_threshold": float(optimized_threshold),
        "train_positive_count": split_counts["train"]["positive"],
        "train_negative_count": split_counts["train"]["negative"],
        "val_positive_count": split_counts["val"]["positive"],
        "val_negative_count": split_counts["val"]["negative"],
        "test_positive_count": split_counts["test"]["positive"],
        "test_negative_count": split_counts["test"]["negative"],
        "original_metrics": original_metrics,
        "optimized_metrics": optimized_metrics,
        "baseline_metrics": baseline_metrics,
        "split_by_episode": bool(split_by_episode),
        "train_episode_count": count_episodes(dataset.rows, train_idx),
        "val_episode_count": count_episodes(dataset.rows, val_idx),
        "test_episode_count": count_episodes(dataset.rows, test_idx),
        "val_selection_metrics": best_val_metrics,
    }

    json_path = output_paths["logs"] / "score_weight_optimization.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_weight_comparison(payload["original_weights"], payload["optimized_weights"], output_paths["figures"] / "score_weights_original_vs_optimized.png")
    plot_metrics_comparison(
        original_metrics=original_metrics,
        optimized_metrics=optimized_metrics,
        baseline_metrics=baseline_metrics,
        output_path=output_paths["figures"] / "score_weight_metrics_comparison.png",
    )
    write_markdown_report(
        payload,
        report_path=Path("reports") / "score_weight_optimization.md",
        weights_figure=output_paths["figures"] / "score_weights_original_vs_optimized.png",
        metrics_figure=output_paths["figures"] / "score_weight_metrics_comparison.png",
        json_path=json_path,
    )
    return payload


def load_weight_optimization_dataset(dataset_csv: str | Path) -> WeightOptimizationDataset:
    path = Path(dataset_csv)
    if not path.exists():
        raise FileNotFoundError(f"dataset csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise ValueError(f"dataset csv is empty: {path}")

    has_label_available = "label_available" in rows[0]
    filtered: list[dict[str, str]] = []
    for row in rows:
        success = parse_bool(row.get("success", ""))
        if success is None:
            continue
        if has_label_available:
            if parse_bool(row.get("label_available", "")) is not True:
                continue
            if parse_bool(row.get("executed", "")) is not True:
                continue
        elif parse_bool(row.get("selected", "")) is not True:
            continue
        filtered.append(row)

    if not filtered:
        raise ValueError("no usable success/failure samples after filtering")

    x = np.asarray([[safe_float(row.get(term, "")) for term in SCORE_TERMS] for row in filtered], dtype=np.float64)
    y = np.asarray([1.0 if parse_bool(row.get("success", "")) else 0.0 for row in filtered], dtype=np.float64)
    baseline_score = None
    if "baseline_score" in rows[0]:
        baseline_score = np.asarray([safe_float(row.get("baseline_score", "")) for row in filtered], dtype=np.float64)
    return WeightOptimizationDataset(x=x, y=y, rows=filtered, baseline_score=baseline_score)


def random_search_weights(
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_trials: int,
    selection_metric: str,
    seed: int = 7,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if len(y_val) == 0:
        print("WARNING: validation split is empty; using original weights.")
        original = weights_dict_to_array(ORIGINAL_WEIGHTS)
        return original, 0.5, evaluate_named_scores(y_val, np.asarray([]), 0.5, "optimized_score")

    rng = np.random.default_rng(seed)
    best_weights = weights_dict_to_array(ORIGINAL_WEIGHTS)
    best_threshold, best_metrics = best_threshold_for_scores(y_val, score_candidates(x_val, best_weights))
    best_key = selection_key(best_metrics, selection_metric)

    for _ in range(max(1, int(num_trials))):
        weights = rng.random(len(SCORE_TERMS))
        total = float(np.sum(weights))
        weights = weights / total if total > 0 else np.full(len(SCORE_TERMS), 1.0 / len(SCORE_TERMS))
        scores = score_candidates(x_val, weights)
        threshold, metrics = best_threshold_for_scores(y_val, scores)
        key = selection_key(metrics, selection_metric)
        if key > best_key:
            best_weights = weights
            best_threshold = threshold
            best_metrics = metrics
            best_key = key

    return best_weights, float(best_threshold), best_metrics


def best_threshold_for_scores(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, Any]]:
    if len(y_true) == 0:
        return 0.5, empty_metrics(0.5)
    thresholds = sorted({0.0, 1.0, 0.5, *[float(value) for value in y_score]})
    best_threshold = 0.5
    best_metrics: dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = evaluate_named_scores(y_true, y_score, threshold, "threshold_search", compute_auc=False)
        if best_metrics is None or (float(metrics["f1"]), float(metrics["accuracy"])) > (
            float(best_metrics["f1"]),
            float(best_metrics["accuracy"]),
        ):
            best_threshold = float(threshold)
            best_metrics = metrics
    assert best_metrics is not None
    best_metrics["auc"] = binary_auc(y_true, y_score)
    return best_threshold, best_metrics


def evaluate_named_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    name: str,
    compute_auc: bool = True,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return empty_metrics(threshold)
    metrics = binary_classification_metrics(
        y_true,
        y_score,
        threshold=threshold,
        warn_prefix=name,
        compute_auc=compute_auc,
    )
    metrics["best_threshold"] = float(threshold)
    return metrics


def split_indices(rows: Sequence[dict[str, str]], split_by_episode: bool, seed: int = 7) -> dict[str, np.ndarray]:
    if split_by_episode:
        return split_indices_by_episode(rows, seed=seed)
    return split_indices_by_row(len(rows), seed=seed)


def split_indices_by_row(num_samples: int, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    train_count, val_count = split_counts(num_samples)
    return {
        "train": np.asarray(sorted(indices[:train_count]), dtype=np.int64),
        "val": np.asarray(sorted(indices[train_count : train_count + val_count]), dtype=np.int64),
        "test": np.asarray(sorted(indices[train_count + val_count :]), dtype=np.int64),
    }


def split_indices_by_episode(rows: Sequence[dict[str, str]], seed: int = 7) -> dict[str, np.ndarray]:
    episode_to_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        episode_id = row.get("episode_id", "").strip() or f"__missing_episode_{index}"
        episode_to_indices.setdefault(episode_id, []).append(index)
    episode_ids = sorted(episode_to_indices)
    if len(episode_ids) < 2:
        print("WARNING: fewer than 2 episodes; falling back to row split so train and val can exist.")
        return split_indices_by_row(len(rows), seed=seed)

    rng = np.random.default_rng(seed)
    shuffled = [episode_ids[index] for index in rng.permutation(len(episode_ids))]
    train_count, val_count = split_counts(len(shuffled))
    split_episode_ids = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }
    result: dict[str, np.ndarray] = {}
    for split_name, ids in split_episode_ids.items():
        indices: list[int] = []
        for episode_id in ids:
            indices.extend(episode_to_indices[episode_id])
        result[split_name] = np.asarray(sorted(indices), dtype=np.int64)
    return result


def split_counts(total: int) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    if total == 1:
        return 1, 0
    if total == 2:
        return 1, 1
    train_count = int(round(total * 0.70))
    val_count = int(round(total * 0.15))
    train_count = max(1, min(train_count, total - 2))
    val_count = max(1, min(val_count, total - train_count - 1))
    return train_count, val_count


def plot_weight_comparison(original_weights: dict[str, float], optimized_weights: dict[str, float], output_path: Path) -> None:
    x = np.arange(len(SCORE_TERMS))
    width = 0.38
    plt.figure(figsize=(8.5, 4.8))
    plt.bar(x - width / 2, [original_weights[term] for term in SCORE_TERMS], width, label="original", color="#4C78A8")
    plt.bar(x + width / 2, [optimized_weights[term] for term in SCORE_TERMS], width, label="optimized", color="#F58518")
    plt.xticks(x, SHORT_TERM_LABELS, rotation=20, ha="right")
    plt.ylabel("weight")
    plt.ylim(0, max(0.45, max(optimized_weights.values()) + 0.08))
    plt.title("Original vs Optimized Score Weights")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_metrics_comparison(
    original_metrics: dict[str, Any],
    optimized_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    output_path: Path,
) -> None:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    if any(metrics.get("auc") is not None for metrics in (original_metrics, optimized_metrics, baseline_metrics or {})):
        metric_names.append("auc")
    series = [
        ("original", original_metrics, "#4C78A8"),
        ("optimized", optimized_metrics, "#F58518"),
    ]
    if baseline_metrics is not None:
        series.append(("baseline", baseline_metrics, "#54A24B"))

    x = np.arange(len(metric_names))
    bar_width = 0.8 / len(series)
    plt.figure(figsize=(8.5, 4.8))
    for idx, (name, metrics, color) in enumerate(series):
        offset = (idx - (len(series) - 1) / 2) * bar_width
        values = [0.0 if metrics.get(metric) is None else float(metrics.get(metric, 0.0)) for metric in metric_names]
        plt.bar(x + offset, values, bar_width, label=name, color=color)
    plt.xticks(x, metric_names)
    plt.ylabel("metric")
    plt.ylim(0, 1.05)
    plt.title("Original vs Optimized Score Metrics")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_markdown_report(
    payload: dict[str, Any],
    report_path: Path,
    weights_figure: Path,
    metrics_figure: Path,
    json_path: Path,
) -> None:
    original_metrics = payload["original_metrics"]
    optimized_metrics = payload["optimized_metrics"]
    baseline_metrics = payload.get("baseline_metrics")
    is_synthetic = "synthetic" in str(payload["dataset_csv"]).lower()
    purpose_text = (
        "使用明确标注的 synthetic sanity check 数据，对原始启发式评分权重进行 random search 优化，"
        "用于验证优化器在存在可学习权重信号时能够找到更优的可解释权重。"
        if is_synthetic
        else "使用 Isaac Sim 采集的 candidate-level success/failure 标签，对原始启发式评分权重进行 random search 优化，"
        "保持评分函数可解释，同时让候选动作排序更贴近真实抓取结果。"
    )
    ppt_text = (
        "通过 synthetic sanity check 数据验证 random search 权重优化流程：在保持评分函数可解释性的同时，优化器能够从 success/failure 标签中恢复更合适的评分权重，并提升候选动作成功预测指标。"
        if is_synthetic
        else "通过 Isaac Sim 采集的候选动作 success/failure 数据，对原始启发式评分权重进行 random search 优化。在保持评分函数可解释性的同时，使候选动作排序更贴近实际抓取结果。"
    )
    lines = [
        "# Score Weight Optimization",
        "",
        "## 优化目的",
        purpose_text,
        "",
        "## 数据集",
        f"- 数据集路径: `{payload['dataset_csv']}`",
        f"- 使用样本数量: {payload['num_samples_used']}",
        f"- 搜索方法: {payload['method']}",
        f"- 搜索次数: {payload['num_trials']}",
        f"- 选择指标: {payload['selection_metric']}",
        "",
        "## 原始权重",
        weights_table(payload["original_weights"]),
        "",
        "## 优化权重",
        weights_table(payload["optimized_weights"]),
        f"- best_threshold: {payload['best_threshold']:.4f}",
        "",
        "## Train / Val / Test 正负样本数量",
        "| split | positive | negative | episodes |",
        "|---|---:|---:|---:|",
        f"| train | {payload['train_positive_count']} | {payload['train_negative_count']} | {payload['train_episode_count']} |",
        f"| val | {payload['val_positive_count']} | {payload['val_negative_count']} | {payload['val_episode_count']} |",
        f"| test | {payload['test_positive_count']} | {payload['test_negative_count']} | {payload['test_episode_count']} |",
        "",
        "## Original vs Optimized 指标对比",
        metrics_table(original_metrics, optimized_metrics, baseline_metrics),
        "",
        "## 生成图表路径",
        f"- JSON: `{json_path}`",
        f"- 权重对比图: `{weights_figure}`",
        f"- 指标对比图: `{metrics_figure}`",
        "",
        "## PPT 总结话术",
        ppt_text,
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def weights_table(weights: dict[str, float]) -> str:
    lines = ["| term | weight |", "|---|---:|"]
    for term in SCORE_TERMS:
        lines.append(f"| {term} | {float(weights[term]):.4f} |")
    return "\n".join(lines)


def metrics_table(
    original_metrics: dict[str, Any],
    optimized_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
) -> str:
    rows = [
        ("original_score", original_metrics),
        ("optimized_score", optimized_metrics),
    ]
    if baseline_metrics is not None:
        rows.append(("baseline_score", baseline_metrics))
    lines = ["| score | accuracy | precision | recall | f1 | auc | threshold |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, metrics in rows:
        auc = metrics.get("auc")
        auc_text = "null" if auc is None else f"{float(auc):.4f}"
        lines.append(
            f"| {name} | {float(metrics.get('accuracy', 0.0)):.4f} | "
            f"{float(metrics.get('precision', 0.0)):.4f} | "
            f"{float(metrics.get('recall', 0.0)):.4f} | "
            f"{float(metrics.get('f1', 0.0)):.4f} | "
            f"{auc_text} | {float(metrics.get('best_threshold', metrics.get('threshold', 0.5))):.4f} |"
        )
    return "\n".join(lines)


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    paths = {"root": root, "logs": root / "logs", "figures": root / "figures"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def score_candidates(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.asarray([], dtype=np.float64)
    return np.asarray(x @ weights, dtype=np.float64)


def selection_key(metrics: dict[str, Any], selection_metric: str) -> tuple[float, float]:
    primary_name = selection_metric if selection_metric in {"accuracy", "precision", "recall", "f1", "auc"} else "f1"
    primary = metrics.get(primary_name)
    primary_value = -1.0 if primary is None else float(primary)
    return primary_value, float(metrics.get("accuracy", 0.0))


def weights_dict_to_array(weights: dict[str, float]) -> np.ndarray:
    return np.asarray([float(weights[term]) for term in SCORE_TERMS], dtype=np.float64)


def weights_array_to_dict(weights: np.ndarray) -> dict[str, float]:
    clipped = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(clipped))
    normalized = clipped / total if total > 0 else np.full(len(SCORE_TERMS), 1.0 / len(SCORE_TERMS))
    return {term: float(value) for term, value in zip(SCORE_TERMS, normalized)}


def safe_float(value: object) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else 0.0
    except (TypeError, ValueError):
        return 0.0


def count_episodes(rows: Sequence[dict[str, str]], indices: np.ndarray) -> int:
    return len({rows[int(index)].get("episode_id", "").strip() or f"__missing_episode_{int(index)}" for index in indices})


def _warn_for_single_class_splits(y: np.ndarray, splits: dict[str, np.ndarray]) -> None:
    for split_name, indices in splits.items():
        if len(indices) == 0:
            print(f"WARNING: {split_name} split is empty.")
            continue
        values = set(int(v) for v in y[indices])
        if len(values) < 2:
            print(f"WARNING: {split_name} split contains a single class: {sorted(values)}")


def empty_metrics(threshold: float) -> dict[str, Any]:
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": None,
        "confusion_matrix": [[0, 0], [0, 0]],
        "threshold": float(threshold),
        "best_threshold": float(threshold),
        "num_samples": 0,
    }
