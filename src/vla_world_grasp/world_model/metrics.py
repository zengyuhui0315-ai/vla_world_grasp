"""Metrics for binary grasp success scoring."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def confusion_matrix_counts(y_true: Sequence[float], y_pred: Sequence[float]) -> List[List[int]]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    return [[tn, fp], [fn, tp]]


def binary_auc(y_true: Sequence[float], y_score: Sequence[float]) -> Optional[float]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    positives = y_true_arr == 1
    negatives = y_true_arr == 0
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(y_score_arr)
    sorted_scores = y_score_arr[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[start:end] = avg_rank
        start = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    rank_sum_pos = float(np.sum(original_ranks[positives]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def class_counts(y_true: Sequence[float]) -> Dict[str, int]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    positives = int(np.sum(y_true_arr == 1))
    negatives = int(np.sum(y_true_arr == 0))
    return {"positive": positives, "negative": negatives}


def binary_classification_metrics(
    y_true: Sequence[float],
    y_score: Sequence[float],
    threshold: float = 0.5,
    warn_prefix: str = "",
    compute_auc: bool = True,
) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    y_pred_arr = (y_score_arr >= threshold).astype(np.int64)
    matrix = confusion_matrix_counts(y_true_arr, y_pred_arr)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = max(int(len(y_true_arr)), 1)
    accuracy = float((tp + tn) / total)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    auc = binary_auc(y_true_arr, y_score_arr) if compute_auc else None
    if auc is None:
        if compute_auc:
            prefix = f"{warn_prefix}: " if warn_prefix else ""
            print(f"WARNING: {prefix}AUC skipped because only one class is present.")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": matrix,
        "threshold": threshold,
        "num_samples": int(len(y_true_arr)),
    }


def best_threshold_by_f1(
    y_true: Sequence[float],
    y_score: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> tuple[float, Dict[str, object]]:
    values = thresholds if thresholds is not None else [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    best_threshold = 0.5
    best_metrics: Dict[str, object] | None = None
    for threshold in values:
        metrics = binary_classification_metrics(y_true, y_score, threshold=float(threshold), warn_prefix="", compute_auc=False)
        if best_metrics is None:
            best_threshold = float(threshold)
            best_metrics = metrics
            continue
        f1 = float(metrics["f1"])
        acc = float(metrics["accuracy"])
        best_f1 = float(best_metrics["f1"])
        best_acc = float(best_metrics["accuracy"])
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        best_metrics = binary_classification_metrics(y_true, y_score, threshold=0.5)
    return best_threshold, best_metrics


def roc_curve_points(y_true: Sequence[float], y_score: Sequence[float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true_arr)) < 2:
        return None
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score_arr))[::-1], -np.inf]
    fpr_values = []
    tpr_values = []
    for threshold in thresholds:
        y_pred = (y_score_arr >= threshold).astype(np.int64)
        matrix = confusion_matrix_counts(y_true_arr, y_pred)
        tn, fp = matrix[0]
        fn, tp = matrix[1]
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr_values.append(fpr)
        tpr_values.append(tpr)
    return np.asarray(fpr_values), np.asarray(tpr_values)
