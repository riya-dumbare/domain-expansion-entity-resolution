"""Threshold tuning and evaluation using F0.5 score.

In entity resolution competitions where precision is prioritized over recall,
F0.5 weights precision twice as heavily as recall:
    F0.5 = (1 + 0.5^2) * (P * R) / (0.5^2 * P + R)

This module provides:
- Grid search for optimal probability threshold on validation set
- Pair-level and Entity-level F0.5 / Precision / Recall metrics
- Threshold curve reporting and evaluation tables
"""

from typing import Dict, List, Set, Tuple

import numpy as np


def compute_f_beta(
    precision: float,
    recall: float,
    beta: float = 0.5
) -> float:
    """Compute F-beta score from precision and recall."""

    if precision <= 0.0 or recall <= 0.0:
        return 0.0

    beta_sq = beta ** 2

    numerator = (1 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    if denominator <= 0.0:
        return 0.0

    return float(numerator / denominator)


def evaluate_pair_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    beta: float = 0.5
) -> Dict[str, float]:
    """Evaluate binary classification metrics at a given probability threshold."""

    # LightGBM predict_proba() returns:
    # column 0 = probability of class 0
    # column 1 = probability of class 1
    #
    # We need only class 1 probability.
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    y_prob = np.asarray(y_prob).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)

    if len(y_prob) != len(y_true):
        raise ValueError(
            f"Shape mismatch: y_prob has {len(y_prob)} values "
            f"but y_true has {len(y_true)} values."
        )

    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = (
        float(tp / (tp + fp))
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        float(tp / (tp + fn))
        if (tp + fn) > 0
        else 0.0
    )

    f_beta = compute_f_beta(
        precision,
        recall,
        beta=beta
    )

    f1 = compute_f_beta(
        precision,
        recall,
        beta=1.0
    )

    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        f"f{beta}": round(f_beta, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 0.5,
    min_threshold: float = 0.10,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """Search for probability threshold that maximizes F-beta."""

    thresholds = np.arange(
        min_threshold,
        max_threshold + step / 2,
        step
    )

    best_thresh = 0.5
    best_score = -1.0

    history: List[Dict[str, float]] = []

    for t in thresholds:

        metrics = evaluate_pair_predictions(
            y_true,
            y_prob,
            threshold=float(t),
            beta=beta
        )

        score = metrics[f"f{beta}"]

        history.append(metrics)

        if score > best_score:
            best_score = score
            best_thresh = float(t)

    return best_thresh, best_score, history


def evaluate_entity_level(
    ground_truth: Dict[str, Set[str]],
    predicted_matches: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Dict[str, float]:
    """Compute entity-level Precision, Recall, and F-beta."""

    total_tp = 0
    total_fp = 0
    total_fn = 0

    all_s1 = set(
        ground_truth.keys()
    ) | set(
        predicted_matches.keys()
    )

    for s1 in all_s1:

        true_set = ground_truth.get(
            s1,
            set()
        )

        pred_set = predicted_matches.get(
            s1,
            set()
        )

        tp = len(
            true_set & pred_set
        )

        fp = len(
            pred_set - true_set
        )

        fn = len(
            true_set - pred_set
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = (
        float(total_tp / (total_tp + total_fp))
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        float(total_tp / (total_tp + total_fn))
        if (total_tp + total_fn) > 0
        else 0.0
    )

    f_beta = compute_f_beta(
        precision,
        recall,
        beta=beta
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        f"f{beta}": round(f_beta, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }