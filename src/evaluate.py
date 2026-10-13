"""Evaluation module for Cybersecurity Threat Detection models.

Calculates standard classification metrics, False Positive Rate (FPR),
and confusion matrix, with JSON persistence for reporting and dashboards.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Computes comprehensive cybersecurity detection metrics.

    Args:
        y_true: Ground truth binary labels (0 = Normal, 1 = Attack).
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary of computed performance metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # False Positive Rate: FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": fpr,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "total_samples": int(len(y_true)),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["Normal", "Attack"], output_dict=True, zero_division=0
        ),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["average_precision"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["average_precision"] = None

    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: Path) -> None:
    """Saves metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary of metrics.
        filepath: Target Path object.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_metrics(filepath: Path) -> Dict[str, Any]:
    """Loads metrics dictionary from a JSON file.

    Args:
        filepath: Path to metrics JSON.

    Returns:
        Dictionary of metrics.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file not found at: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
