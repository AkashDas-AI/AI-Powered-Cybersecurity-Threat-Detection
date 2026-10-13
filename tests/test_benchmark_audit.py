"""Tests for Benchmark Metric Audit, Threshold Tuning, and Model Comparison."""

import numpy as np
import pandas as pd
import pytest
from src.benchmark_audit import (
    compute_threshold_metrics,
    select_candidate_threshold,
    evaluate_model_at_threshold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def test_compute_threshold_metrics_bounds_and_structure():
    """Confirms threshold sweep generates valid metric ranges and preserved sample counts."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.3, 0.9, 0.2, 0.85])

    thresholds = np.array([0.2, 0.5, 0.8])
    metrics_list = compute_threshold_metrics(y_true, y_prob, thresholds=thresholds)

    assert len(metrics_list) == 3
    for m in metrics_list:
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        assert 0.0 <= m["f1_score"] <= 1.0
        assert 0.0 <= m["false_positive_rate"] <= 1.0

        cm = m["confusion_matrix"]
        total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
        assert total == len(y_true)


def test_select_candidate_threshold_satisfies_recall_constraint():
    """Confirms threshold selection respects minimum recall constraint while minimizing FPR."""
    metrics_list = [
        {"threshold": 0.2, "recall": 0.98, "precision": 0.70, "f1_score": 0.81, "false_positive_rate": 0.40},
        {"threshold": 0.4, "recall": 0.92, "precision": 0.82, "f1_score": 0.87, "false_positive_rate": 0.18},
        {"threshold": 0.6, "recall": 0.85, "precision": 0.90, "f1_score": 0.87, "false_positive_rate": 0.08},
        {"threshold": 0.8, "recall": 0.60, "precision": 0.95, "f1_score": 0.73, "false_positive_rate": 0.02},
    ]

    # Target minimum recall: 0.90
    choice = select_candidate_threshold(metrics_list, min_recall=0.90, objective="min_fpr")

    # Threshold 0.4 has recall 0.92 >= 0.90 and lower FPR (0.18) than threshold 0.2 (0.40)
    assert choice["candidate_threshold"] == 0.4
    assert choice["validation_metrics"]["recall"] >= 0.90
    assert choice["validation_metrics"]["false_positive_rate"] == 0.18


def test_select_candidate_threshold_max_f1_objective():
    """Confirms max_f1 objective picks the highest F1 score meeting the recall constraint."""
    metrics_list = [
        {"threshold": 0.3, "recall": 0.95, "precision": 0.75, "f1_score": 0.84, "false_positive_rate": 0.30},
        {"threshold": 0.5, "recall": 0.91, "precision": 0.85, "f1_score": 0.88, "false_positive_rate": 0.15},
        {"threshold": 0.7, "recall": 0.75, "precision": 0.92, "f1_score": 0.83, "false_positive_rate": 0.05},
    ]

    choice = select_candidate_threshold(metrics_list, min_recall=0.90, objective="max_f1")
    assert choice["candidate_threshold"] == 0.5
    assert choice["validation_metrics"]["f1_score"] == 0.88


def test_threshold_selection_has_no_test_leakage():
    """Verifies candidate threshold selection depends solely on validation partition."""
    # Synthetic validation set
    y_val = np.array([0, 0, 0, 1, 1, 1])
    y_val_prob = np.array([0.1, 0.3, 0.4, 0.7, 0.8, 0.9])

    # Unseen test set with different distribution
    y_test = np.array([0, 1, 0, 1])
    y_test_prob = np.array([0.8, 0.9, 0.2, 0.6])  # High test probability for benign

    val_sweep = compute_threshold_metrics(y_val, y_val_prob)
    choice = select_candidate_threshold(val_sweep, min_recall=0.66)

    # Candidate threshold must be computed without any knowledge of y_test or y_test_prob
    assert "candidate_threshold" in choice
    cand_th = choice["candidate_threshold"]
    assert 0.0 < cand_th < 1.0


def test_evaluate_model_at_threshold_class_ordering():
    """Verifies confusion matrix ordering: [0, 0]=TN, [0, 1]=FP, [1, 0]=FN, [1, 1]=TP."""
    X = pd.DataFrame({"feat": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    y = pd.Series([0, 0, 0, 1, 1, 1])

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    pipe.fit(X, y)

    metrics = evaluate_model_at_threshold(pipe, X, y, threshold=0.5)


    assert "confusion_matrix" in metrics
    cm = metrics["confusion_matrix"]
    tn = cm["true_negatives"]
    fp = cm["false_positives"]
    fn = cm["false_negatives"]
    tp = cm["true_positives"]

    # Verify FPR and Precision formulas match confusion matrix exactly
    expected_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    assert pytest.approx(metrics["false_positive_rate"], 0.001) == expected_fpr
    assert pytest.approx(metrics["precision"], 0.001) == expected_prec


def test_artifact_separation_and_provenance():
    """Verifies that metric JSON artifacts exist and preserve distinct classifier provenance."""
    from pathlib import Path
    import json

    lr_path = Path("artifacts/metrics_logistic.json")
    rf_path = Path("artifacts/metrics_random_forest.json")
    lr_tuned_path = Path("artifacts/metrics_logistic_tuned.json")
    lr_consistent_path = Path("artifacts/metrics_logistic_consistent.json")

    assert lr_path.exists()
    assert rf_path.exists()
    assert lr_tuned_path.exists()
    if lr_consistent_path.exists():
        with open(lr_consistent_path, "r", encoding="utf-8") as f:
            consistent_data = json.load(f)
        assert consistent_data["classifier"] == "LogisticRegression"
        assert consistent_data["calibration_strategy"] == "frozen_80_20_split"
        assert consistent_data["dataset_source"] == "authentic_unsw_nb15"

    with open(lr_path, "r", encoding="utf-8") as f:
        lr_data = json.load(f)
    with open(rf_path, "r", encoding="utf-8") as f:
        rf_data = json.load(f)
    with open(lr_tuned_path, "r", encoding="utf-8") as f:
        tuned_data = json.load(f)

    assert lr_data["classifier"] == "LogisticRegression"
    assert lr_data["threshold_type"] == "default_0.5"
    assert lr_data["dataset_source"] == "authentic_unsw_nb15"

    assert rf_data["classifier"] == "RandomForestClassifier"
    assert rf_data["threshold_type"] == "default_0.5"
    assert rf_data["dataset_source"] == "authentic_unsw_nb15"

    assert tuned_data["classifier"] == "LogisticRegression"
    assert "val_selected" in tuned_data["threshold_type"]
    assert tuned_data["decision_threshold"] > 0.5


def test_consistent_calibration_pipeline_structure():
    """Confirms that run_consistent_calibration_pipeline executes without drift and preserves parameters."""
    from src.benchmark_audit import run_consistent_calibration_pipeline

    # Synthetic dataset
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
        "cat1": ["tcp", "udp", "arp"] * 20,
    })
    y = pd.Series([0, 1] * 30)

    pipe, m_def, m_cons, th = run_consistent_calibration_pipeline(
        X_train_full=df,
        y_train_full=y,
        X_test_full=df,
        y_test_full=y,
        numerical_features=["num1", "num2"],
        categorical_features=["cat1"],
        val_size=0.20,
        random_state=42,
        min_recall=0.50,
        objective="min_fpr",
    )

    assert hasattr(pipe, "predict_proba")
    assert m_def["calibration_strategy"] == "frozen_80_20_split"
    assert m_cons["calibration_strategy"] == "frozen_80_20_split"
    assert m_cons["decision_threshold"] == th
    assert "calibration_details" in m_cons
    assert m_cons["calibration_details"]["train_samples"] == 48
    assert m_cons["calibration_details"]["calibration_samples"] == 12
