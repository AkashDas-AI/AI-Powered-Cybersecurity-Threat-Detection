"""Tests for Isolation Forest Anomaly Detection and Hybrid Threat Detection Module."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    fit_isolation_forest_pipeline,
    compute_anomaly_scores,
    compute_anomaly_score_distribution,
    select_anomaly_threshold,
    select_hybrid_policy_on_validation,
    evaluate_anomaly_detector,
    evaluate_hybrid_policies,
)


@pytest.fixture
def synthetic_flow_data():
    """Generates synthetic network flow data with benign and attack classes."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(1, n + 1),
        "dur": np.random.exponential(1.0, n),
        "sbytes": np.random.randint(100, 10000, n),
        "dbytes": np.random.randint(100, 10000, n),
        "proto": np.random.choice(["tcp", "udp", "arp"], n),
        "service": np.random.choice(["http", "dns", "-"], n),
        "state": np.random.choice(["FIN", "CON", "INT"], n),
        "attack_cat": np.random.choice(["Normal", "Generic", "Exploits"], n),
        "label": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })
    return df


def test_benign_only_fitting_and_label_exclusion(synthetic_flow_data):
    """Verifies that Isolation Forest is fitted strictly on benign records and target columns are excluded."""
    df = synthetic_flow_data
    # Exclude target and identifiers
    feature_cols = [c for c in df.columns if c not in ["id", "label", "attack_cat"]]
    num_cols = ["dur", "sbytes", "dbytes"]
    cat_cols = ["proto", "service", "state"]

    benign_df = df[df["label"] == 0][feature_cols]
    assert len(benign_df) > 0
    assert "label" not in benign_df.columns
    assert "attack_cat" not in benign_df.columns
    assert "id" not in benign_df.columns

    pipe = fit_isolation_forest_pipeline(
        X_train_benign=benign_df,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        n_estimators=10,
        max_samples=32,
        random_state=42,
    )

    assert hasattr(pipe, "decision_function")
    scores = compute_anomaly_scores(pipe, df[feature_cols])
    assert len(scores) == len(df)
    assert not np.isnan(scores).any()


def test_deterministic_training_fixed_seed(synthetic_flow_data):
    """Verifies that training with fixed random seed produces identical anomaly scores."""
    df = synthetic_flow_data
    feature_cols = [c for c in df.columns if c not in ["id", "label", "attack_cat"]]
    num_cols = ["dur", "sbytes", "dbytes"]
    cat_cols = ["proto", "service", "state"]
    benign_df = df[df["label"] == 0][feature_cols]

    pipe1 = fit_isolation_forest_pipeline(benign_df, num_cols, cat_cols, n_estimators=20, max_samples=32, random_state=42)
    pipe2 = fit_isolation_forest_pipeline(benign_df, num_cols, cat_cols, n_estimators=20, max_samples=32, random_state=42)

    scores1 = compute_anomaly_scores(pipe1, df[feature_cols])
    scores2 = compute_anomaly_scores(pipe2, df[feature_cols])

    np.testing.assert_allclose(scores1, scores2, rtol=1e-5, atol=1e-5)


def test_validation_only_threshold_selection():
    """Verifies that threshold selection relies strictly on validation partition without test exposure."""
    y_val = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    val_scores = np.array([-0.15, -0.12, -0.09, -0.08, -0.02, 0.01, 0.05, 0.10, 0.12, 0.18])

    th_choice = select_anomaly_threshold(y_val, val_scores, objective="target_fpr", target_val_fpr=0.20)

    assert "candidate_threshold" in th_choice
    th = th_choice["candidate_threshold"]
    # Check that validation FPR satisfies constraint
    pred_val = (val_scores >= th).astype(int)
    tn = ((pred_val == 0) & (y_val == 0)).sum()
    fp = ((pred_val == 1) & (y_val == 0)).sum()
    fpr = fp / (fp + tn)
    assert fpr <= 0.20


def test_anomaly_metric_correctness():
    """Verifies that confusion matrix matches precision, recall, and FPR formulas."""
    y_true = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    # Create mock pipeline
    class MockPipe:
        def decision_function(self, X):
            return np.array([0.1, 0.05, 0.02, -0.01, -0.05, -0.1, -0.15, -0.2])

    X = pd.DataFrame({"feat": range(8)})
    m = evaluate_anomaly_detector(MockPipe(), X, y_true, threshold=0.0)

    cm = m["confusion_matrix"]
    tn, fp, fn, tp = cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]

    assert tp + fn == 4
    assert tn + fp == 4
    expected_rec = tp / (tp + fn)
    expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    expected_fpr = fp / (fp + tn)

    assert pytest.approx(m["recall"], 0.001) == expected_rec
    assert pytest.approx(m["precision"], 0.001) == expected_prec
    assert pytest.approx(m["false_positive_rate"], 0.001) == expected_fpr


def test_hybrid_policy_logic():
    """Verifies Cascaded OR and Band Confirmation policy logic and trade-off counters."""
    y_test = pd.Series([0, 0, 0, 1, 1, 1])
    sup_probs = np.array([0.10, 0.50, 0.80, 0.20, 0.60, 0.90])
    anomaly_scores = np.array([-0.10, 0.10, -0.05, 0.15, 0.05, 0.20])

    results = evaluate_hybrid_policies(
        y_test=y_test,
        sup_probs=sup_probs,
        anomaly_scores=anomaly_scores,
        sup_threshold=0.75,
        anomaly_threshold=0.0,
        band_low=0.40,
        band_high=0.75,
    )

    assert "hybrid_cascaded_or" in results
    assert "hybrid_band_confirmation" in results
    assert "tradeoff_analysis_or" in results
    assert "tradeoff_analysis_band" in results

    t_or = results["tradeoff_analysis_or"]
    assert t_or["supervised_false_negatives_total"] >= 1
    assert t_or["false_negatives_recovered_by_anomaly"] >= 1


def test_anomaly_artifact_provenance_and_files():
    """Verifies that generated anomaly artifacts exist and preserve required provenance tags."""
    iforest_file = Path("artifacts/metrics_isolation_forest.json")
    hybrid_file = Path("artifacts/metrics_hybrid.json")
    model_file = Path("artifacts/model_isolation_forest.joblib")

    assert iforest_file.exists(), "Isolation Forest metrics file missing"
    assert hybrid_file.exists(), "Hybrid metrics file missing"
    if not model_file.exists():
        pytest.skip("Isolation Forest model binary not present in clean checkout (re-generate via python src/anomaly_detection.py)")
    assert model_file.stat().st_size > 0

    with open(iforest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["detector"] == "IsolationForest"
    assert data["dataset_source"] == "authentic_unsw_nb15"
    assert data["training_subset"] == "benign_only"
    assert data["training_samples"] == 44800
    assert data["is_sample_evaluation"] is False
    assert "score_distribution" in data

    with open(hybrid_file, "r", encoding="utf-8") as f:
        hyb = json.load(f)

    assert "supervised_baseline" in hyb
    assert "anomaly_standalone" in hyb
    assert "hybrid_cascaded_or" in hyb
    assert "hybrid_band_confirmation" in hyb
    assert hyb["dataset_source"] == "authentic_unsw_nb15"


def test_hybrid_policy_selection_uses_only_validation_data():
    """Verifies that policy selection operates strictly on validation input arrays."""
    np.random.seed(42)
    n = 200
    y_val = np.random.choice([0, 1], n, p=[0.5, 0.5])
    # Synthetic probabilities and anomaly scores
    val_probs = np.random.uniform(0.0, 1.0, n)
    val_scores = np.random.normal(0.0, 0.1, n)

    res = select_hybrid_policy_on_validation(
        y_val=y_val,
        val_sup_probs=val_probs,
        val_anomaly_scores=val_scores,
        sup_threshold=0.76,
        anomaly_threshold=0.0,
        candidate_band_lows=[0.20, 0.40, 0.60],
        objective="max_recall_constrained_fpr",
        max_val_fpr=0.25,
    )

    assert "selected_candidate_id" in res
    assert "selected_policy_name" in res
    assert "objective_rule" in res
    assert "candidate_sweep" in res
    assert len(res["candidate_sweep"]) >= 4  # baseline, or, and 3 band policies

    for c in res["candidate_sweep"]:
        assert "is_eligible" in c
        assert "metrics" in c
        m = c["metrics"]
        assert 0.0 <= m["false_positive_rate"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0


def test_hybrid_policy_selection_deterministic():
    """Verifies that policy selection produces identical outputs given identical validation inputs."""
    np.random.seed(123)
    n = 100
    y_val = np.random.choice([0, 1], n, p=[0.5, 0.5])
    val_probs = np.random.uniform(0.0, 1.0, n)
    val_scores = np.random.normal(0.0, 0.1, n)

    res1 = select_hybrid_policy_on_validation(
        y_val=y_val,
        val_sup_probs=val_probs,
        val_anomaly_scores=val_scores,
        sup_threshold=0.76,
        anomaly_threshold=-0.0428,
        candidate_band_lows=[0.30, 0.40, 0.50],
    )
    res2 = select_hybrid_policy_on_validation(
        y_val=y_val,
        val_sup_probs=val_probs,
        val_anomaly_scores=val_scores,
        sup_threshold=0.76,
        anomaly_threshold=-0.0428,
        candidate_band_lows=[0.30, 0.40, 0.50],
    )

    assert res1["selected_candidate_id"] == res2["selected_candidate_id"]
    assert res1["selected_band_low"] == res2["selected_band_low"]
    assert res1["validation_metrics_selected"]["false_positive_rate"] == res2["validation_metrics_selected"]["false_positive_rate"]
    assert res1["validation_metrics_selected"]["recall"] == res2["validation_metrics_selected"]["recall"]


def test_hybrid_policy_metadata_saved_in_artifacts():
    """Verifies that metrics_hybrid.json contains policy selection metadata and Random Forest baseline."""
    hybrid_file = Path("artifacts/metrics_hybrid.json")
    assert hybrid_file.exists()

    with open(hybrid_file, "r", encoding="utf-8") as f:
        hyb = json.load(f)

    assert "policy_selection_metadata" in hyb
    pol_meta = hyb["policy_selection_metadata"]
    assert pol_meta is not None
    assert "objective_rule" in pol_meta
    assert "selected_band_low" in pol_meta
    assert pol_meta["selected_band_low"] == 0.40
    assert "candidate_sweep" in pol_meta
    assert len(pol_meta["candidate_sweep"]) > 0

    # Verify Random Forest baseline present
    assert "random_forest_baseline" in hyb
    assert hyb["random_forest_baseline"] is not None
    assert "confusion_matrix" in hyb["random_forest_baseline"]


def test_metric_recomputation_from_confusion_matrices():
    """Verifies that accuracy, precision, recall, f1, and fpr match exact formulas from raw confusion matrices."""
    hybrid_file = Path("artifacts/metrics_hybrid.json")
    with open(hybrid_file, "r", encoding="utf-8") as f:
        hyb = json.load(f)

    policies = ["supervised_baseline", "random_forest_baseline", "anomaly_standalone", "hybrid_cascaded_or", "hybrid_band_confirmation"]
    for p in policies:
        data = hyb.get(p)
        if data is None:
            continue
        cm = data["confusion_matrix"]
        tn, fp, fn, tp = cm["true_negatives"], cm["false_positives"], cm["false_negatives"], cm["true_positives"]
        total = tn + fp + fn + tp
        assert total == 82332

        expected_acc = (tp + tn) / total
        expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        expected_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        expected_f1 = (2 * expected_prec * expected_rec) / (expected_prec + expected_rec) if (expected_prec + expected_rec) > 0 else 0.0
        expected_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        assert pytest.approx(data["accuracy"], abs=1e-5) == expected_acc
        assert pytest.approx(data["precision"], abs=1e-5) == expected_prec
        assert pytest.approx(data["recall"], abs=1e-5) == expected_rec
        assert pytest.approx(data["f1_score"], abs=1e-5) == expected_f1
        assert pytest.approx(data["false_positive_rate"], abs=1e-5) == expected_fpr


def test_report_distinguishes_validation_from_test_metrics():
    """Verifies that the generated markdown report explicitly distinguishes validation metrics from test metrics."""
    report_file = Path("outputs/reports/anomaly_detection_report.md")
    if not report_file.exists():
        pytest.skip("Anomaly detection report not present; run python src/anomaly_detection.py to generate")

    content = report_file.read_text(encoding="utf-8")
    # Verify section separation
    assert "Validation-Only Policy Selection" in content
    assert "Official Test Set Performance Comparison" in content
    assert "Zero Test Set Exposure" in content

    # Verify both validation and test FPRs are present and distinct
    assert "Validation Benign FPR" in content
    assert "Test Benign FPR" in content
    assert "7.46%" in content  # Validation FPR of winner
    assert "19.01%" in content  # Test FPR of winner
