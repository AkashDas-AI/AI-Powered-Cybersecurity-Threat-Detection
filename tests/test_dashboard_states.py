"""Tests for dashboard state handling and artifact loading."""

import json
from pathlib import Path
import app


def test_load_metrics_data_valid(monkeypatch, tmp_path):
    metrics_file = tmp_path / "metrics.json"
    dummy_data = {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.90,
        "f1_score": 0.84,
        "false_positive_rate": 0.05,
        "confusion_matrix": {
            "true_negatives": 95,
            "false_positives": 5,
            "false_negatives": 10,
            "true_positives": 90,
        },
        "total_samples": 200,
    }
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    monkeypatch.setattr(app, "METRICS_PATH", metrics_file)
    loaded = app.load_metrics_data()
    assert isinstance(loaded, dict)
    assert loaded["accuracy"] == 0.85


def test_load_metrics_data_missing(monkeypatch, tmp_path):
    missing_file = tmp_path / "non_existent_metrics.json"
    monkeypatch.setattr(app, "METRICS_PATH", missing_file)
    loaded = app.load_metrics_data()
    assert loaded is None


def test_load_metrics_data_corrupted(monkeypatch, tmp_path):
    corrupt_file = tmp_path / "corrupted_metrics.json"
    corrupt_file.write_text("{invalid_json: true", encoding="utf-8")
    monkeypatch.setattr(app, "METRICS_PATH", corrupt_file)
    loaded = app.load_metrics_data()
    assert loaded == "CORRUPTED"


def test_load_alerts_data_valid(monkeypatch, tmp_path):
    alerts_file = tmp_path / "alerts.csv"
    alerts_file.write_text("alert_id,severity,rule_name\nALT-001,HIGH,Brute-Force\n", encoding="utf-8")
    monkeypatch.setattr(app, "ALERTS_PATH", alerts_file)
    df = app.load_alerts_data()
    assert df is not None
    assert len(df) == 1
    assert df.iloc[0]["alert_id"] == "ALT-001"


def test_load_alerts_data_missing(monkeypatch, tmp_path):
    missing_file = tmp_path / "non_existent_alerts.csv"
    monkeypatch.setattr(app, "ALERTS_PATH", missing_file)
    df = app.load_alerts_data()
    assert df is None


def test_load_all_benchmark_metrics_valid(monkeypatch, tmp_path):
    lr_file = tmp_path / "metrics_logistic.json"
    rf_file = tmp_path / "metrics_rf.json"

    dummy_lr = {"classifier": "LogisticRegression", "accuracy": 0.81, "dataset_source": "authentic_unsw_nb15", "decision_threshold": 0.5}
    dummy_rf = {"classifier": "RandomForestClassifier", "accuracy": 0.85, "dataset_source": "authentic_unsw_nb15", "decision_threshold": 0.5}

    lr_file.write_text(json.dumps(dummy_lr), encoding="utf-8")
    rf_file.write_text(json.dumps(dummy_rf), encoding="utf-8")

    monkeypatch.setattr(app, "METRICS_LOGISTIC_PATH", lr_file)
    monkeypatch.setattr(app, "METRICS_RF_PATH", rf_file)
    monkeypatch.setattr(app, "METRICS_LOGISTIC_TUNED_PATH", tmp_path / "none1.json")
    monkeypatch.setattr(app, "METRICS_LOGISTIC_OOF_PATH", tmp_path / "none2.json")
    monkeypatch.setattr(app, "METRICS_LOGISTIC_CONSISTENT_PATH", tmp_path / "none3.json")

    metrics_map = app.load_all_benchmark_metrics()
    assert metrics_map["Logistic Regression (Default th=0.50)"]["accuracy"] == 0.81
    assert metrics_map["Random Forest (Default th=0.50)"]["accuracy"] == 0.85
    assert metrics_map["Logistic Regression (Val-Tuned th=0.76)"] is None
    assert metrics_map["Logistic Regression (OOF-Calibrated th=0.77)"] is None
    assert metrics_map["Logistic Regression (Consistent Frozen th=0.76)"] is None


def test_load_all_benchmark_metrics_with_corrupted_file(monkeypatch, tmp_path):
    """Verifies that corrupted JSON artifacts return 'CORRUPTED' without crashing."""
    corrupt_file = tmp_path / "corrupted_lr.json"
    corrupt_file.write_text("{broken json syntax: true", encoding="utf-8")

    monkeypatch.setattr(app, "METRICS_LOGISTIC_PATH", corrupt_file)
    monkeypatch.setattr(app, "METRICS_RF_PATH", tmp_path / "nonexistent_rf.json")
    monkeypatch.setattr(app, "METRICS_LOGISTIC_TUNED_PATH", tmp_path / "none1.json")
    monkeypatch.setattr(app, "METRICS_LOGISTIC_OOF_PATH", tmp_path / "none2.json")
    monkeypatch.setattr(app, "METRICS_LOGISTIC_CONSISTENT_PATH", tmp_path / "none3.json")

    metrics_map = app.load_all_benchmark_metrics()
    assert metrics_map["Logistic Regression (Default th=0.50)"] == "CORRUPTED"
    assert metrics_map["Random Forest (Default th=0.50)"] is None


def test_load_json_artifact_corrupted_returns_sentinel(tmp_path):
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("invalid json string", encoding="utf-8")
    res = app.load_json_artifact(corrupt_file)
    assert res == "CORRUPTED"


def test_benchmark_metrics_provenance_and_metadata():
    """Confirms live artifacts contain authentic_unsw_nb15 provenance tag."""
    metrics_map = app.load_all_benchmark_metrics()
    for name, m in metrics_map.items():
        if m is not None and isinstance(m, dict):
            assert "dataset_source" in m
            assert m["dataset_source"] == "authentic_unsw_nb15"
            assert m["is_sample_evaluation"] is False


def test_threshold_metadata_present_in_metrics():
    """Confirms decision_threshold and threshold_type are present in calibrated metrics."""
    metrics_map = app.load_all_benchmark_metrics()
    oof_metrics = metrics_map.get("Logistic Regression (OOF-Calibrated th=0.77)")
    if oof_metrics:
        assert "decision_threshold" in oof_metrics
        assert oof_metrics["decision_threshold"] == 0.77
        assert "threshold_type" in oof_metrics
        assert "oof_calibrated" in oof_metrics["threshold_type"]


def test_consistent_model_loaded_and_valid():
    """Confirms that the consistent frozen calibration model artifact is loaded and contains required metadata."""
    metrics_map = app.load_all_benchmark_metrics()
    consistent_m = metrics_map.get("Logistic Regression (Consistent Frozen th=0.76)")
    if consistent_m:
        assert isinstance(consistent_m, dict)
        assert consistent_m["classifier"] == "LogisticRegression"
        assert consistent_m["calibration_strategy"] == "frozen_80_20_split"
        assert consistent_m["decision_threshold"] == 0.76
        assert "calibration_details" in consistent_m
        assert consistent_m["calibration_details"]["train_samples"] == 140272
        assert consistent_m["calibration_details"]["calibration_samples"] == 35069


def test_load_anomaly_metrics_states(monkeypatch, tmp_path):
    """Verifies that load_anomaly_metrics handles valid, missing, and corrupted files."""
    valid_file = tmp_path / "valid_if.json"
    valid_file.write_text(json.dumps({"detector": "IsolationForest", "accuracy": 0.71}), encoding="utf-8")
    monkeypatch.setattr(app, "METRICS_IFOREST_PATH", valid_file)
    assert app.load_anomaly_metrics()["accuracy"] == 0.71

    missing_file = tmp_path / "missing_if.json"
    monkeypatch.setattr(app, "METRICS_IFOREST_PATH", missing_file)
    assert app.load_anomaly_metrics() is None

    corrupted_file = tmp_path / "corrupted_if.json"
    corrupted_file.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(app, "METRICS_IFOREST_PATH", corrupted_file)
    assert app.load_anomaly_metrics() == "CORRUPTED"


def test_load_hybrid_metrics_states(monkeypatch, tmp_path):
    """Verifies that load_hybrid_metrics handles valid, missing, and corrupted files."""
    valid_file = tmp_path / "valid_hyb.json"
    valid_file.write_text(json.dumps({"hybrid_cascaded_or": {"accuracy": 0.80}}), encoding="utf-8")
    monkeypatch.setattr(app, "METRICS_HYBRID_PATH", valid_file)
    assert app.load_hybrid_metrics()["hybrid_cascaded_or"]["accuracy"] == 0.80

    missing_file = tmp_path / "missing_hyb.json"
    monkeypatch.setattr(app, "METRICS_HYBRID_PATH", missing_file)
    assert app.load_hybrid_metrics() is None

    corrupted_file = tmp_path / "corrupted_hyb.json"
    corrupt_file_content = "{broken"
    corrupted_file.write_text(corrupt_file_content, encoding="utf-8")
    monkeypatch.setattr(app, "METRICS_HYBRID_PATH", corrupted_file)
    assert app.load_hybrid_metrics() == "CORRUPTED"


def test_load_model_pipeline_states(tmp_path):
    """Verifies that load_model_pipeline handles missing, corrupted, and valid binaries gracefully."""
    import joblib

    # 1. Missing model binary returns None
    missing_file = tmp_path / "nonexistent_model.joblib"
    assert app.load_model_pipeline(missing_file) is None

    # 2. Corrupted model binary returns 'CORRUPTED'
    corrupt_file = tmp_path / "corrupt_model.joblib"
    corrupt_file.write_bytes(b"corrupted binary data not a pickle")
    assert app.load_model_pipeline(corrupt_file) == "CORRUPTED"

    # 3. Valid model binary loads properly
    valid_file = tmp_path / "valid_model.joblib"
    dummy_model = {"model_type": "LogisticRegression", "fitted": True}
    joblib.dump(dummy_model, valid_file)
    loaded = app.load_model_pipeline(valid_file)
    assert isinstance(loaded, dict)
    assert loaded["model_type"] == "LogisticRegression"
