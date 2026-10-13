"""Tests for UNSW-NB15 preprocessing and evaluation metrics pipeline."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.preprocessing import clean_dataframe, prepare_features_and_labels, build_column_transformer
from src.evaluate import compute_metrics
from src.train import (
    validate_preflight,
    train_pipeline,
    DEFAULT_TRAIN_PATH,
    DEFAULT_TEST_PATH,
    SAMPLE_PATH,
    DEFAULT_METRICS_ARTIFACT,
    DEFAULT_SAMPLE_METRICS_ARTIFACT,
    DEFAULT_SAMPLE_MODEL_ARTIFACT,
)




def test_clean_dataframe_handles_special_characters_and_nans():
    df = pd.DataFrame({
        "Proto ": ["tcp", "udp", "-"],
        " Service": ["http", "-", "dns"],
        "Dur": [1.2, np.nan, np.inf],
    })
    cleaned = clean_dataframe(df)

    assert "proto" in cleaned.columns
    assert "service" in cleaned.columns
    assert "dur" in cleaned.columns
    assert (cleaned["service"] == "-").sum() == 0
    assert cleaned["dur"].isna().sum() == 0
    assert not np.isinf(cleaned["dur"]).any()


def test_prepare_features_and_labels():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "dur": [0.1, 0.5, 1.2],
        "proto": ["tcp", "udp", "tcp"],
        "service": ["http", "dns", "http"],
        "state": ["FIN", "CON", "FIN"],
        "attack_cat": ["Normal", "Generic", "Normal"],
        "label": [0, 1, 0],
    })
    X, y = prepare_features_and_labels(df)

    assert len(X) == 3
    assert len(y) == 3
    assert "id" not in X.columns
    assert "attack_cat" not in X.columns
    assert "label" not in X.columns
    assert list(y) == [0, 1, 0]


def test_build_column_transformer_fitting():
    df = pd.DataFrame({
        "dur": [0.1, 0.5, 1.2],
        "spkts": [10, 20, 30],
        "proto": ["tcp", "udp", "tcp"],
        "service": ["http", "dns", "http"],
        "state": ["FIN", "CON", "FIN"],
    })
    preprocessor = build_column_transformer(
        numerical_features=["dur", "spkts"],
        categorical_features=["proto", "service", "state"],
    )
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 2  # Expanded by one-hot encoding


def test_compute_metrics():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    # TN=2, FP=1, FN=1, TP=2

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["confusion_matrix"]["true_negatives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 1
    assert metrics["confusion_matrix"]["false_negatives"] == 1
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert pytest.approx(metrics["false_positive_rate"], 0.01) == 1 / 3
    assert pytest.approx(metrics["recall"], 0.01) == 2 / 3


def test_train_pipeline_logistic_baseline(tmp_path):
    from pathlib import Path
    from src.train import train_pipeline

    sample_csv = Path("data/sample/unsw_sample.csv")
    model_out = tmp_path / "model.joblib"
    metrics_out = tmp_path / "metrics.json"

    metrics = train_pipeline(
        train_path=sample_csv,
        model_output=model_out,
        metrics_output=metrics_out,
        classifier_type="logistic",
    )

    assert model_out.exists()
    assert metrics_out.exists()
    assert metrics["classifier"] == "LogisticRegression"
    assert "accuracy" in metrics
    assert "false_positive_rate" in metrics
    assert metrics["is_sample_evaluation"] is True


def test_train_pipeline_random_forest(tmp_path):
    from pathlib import Path
    from src.train import train_pipeline

    sample_csv = Path("data/sample/unsw_sample.csv")
    model_out = tmp_path / "model_rf.joblib"
    metrics_out = tmp_path / "metrics_rf.json"

    metrics = train_pipeline(
        train_path=sample_csv,
        model_output=model_out,
        metrics_output=metrics_out,
        classifier_type="random_forest",
    )

    assert model_out.exists()
    assert metrics["classifier"] == "RandomForestClassifier"


def test_prepare_features_and_labels_filters_invalid_targets():
    df = pd.DataFrame({
        "dur": [0.1, 0.2, 0.3, 0.4],
        "proto": ["tcp", "udp", "tcp", "udp"],
        "label": [0, 1, np.nan, 99],  # 2 valid, 1 NaN, 1 invalid
    })
    X, y = prepare_features_and_labels(df)
    assert len(X) == 2
    assert len(y) == 2
    assert list(y) == [0, 1]


def test_prepare_features_and_labels_all_invalid_raises():
    df = pd.DataFrame({
        "dur": [0.1, 0.2],
        "proto": ["tcp", "udp"],
        "label": ["invalid", np.nan],
    })
    with pytest.raises(ValueError, match="No valid binary target values"):
        prepare_features_and_labels(df)


def test_train_pipeline_incompatible_schema_raises(tmp_path):
    from pathlib import Path
    from src.train import train_pipeline

    sample_csv = Path("data/sample/unsw_sample.csv")
    incompatible_csv = tmp_path / "incompatible.csv"
    # Create test CSV with only 2 columns (missing > 80% features)
    pd.DataFrame({"dur": [0.1, 0.2], "label": [0, 1]}).to_csv(incompatible_csv, index=False)

    with pytest.raises(ValueError, match="fundamentally incompatible"):
        train_pipeline(
            train_path=sample_csv,
            test_path=incompatible_csv,
            model_output=tmp_path / "m.joblib",
            metrics_output=tmp_path / "met.json",
        )


def test_train_pipeline_creates_backup(tmp_path):
    from pathlib import Path
    from src.train import train_pipeline

    sample_csv = Path("data/sample/unsw_sample.csv")
    model_out = tmp_path / "model.joblib"
    metrics_out = tmp_path / "metrics.json"

    # Pre-create artifacts
    model_out.write_text("old_model")
    metrics_out.write_text("old_metrics")

    train_pipeline(
        train_path=sample_csv,
        model_output=model_out,
        metrics_output=metrics_out,
    )

    backup_model = tmp_path / "model.joblib.bak"
    backup_metrics = tmp_path / "metrics.json.bak"
    assert backup_model.exists()
    assert backup_metrics.exists()
    assert backup_model.read_text() == "old_model"
    assert backup_metrics.read_text() == "old_metrics"


def test_label_and_attack_cat_leakage_prevented():
    """Confirms label, attack_cat, and id are excluded under case/whitespace variations."""
    df = pd.DataFrame({
        " ID ": [1, 2, 3],
        "Dur": [0.1, 0.5, 1.2],
        "Proto": ["tcp", "udp", "tcp"],
        "Service": ["http", "dns", "http"],
        "State": ["FIN", "CON", "FIN"],
        " Attack_Cat ": ["Normal", "Generic", "Normal"],
        " Label ": [0, 1, 0],
    })

    X, y = prepare_features_and_labels(df)

    # Feature matrix X checks
    assert "label" not in X.columns
    assert "attack_cat" not in X.columns
    assert "id" not in X.columns
    assert list(y) == [0, 1, 0]

    # Preprocessor transform checks
    num_cols = ["dur"]
    cat_cols = ["proto", "service", "state"]
    preprocessor = build_column_transformer(num_cols, cat_cols)
    preprocessor.fit(X)

    feature_names = preprocessor.get_feature_names_out()
    for name in feature_names:
        assert "label" not in name.lower()
        assert "attack_cat" not in name.lower()
        assert "id" not in name.lower()


def test_preprocessing_fitted_on_training_only():
    """Confirms preprocessor parameters match training set only and are unshifted by test set."""
    train_df = pd.DataFrame({
        "dur": [10.0, 20.0, 30.0],
        "proto": ["tcp", "tcp", "udp"],
        "service": ["http", "dns", "http"],
        "state": ["FIN", "FIN", "CON"],
        "label": [0, 1, 0],
    })
    test_df = pd.DataFrame({
        "dur": [1000.0, 2000.0, 3000.0],  # Radically different distribution
        "proto": ["tcp", "udp", "tcp"],
        "service": ["http", "http", "dns"],
        "state": ["FIN", "CON", "FIN"],
        "label": [1, 0, 1],
    })

    X_train, y_train = prepare_features_and_labels(train_df)
    X_test, y_test = prepare_features_and_labels(test_df)

    num_cols = ["dur"]
    cat_cols = ["proto", "service", "state"]
    preprocessor = build_column_transformer(num_cols, cat_cols)

    # Fit only on training data
    preprocessor.fit(X_train)
    scaler = preprocessor.named_transformers_["num"]

    expected_mean = np.mean([10.0, 20.0, 30.0])
    expected_var = np.var([10.0, 20.0, 30.0])

    assert pytest.approx(scaler.mean_[0], 0.001) == expected_mean
    assert pytest.approx(scaler.var_[0], 0.001) == expected_var

    # Transforming test data must NOT change fitted parameters
    _ = preprocessor.transform(X_test)
    assert pytest.approx(scaler.mean_[0], 0.001) == expected_mean
    assert pytest.approx(scaler.var_[0], 0.001) == expected_var


def test_default_benchmark_paths_resolution():
    """Verifies canonical benchmark paths match the documented locations."""
    from pathlib import Path
    assert DEFAULT_TRAIN_PATH == Path("data/raw/UNSW_NB15_training-set.csv")
    assert DEFAULT_TEST_PATH == Path("data/raw/UNSW_NB15_testing-set.csv")
    assert SAMPLE_PATH == Path("data/sample/unsw_sample.csv")


def test_validate_preflight_missing_training_file_raises(tmp_path):
    """Verifies preflight raises FileNotFoundError with instructions when train set is missing."""
    missing_train = tmp_path / "nonexistent_train.csv"
    existing_test = tmp_path / "existing_test.csv"
    existing_test.write_text("dummy")

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_preflight(train_path=missing_train, test_path=existing_test, is_sample=False)

    err_msg = str(exc_info.value)
    assert "PREFLIGHT CHECK FAILED" in err_msg
    assert str(missing_train) in err_msg
    assert "https://research.unsw.edu.au/projects/unsw-nb15-dataset" in err_msg


def test_validate_preflight_missing_testing_file_raises(tmp_path):
    """Verifies preflight raises FileNotFoundError with instructions when test set is missing."""
    existing_train = tmp_path / "existing_train.csv"
    missing_test = tmp_path / "nonexistent_test.csv"
    existing_train.write_text("dummy")

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_preflight(train_path=existing_train, test_path=missing_test, is_sample=False)

    err_msg = str(exc_info.value)
    assert "PREFLIGHT CHECK FAILED" in err_msg
    assert str(missing_test) in err_msg
    assert "data/raw/UNSW_NB15_testing-set.csv" in err_msg


def test_validate_preflight_missing_both_files_reports_both(tmp_path):
    """Verifies preflight reports both files as missing when neither exists."""
    missing_train = tmp_path / "nonexistent_train.csv"
    missing_test = tmp_path / "nonexistent_test.csv"

    with pytest.raises(FileNotFoundError) as exc_info:
        validate_preflight(train_path=missing_train, test_path=missing_test, is_sample=False)

    err_msg = str(exc_info.value)
    assert str(missing_train) in err_msg
    assert str(missing_test) in err_msg


def test_validate_preflight_no_test_path_for_benchmark_raises(tmp_path):
    """Verifies preflight forbids omitted test_path during benchmark mode to preserve official split."""
    existing_train = tmp_path / "train.csv"
    existing_train.write_text("dummy")

    with pytest.raises(ValueError, match="Authentic benchmark training requires a separate test dataset"):
        validate_preflight(train_path=existing_train, test_path=None, is_sample=False)


def test_preflight_prevents_silent_sample_fallback(tmp_path):
    """Confirms train_pipeline raises FileNotFoundError instead of silently falling back to sample."""
    missing_train = tmp_path / "unsw_train_missing.csv"
    missing_test = tmp_path / "unsw_test_missing.csv"

    with pytest.raises(FileNotFoundError, match="PREFLIGHT CHECK FAILED"):
        train_pipeline(
            train_path=missing_train,
            test_path=missing_test,
            model_output=tmp_path / "out.joblib",
            metrics_output=tmp_path / "met.json",
        )


def test_sample_mode_provenance_and_execution(tmp_path):
    """Confirms sample mode succeeds and sets sample_demonstration provenance."""
    sample_csv = Path("data/sample/unsw_sample.csv")
    model_out = tmp_path / "sample_model.joblib"
    metrics_out = tmp_path / "sample_metrics.json"

    metrics = train_pipeline(
        train_path=sample_csv,
        model_output=model_out,
        metrics_output=metrics_out,
    )

    assert metrics["is_sample_evaluation"] is True
    assert metrics["dataset_source"] == "sample_demonstration"


def test_sample_mode_default_paths_separation(monkeypatch, tmp_path):
    """Confirms that sample mode defaults to sample artifacts and does not overwrite canonical benchmark metrics."""
    sample_csv = Path("data/sample/unsw_sample.csv")
    dummy_canonical_metrics = tmp_path / "metrics.json"
    dummy_canonical_metrics.write_text('{"benchmark": true}', encoding="utf-8")

    sample_metrics_target = tmp_path / "metrics_sample.json"
    sample_model_target = tmp_path / "model_sample.joblib"

    monkeypatch.setattr("src.train.DEFAULT_METRICS_ARTIFACT", dummy_canonical_metrics)
    monkeypatch.setattr("src.train.DEFAULT_SAMPLE_METRICS_ARTIFACT", sample_metrics_target)
    monkeypatch.setattr("src.train.DEFAULT_SAMPLE_MODEL_ARTIFACT", sample_model_target)

    metrics = train_pipeline(train_path=sample_csv)

    # Sample outputs were written
    assert sample_metrics_target.exists()
    assert sample_model_target.exists()
    assert metrics["is_sample_evaluation"] is True

    # Canonical benchmark metrics file was untouched!
    assert dummy_canonical_metrics.read_text(encoding="utf-8") == '{"benchmark": true}'
