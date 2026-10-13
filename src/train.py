"""Training script for the UNSW-NB15 Cybersecurity Threat Detection model.

Loads training and testing data, constructs the scikit-learn preprocessing
and classification pipeline, trains the model, evaluates test metrics,
and saves artifacts to the canonical artifacts/ directory.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.preprocessing import (
    prepare_features_and_labels,
    build_column_transformer,
    CATEGORICAL_FEATURES,
)
from src.evaluate import compute_metrics, save_metrics


# Canonical paths
DEFAULT_TRAIN_PATH = Path("data/raw/UNSW_NB15_training-set.csv")
DEFAULT_TEST_PATH = Path("data/raw/UNSW_NB15_testing-set.csv")
SAMPLE_PATH = Path("data/sample/unsw_sample.csv")
DEFAULT_MODEL_ARTIFACT = Path("artifacts/model_pipeline.joblib")
DEFAULT_METRICS_ARTIFACT = Path("artifacts/metrics.json")
DEFAULT_SAMPLE_MODEL_ARTIFACT = Path("artifacts/model_sample.joblib")
DEFAULT_SAMPLE_METRICS_ARTIFACT = Path("artifacts/metrics_sample.json")


def validate_preflight(
    train_path: Path,
    test_path: Optional[Path] = None,
    is_sample: bool = False,
) -> None:
    """Performs dataset preflight validation before training or evaluation.

    Ensures authentic benchmark files exist and prevents silent fallback to
    synthetic sample data.

    Args:
        train_path: Path to training dataset.
        test_path: Path to testing dataset (required for authentic benchmark).
        is_sample: Boolean indicating if explicit sample demonstration mode is enabled.

    Raises:
        FileNotFoundError: If any required dataset is missing.
        ValueError: If benchmark training is attempted without a separate test dataset.
    """
    train_path = Path(train_path)

    if is_sample:
        if not train_path.exists():
            raise FileNotFoundError(
                f"[PREFLIGHT FAILURE] Sample demonstration dataset not found at: {train_path}\n"
                f"Ensure 'data/sample/unsw_sample.csv' exists in the repository."
            )
        return

    # Benchmark dataset preflight check
    missing_files = []
    if not train_path.exists():
        missing_files.append(f"Training dataset: {train_path} [MISSING]")

    if test_path is None:
        raise ValueError(
            "[PREFLIGHT FAILURE] Authentic benchmark training requires a separate test dataset "
            "to preserve the official split without random merging or re-splitting.\n"
            "Please provide '--test-path' or ensure 'data/raw/UNSW_NB15_testing-set.csv' exists."
        )

    test_path = Path(test_path)
    if not test_path.exists():
        missing_files.append(f"Testing dataset:  {test_path} [MISSING]")

    if missing_files:
        missing_details = "\n  - ".join(missing_files)
        msg = (
            f"\n[PREFLIGHT CHECK FAILED] Required authentic UNSW-NB15 benchmark dataset file(s) missing:\n"
            f"  - {missing_details}\n\n"
            f"Instructions to resolve:\n"
            f"  1. Download the official UNSW-NB15 dataset files from:\n"
            f"     https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            f"  2. Place the downloaded files in 'data/raw/':\n"
            f"     - data/raw/UNSW_NB15_training-set.csv\n"
            f"     - data/raw/UNSW_NB15_testing-set.csv\n"
            f"  3. Note: The training pipeline will NEVER silently fall back to synthetic demonstration data.\n"
            f"     To run an offline test using the demonstration mock data instead, explicitly use:\n"
            f"     python src/train.py --sample\n"
        )
        raise FileNotFoundError(msg)


def train_pipeline(
    train_path: Path,
    test_path: Optional[Path] = None,
    model_output: Optional[Path] = None,
    metrics_output: Optional[Path] = None,
    random_state: int = 42,
    classifier_type: str = "logistic",
) -> dict:
    """Executes the full model training, evaluation, and artifact serialization.

    Args:
        train_path: Path to training CSV.
        test_path: Path to testing CSV (optional, splits train_path if None in sample mode).
        model_output: Output path for serialized joblib model pipeline (defaults to sample or benchmark artifact).
        metrics_output: Output path for evaluation metrics JSON (defaults to sample or benchmark artifact).
        random_state: Reproducibility seed.
        classifier_type: Classifier architecture ('logistic' or 'random_forest').

    Returns:
        Evaluation metrics dictionary.
    """
    train_path = Path(train_path)
    is_sample = bool(train_path.resolve() == SAMPLE_PATH.resolve())
    validate_preflight(train_path=train_path, test_path=test_path, is_sample=is_sample)

    if model_output is None:
        model_output = DEFAULT_SAMPLE_MODEL_ARTIFACT if is_sample else DEFAULT_MODEL_ARTIFACT
    else:
        model_output = Path(model_output)

    if metrics_output is None:
        metrics_output = DEFAULT_SAMPLE_METRICS_ARTIFACT if is_sample else DEFAULT_METRICS_ARTIFACT
    else:
        metrics_output = Path(metrics_output)

    print(f"[*] Loading training dataset from: {train_path}")
    train_df = pd.read_csv(train_path)
    X_train, y_train = prepare_features_and_labels(train_df)

    if test_path and Path(test_path).exists():
        print(f"[*] Loading testing dataset from: {test_path}")
        test_df = pd.read_csv(test_path)
        X_test, y_test = prepare_features_and_labels(test_df)
    else:
        # Only executed in sample demonstration mode where test_path is None
        print("[*] Demonstration mode: Performing stratified 80/20 split on sample data.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )


    # Determine numerical and categorical columns present on training split
    present_cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    present_num_cols = [c for c in X_train.columns if c not in present_cat_cols]

    # Check and align test dataset schema against training schema
    missing_in_test = [c for c in X_train.columns if c not in X_test.columns]
    extra_in_test = [c for c in X_test.columns if c not in X_train.columns]

    if len(missing_in_test) > len(X_train.columns) * 0.2:
        raise ValueError(
            f"Test dataset is fundamentally incompatible with training schema.\n"
            f"Missing {len(missing_in_test)} columns out of {len(X_train.columns)}: {missing_in_test[:5]}..."
        )
    elif missing_in_test:
        print(f"[!] Aligning test schema: Imputing {len(missing_in_test)} minor missing test columns.")
        for col in missing_in_test:
            X_test[col] = "unknown" if col in present_cat_cols else 0.0

    if extra_in_test:
        print(f"[*] Aligning test schema: Dropping {len(extra_in_test)} extra test columns not in training set.")

    X_test = X_test[X_train.columns]

    print(f"[*] Features: {len(present_num_cols)} numerical, {len(present_cat_cols)} categorical.")
    print(f"[*] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    preprocessor = build_column_transformer(
        numerical_features=present_num_cols,
        categorical_features=present_cat_cols
    )

    if classifier_type.lower() == "random_forest":
        print("[*] Instantiating Random Forest Classifier...")
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            n_jobs=-1,
            random_state=random_state
        )
    else:
        print("[*] Instantiating Logistic Regression Baseline Classifier...")
        classifier = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    print(f"[*] Training {type(classifier).__name__} pipeline...")
    pipeline.fit(X_train, y_train)

    print("[*] Evaluating model on held-out test split...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_prob)
    is_sample = bool(train_path.resolve() == SAMPLE_PATH.resolve())
    metrics["classifier"] = type(classifier).__name__
    metrics["is_sample_evaluation"] = is_sample
    metrics["dataset_source"] = "sample_demonstration" if is_sample else "authentic_unsw_nb15"

    print(f"[+] Evaluation Results ({metrics['classifier']}):")
    print(f"    - Accuracy:          {metrics['accuracy']:.4f}")
    print(f"    - Attack Precision:  {metrics['precision']:.4f}")
    print(f"    - Attack Recall:     {metrics['recall']:.4f}")
    print(f"    - Attack F1-Score:   {metrics['f1_score']:.4f}")
    print(f"    - False Alarm Rate:  {metrics['false_positive_rate']:.4f}")
    if metrics.get("average_precision") is not None:
        print(f"    - Avg Precision:     {metrics['average_precision']:.4f}")
    if metrics.get("roc_auc") is not None:
        print(f"    - ROC-AUC:           {metrics['roc_auc']:.4f}")
    if metrics["is_sample_evaluation"]:
        print("    [!] NOTE: Metrics computed on demonstration sample; does not reflect authentic benchmark performance.")

    # Safe backup of previous artifacts before replacement
    import shutil
    model_output = Path(model_output)
    if model_output.exists():
        backup_model = model_output.with_suffix(".joblib.bak")
        shutil.copy2(model_output, backup_model)

    metrics_output = Path(metrics_output)
    if metrics_output.exists():
        backup_metrics = metrics_output.with_suffix(".json.bak")
        shutil.copy2(metrics_output, backup_metrics)

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_output)
    print(f"[+] Serialized model pipeline to: {model_output}")

    save_metrics(metrics, metrics_output)
    print(f"[+] Saved metrics to: {metrics_output}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train UNSW-NB15 Threat Detection Classifier")
    parser.add_argument(
        "--train-path",
        type=str,
        default=str(DEFAULT_TRAIN_PATH),
        help=f"Path to training CSV (default: {DEFAULT_TRAIN_PATH})"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=str(DEFAULT_TEST_PATH),
        help=f"Path to testing CSV (default: {DEFAULT_TEST_PATH})"
    )
    parser.add_argument("--sample", action="store_true", help="Train on the demo sample dataset in data/sample/")
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
        help="Model output path (default: artifacts/model_sample.joblib for --sample, artifacts/model_pipeline.joblib for benchmark)"
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default=None,
        help="Metrics output path (default: artifacts/metrics_sample.json for --sample, artifacts/metrics.json for benchmark)"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        choices=["logistic", "random_forest"],
        default="logistic",
        help="Supervised classifier type (default: logistic baseline)"
    )
    args = parser.parse_args()

    train_path = Path(SAMPLE_PATH if args.sample else args.train_path)
    test_path = None if args.sample else Path(args.test_path)

    model_output = Path(args.model_out) if args.model_out else None
    metrics_output = Path(args.metrics_out) if args.metrics_out else None

    try:
        train_pipeline(
            train_path=train_path,
            test_path=test_path,
            model_output=model_output,
            metrics_output=metrics_output,
            classifier_type=args.classifier,
        )
    except (FileNotFoundError, ValueError) as err:
        print(f"\n[ERROR] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
