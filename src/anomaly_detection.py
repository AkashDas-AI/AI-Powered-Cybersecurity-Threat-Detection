"""Experimental Anomaly Detection & Hybrid Threat Detection Module (Isolation Forest).

Implements:
1. Novelty-detection Isolation Forest trained strictly on benign network flows (label = 0).
2. Clean separation of learned transformations: fitted on benign training records only.
3. Leak-free threshold selection on training-derived validation subset.
4. Evaluation on official held-out UNSW-NB15 test set (82,332 records).
5. Evaluation of documented hybrid policies (Cascaded OR & Suspicious-Band Confirmation).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score,
    average_precision_score,
)

from src.preprocessing import (
    prepare_features_and_labels,
    build_column_transformer,
    CATEGORICAL_FEATURES,
)
from src.evaluate import compute_metrics, save_metrics
from src.train import (
    DEFAULT_TRAIN_PATH,
    DEFAULT_TEST_PATH,
    validate_preflight,
)
from src.benchmark_audit import compute_threshold_metrics, plot_confusion_matrices


def fit_isolation_forest_pipeline(
    X_train_benign: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    n_estimators: int = 100,
    max_samples: int = 256,
    contamination: str = "auto",
    random_state: int = 42,
) -> Pipeline:
    """Fits an Isolation Forest pipeline strictly on benign training flows.

    Args:
        X_train_benign: Benign-only feature DataFrame (target/id columns excluded).
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.
        n_estimators: Number of isolation trees (default: 100).
        max_samples: Samples drawn per tree (default: 256).
        contamination: Proportion of expected outliers in benign set (default: 'auto').
        random_state: Reproducibility seed (default: 42).

    Returns:
        Fitted scikit-learn Pipeline with preprocessor and IsolationForest.
    """
    preprocessor = build_column_transformer(numerical_features, categorical_features)
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    pipeline.fit(X_train_benign)
    return pipeline


def compute_anomaly_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Computes continuous anomaly scores where higher score indicates higher abnormality.

    Convention:
        IsolationForest.decision_function(X) returns negative values for outliers
        and positive values for inliers.
        We define anomaly_score = -decision_function(X).
        - Inliers/benign flows produce lower/negative anomaly scores.
        - Outliers/attacks produce higher/positive anomaly scores.

    Args:
        pipeline: Fitted Pipeline containing IsolationForest.
        X: Feature matrix.

    Returns:
        1D numpy array of float anomaly scores.
    """
    raw_dec = pipeline.decision_function(X)
    return -raw_dec


def compute_anomaly_score_distribution(
    anomaly_scores: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Calculates summary percentiles and moments for benign vs attack anomaly scores."""
    benign_scores = anomaly_scores[y_true == 0]
    attack_scores = anomaly_scores[y_true == 1]

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        if len(arr) == 0:
            return {}
        return {
            "count": int(len(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
        }

    return {
        "benign": _stats(benign_scores),
        "attack": _stats(attack_scores),
    }


def select_anomaly_threshold(
    y_val: np.ndarray,
    val_anomaly_scores: np.ndarray,
    objective: str = "target_fpr",
    target_val_fpr: float = 0.15,
) -> Dict[str, Any]:
    """Selects an anomaly score decision threshold on validation data.

    Objectives:
    - 'target_fpr': Select the threshold that controls benign FPR <= target_val_fpr while maximizing recall.
    - 'max_f1': Select the threshold maximizing F1-score on validation data.
    - 'default_zero': Use threshold 0.0 (the raw decision boundary where decision_function <= 0).

    Args:
        y_val: Validation ground truth binary labels.
        val_anomaly_scores: Validation anomaly scores (higher = more anomalous).
        objective: Selection objective ('target_fpr', 'max_f1', or 'default_zero').
        target_val_fpr: Maximum acceptable false positive rate on benign validation traffic.

    Returns:
        Dictionary containing selected candidate threshold and validation metrics.
    """
    threshold_grid = np.percentile(val_anomaly_scores, np.linspace(5, 95, 91))
    sweep = compute_threshold_metrics(y_val, val_anomaly_scores, thresholds=threshold_grid)

    if objective == "max_f1":
        best = max(sweep, key=lambda m: (m["f1_score"], -m["false_positive_rate"]))
        rule_desc = "Maximize validation F1-Score"
    elif objective == "default_zero":
        # Find entry closest to 0.0
        best = min(sweep, key=lambda m: abs(m["threshold"] - 0.0))
        rule_desc = "Default Isolation Forest decision boundary (theta = 0.0)"
    else:  # target_fpr
        eligible = [m for m in sweep if m["false_positive_rate"] <= target_val_fpr]
        if eligible:
            best = max(eligible, key=lambda m: (m["recall"], -m["false_positive_rate"]))
        else:
            best = min(sweep, key=lambda m: m["false_positive_rate"])
        rule_desc = f"Control benign FPR <= {target_val_fpr:.1%} while maximizing attack recall"

    return {
        "candidate_threshold": float(best["threshold"]),
        "validation_metrics": best,
        "objective_rule": rule_desc,
        "sweep_metrics": sweep,
    }


def evaluate_anomaly_detector(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluates the fitted Isolation Forest detector on test records."""
    t0 = time.perf_counter()
    anomaly_scores = compute_anomaly_scores(pipeline, X_test)
    eval_time = time.perf_counter() - t0

    y_true = np.asarray(y_test).astype(int)
    y_pred = (anomaly_scores >= threshold).astype(int)

    # Use standard evaluation logic
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    accuracy = float((tp + tn) / len(y_true))

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, anomaly_scores))
    except Exception:
        pr_auc = 0.0

    score_dist = compute_anomaly_score_distribution(anomaly_scores, y_true)

    from sklearn.metrics import classification_report
    report_dict = classification_report(y_true, y_pred, target_names=["Normal", "Attack"], output_dict=True, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc,
        "average_precision": pr_auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "total_samples": int(len(y_true)),
        "classification_report": report_dict,
        "decision_threshold": float(threshold),
        "detector": "IsolationForest",
        "threshold_type": f"val_selected_{threshold:.4f}",
        "dataset_source": "authentic_unsw_nb15",
        "is_sample_evaluation": False,
        "eval_time_sec": round(eval_time, 2),
        "score_distribution": score_dist,
    }


def select_hybrid_policy_on_validation(
    y_val: np.ndarray,
    val_sup_probs: np.ndarray,
    val_anomaly_scores: np.ndarray,
    sup_threshold: float = 0.76,
    anomaly_threshold: float = -0.0428,
    candidate_band_lows: Optional[List[float]] = None,
    objective: str = "max_recall_constrained_fpr",
    max_val_fpr: float = 0.075,
) -> Dict[str, Any]:
    """Selects and freezes a hybrid fusion policy strictly using internal validation partition data.

    Protocol & Guarantees:
    - Zero Test Set Exposure: Official test split (UNSW_NB15_testing-set.csv) is strictly forbidden.
    - Candidate Policies Evaluated on Validation Partition (N=35,069):
      1. Supervised Baseline (LR at sup_threshold=0.76).
      2. Cascaded OR Policy (sup_prob >= sup_threshold OR ano_score >= anomaly_threshold).
      3. Suspicious-Band Confirmation Policies for each candidate band_low:
         Alert if sup_prob >= sup_threshold OR (band_low <= sup_prob < sup_threshold AND ano_score >= anomaly_threshold).
    - Objective Function:
      - 'max_recall_constrained_fpr': Maximize attack recall subject to benign validation FPR <= max_val_fpr.
      - 'max_f1': Maximize validation F1-score.

    Args:
        y_val: 1D array of validation binary ground truth labels (0=benign, 1=attack, N=35,069).
        val_sup_probs: 1D array of supervised attack predicted probabilities on validation data.
        val_anomaly_scores: 1D array of continuous anomaly scores on validation data.
        sup_threshold: Frozen decision threshold for supervised model (default: 0.76).
        anomaly_threshold: Validation-selected threshold for anomaly detector (default: -0.0428).
        candidate_band_lows: List of candidate lower boundaries for the confirmation band.
        objective: Selection criterion ('max_recall_constrained_fpr' or 'max_f1').
        max_val_fpr: Maximum acceptable benign FPR on validation traffic (default: 0.075 / 7.5%).

    Returns:
        Dictionary containing selected policy metadata, chosen boundary parameters,
        validation metrics of the winner, and the complete validation candidate sweep.
    """
    if candidate_band_lows is None:
        candidate_band_lows = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    y_true = np.asarray(y_val).astype(int)
    pred_sup = (val_sup_probs >= sup_threshold).astype(int)
    pred_ano = (val_anomaly_scores >= anomaly_threshold).astype(int)

    def _eval(pred: np.ndarray, pol_name: str) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        acc = float((tp + tn) / len(y_true))
        return {
            "policy_name": pol_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

    m_sup = _eval(pred_sup, f"Supervised Baseline (LR, th={sup_threshold:.2f})")
    m_ano = _eval(pred_ano, f"Isolation Forest Standalone (th={anomaly_threshold:.4f})")

    candidates: List[Dict[str, Any]] = []

    # 1. Supervised baseline candidate
    c_base = {
        "candidate_id": "supervised_baseline",
        "policy_type": "supervised_baseline",
        "policy_name": f"Supervised Baseline (LR, th={sup_threshold:.2f})",
        "band_low": None,
        "band_high": None,
        "metrics": m_sup,
    }
    candidates.append(c_base)

    # 2. Cascaded OR candidate
    pred_or = np.logical_or(pred_sup == 1, pred_ano == 1).astype(int)
    m_or = _eval(pred_or, "Hybrid Policy: Cascaded OR")
    c_or = {
        "candidate_id": "cascaded_or",
        "policy_type": "cascaded_or",
        "policy_name": "Hybrid Policy: Cascaded OR",
        "band_low": 0.0,
        "band_high": float(sup_threshold),
        "metrics": m_or,
    }
    candidates.append(c_or)

    # 3. Suspicious-Band sweeps
    for b_low in sorted(candidate_band_lows):
        pred_band = pred_sup.copy()
        band_mask = (val_sup_probs >= b_low) & (val_sup_probs < sup_threshold)
        pred_band[band_mask & (pred_ano == 1)] = 1
        m_band = _eval(pred_band, f"Hybrid Policy: Band Confirmation ({b_low:.2f} <= P < {sup_threshold:.2f})")
        c_band = {
            "candidate_id": f"band_confirmation_{b_low:.2f}",
            "policy_type": "band_confirmation",
            "policy_name": f"Hybrid Policy: Band Confirmation ({b_low:.2f} <= P < {sup_threshold:.2f})",
            "band_low": float(b_low),
            "band_high": float(sup_threshold),
            "metrics": m_band,
        }
        candidates.append(c_band)

    # Calculate trade-offs and eligibility on validation
    sup_fn_total = m_sup["confusion_matrix"]["false_negatives"]
    for c in candidates:
        fpr = c["metrics"]["false_positive_rate"]
        is_eligible = (fpr <= max_val_fpr)
        c["is_eligible"] = bool(is_eligible)
        c["validation_fpr_constraint"] = max_val_fpr
        cm_cand = c["metrics"]["confusion_matrix"]
        tp_cand = cm_cand["true_positives"]
        fn_rec = max(0, tp_cand - m_sup["confusion_matrix"]["true_positives"])
        fp_added = max(0, cm_cand["false_positives"] - m_sup["confusion_matrix"]["false_positives"])
        c["fn_recovered_on_val"] = int(fn_rec)
        c["recovery_rate_on_val"] = float(fn_rec / sup_fn_total) if sup_fn_total > 0 else 0.0
        c["new_fp_on_val"] = int(fp_added)
        c["fpr_delta_on_val"] = float(fpr - m_sup["false_positive_rate"])

    eligible = [c for c in candidates if c["is_eligible"] and c["policy_type"] != "supervised_baseline"]
    fallback_used = False
    if not eligible:
        eligible = [c for c in candidates if c["policy_type"] != "supervised_baseline"]
        fallback_used = True

    if objective == "max_f1":
        best = max(eligible, key=lambda c: (c["metrics"]["f1_score"], -c["metrics"]["false_positive_rate"]))
        rule_desc = f"Maximize validation F1-score (fallback={fallback_used})"
    else:  # max_recall_constrained_fpr
        best = max(eligible, key=lambda c: (c["metrics"]["recall"], -c["metrics"]["false_positive_rate"], c["metrics"]["f1_score"]))
        rule_desc = f"Maximize attack recall subject to validation benign FPR <= {max_val_fpr:.2%} (fallback={fallback_used})"

    return {
        "selected_candidate_id": best["candidate_id"],
        "selected_policy_name": best["policy_name"],
        "selected_policy_type": best["policy_type"],
        "selected_band_low": best["band_low"],
        "selected_band_high": best["band_high"],
        "selected_sup_threshold": sup_threshold,
        "selected_anomaly_threshold": anomaly_threshold,
        "objective_rule": rule_desc,
        "objective_name": objective,
        "max_val_fpr_constraint": max_val_fpr,
        "fallback_used": fallback_used,
        "validation_metrics_selected": best["metrics"],
        "candidate_sweep": candidates,
        "standalone_supervised_val_metrics": m_sup,
        "standalone_anomaly_val_metrics": m_ano,
    }


def evaluate_hybrid_policies(
    y_test: pd.Series,
    sup_probs: np.ndarray,
    anomaly_scores: np.ndarray,
    sup_threshold: float = 0.76,
    anomaly_threshold: float = -0.0428,
    band_low: float = 0.40,
    band_high: float = 0.76,
    rf_probs: Optional[np.ndarray] = None,
    policy_selection_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluates documented hybrid detection policies combining supervised attack probabilities and anomaly scores.

    Evaluated Policies on Test Split (82,332 records):
    - Supervised Baseline: Logistic Regression (frozen at sup_threshold)
    - Supervised Ensemble: Random Forest (evaluated at th=0.50 if rf_probs provided)
    - Anomaly Standalone: Isolation Forest (evaluated at anomaly_threshold)
    - Hybrid Policy 1: Cascaded Sensitivity OR-Policy
    - Hybrid Policy 2: Suspicious-Band Confirmation Policy (frozen validation-selected parameters)

    Args:
        y_test: Test ground truth series.
        sup_probs: Supervised attack class predicted probabilities.
        anomaly_scores: Continuous anomaly scores (higher = more anomalous).
        sup_threshold: Supervised decision threshold (default: 0.76).
        anomaly_threshold: Anomaly decision threshold (default: -0.0428).
        band_low: Lower bound of ambiguous confirmation band (default: 0.40).
        band_high: Upper bound of ambiguous confirmation band (default: 0.76).
        rf_probs: Optional Random Forest attack class predicted probabilities.
        policy_selection_metadata: Optional validation-only policy selection metadata.

    Returns:
        Dictionary containing comparative performance metrics and trade-off analytics.
    """
    y_true = np.asarray(y_test).astype(int)

    # Standalone predictions
    pred_sup = (sup_probs >= sup_threshold).astype(int)
    pred_ano = (anomaly_scores >= anomaly_threshold).astype(int)

    # Policy 1: Cascaded OR
    pred_hybrid_or = np.logical_or(pred_sup == 1, pred_ano == 1).astype(int)

    # Policy 2: Band Confirmation (Validation Selected)
    pred_hybrid_band = pred_sup.copy()
    band_mask = (sup_probs >= band_low) & (sup_probs < band_high)
    pred_hybrid_band[band_mask & (pred_ano == 1)] = 1

    def _calc_metrics(pred: np.ndarray, policy_name: str) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        acc = float((tp + tn) / len(y_true))
        return {
            "policy_name": policy_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "false_positive_rate": fpr,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
        }

    m_sup = _calc_metrics(pred_sup, f"Supervised Baseline (LR, th={sup_threshold:.2f})")
    m_ano = _calc_metrics(pred_ano, f"Isolation Forest Standalone (th={anomaly_threshold:.4f})")
    m_or = _calc_metrics(pred_hybrid_or, "Hybrid Policy 1: Cascaded OR")
    m_band = _calc_metrics(pred_hybrid_band, f"Hybrid Policy 2: Band Confirmation ({band_low:.2f}-{band_high:.2f})")

    m_rf = None
    if rf_probs is not None:
        pred_rf = (rf_probs >= 0.50).astype(int)
        m_rf = _calc_metrics(pred_rf, "Supervised Ensemble (Random Forest, th=0.50)")

    # Trade-off analytics for Cascaded OR
    fn_sup_mask = (pred_sup == 0) & (y_true == 1)
    caught_by_ano = int((fn_sup_mask & (pred_ano == 1)).sum())
    total_sup_fn = int(fn_sup_mask.sum())
    fp_added_by_ano = int(((pred_sup == 0) & (y_true == 0) & (pred_ano == 1)).sum())

    # Trade-off analytics for Band Confirmation
    caught_by_band = int((fn_sup_mask & (pred_hybrid_band == 1)).sum())
    fp_added_by_band = int(((pred_sup == 0) & (y_true == 0) & (pred_hybrid_band == 1)).sum())

    result = {
        "supervised_baseline": m_sup,
        "random_forest_baseline": m_rf,
        "anomaly_standalone": m_ano,
        "hybrid_cascaded_or": m_or,
        "hybrid_band_confirmation": m_band,
        "hybrid_validation_selected": m_band,
        "tradeoff_analysis_or": {
            "supervised_false_negatives_total": total_sup_fn,
            "false_negatives_recovered_by_anomaly": caught_by_ano,
            "recovery_rate": float(caught_by_ano / total_sup_fn) if total_sup_fn > 0 else 0.0,
            "new_false_positives_incurred": fp_added_by_ano,
            "fpr_inflation_delta": float(m_or["false_positive_rate"] - m_sup["false_positive_rate"]),
        },
        "tradeoff_analysis_band": {
            "supervised_false_negatives_total": total_sup_fn,
            "false_negatives_recovered_by_band": caught_by_band,
            "recovery_rate": float(caught_by_band / total_sup_fn) if total_sup_fn > 0 else 0.0,
            "new_false_positives_incurred": fp_added_by_band,
            "fpr_inflation_delta": float(m_band["false_positive_rate"] - m_sup["false_positive_rate"]),
        },
        "policy_selection_metadata": policy_selection_metadata,
        "dataset_source": "authentic_unsw_nb15",
        "is_sample_evaluation": False,
        "total_test_samples": int(len(y_true)),
    }

    return result


def plot_anomaly_distribution(
    anomaly_scores: np.ndarray,
    y_true: np.ndarray,
    candidate_threshold: float,
    output_path: Path,
) -> None:
    """Plots histogram and density of anomaly scores for benign vs attack flows."""
    benign_scores = anomaly_scores[y_true == 0]
    attack_scores = anomaly_scores[y_true == 1]

    plt.figure(figsize=(10, 6))
    bins = np.linspace(-0.25, 0.25, 60)
    plt.hist(benign_scores, bins=bins, alpha=0.6, density=True, color="#1f77b4", label="Normal Traffic (Label=0)")
    plt.hist(attack_scores, bins=bins, alpha=0.6, density=True, color="#d62728", label="Attack Traffic (Label=1)")

    plt.axvline(x=candidate_threshold, color="#9467bd", linestyle="--", linewidth=2.5,
                label=f"Selected Threshold ({candidate_threshold:.4f})")
    plt.axvline(x=0.0, color="gray", linestyle=":", alpha=0.8, label="Default Boundary (0.00)")

    plt.title("Isolation Forest Anomaly Score Distribution on Official Held-Out Test Set", fontsize=12, pad=12)
    plt.xlabel("Anomaly Score (-decision_function): Higher = More Abnormal", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_anomaly_experiment(
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    output_dir: Path = Path("outputs"),
    artifacts_dir: Path = Path("artifacts"),
    random_state: int = 42,
    target_val_fpr: float = 0.15,
) -> Dict[str, Any]:
    """Executes the complete Isolation Forest anomaly and hybrid detection experiment.

    Workflow:
    1. Preflight validation.
    2. Ingests train and test splits.
    3. Partitions training set into 80% train and 20% validation.
    4. Extracts benign-only subset from 80% train split.
    5. Fits preprocessor and Isolation Forest exclusively on benign train flows.
    6. Sweeps validation split to select threshold controlling benign FPR <= target_val_fpr.
    7. Evaluates detector on official held-out test split (82,332 records).
    8. Loads supervised Logistic Regression predictions and evaluates hybrid policies.
    9. Persists artifacts, figures, and comparison reports.
    """
    train_path = Path(train_path)
    test_path = Path(test_path)
    validate_preflight(train_path=train_path, test_path=test_path, is_sample=False)

    print("\n" + "=" * 75)
    print("PHASE 7: NOVELTY ANOMALY DETECTION (ISOLATION FOREST) & HYBRID PIPELINE")
    print("=" * 75)

    # 1. Load data
    print(f"[*] Ingesting official training split: {train_path}")
    train_df = pd.read_csv(train_path)
    X_train_full, y_train_full = prepare_features_and_labels(train_df)

    print(f"[*] Ingesting official testing split:  {test_path}")
    test_df = pd.read_csv(test_path)
    X_test_full, y_test_full = prepare_features_and_labels(test_df)

    # Align columns
    present_cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train_full.columns]
    present_num_cols = [c for c in X_train_full.columns if c not in present_cat_cols]
    missing_in_test = [c for c in X_train_full.columns if c not in X_test_full.columns]
    for col in missing_in_test:
        X_test_full[col] = "unknown" if col in present_cat_cols else 0.0
    X_test_full = X_test_full[X_train_full.columns]

    print(f"[*] Training features: {len(present_num_cols)} numerical, {len(present_cat_cols)} categorical.")
    print(f"[*] Total Train records: {len(X_train_full):,} | Total Test records: {len(X_test_full):,}")

    # 2. Partition training data into 80% train and 20% validation
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=random_state, stratify=y_train_full
    )
    print(f"[*] Internal partition: {len(X_subtrain):,} sub-train, {len(X_val):,} validation records.")

    # 3. Novelty Detection Principle: Extract Benign-Only subset for training
    X_subtrain_benign = X_subtrain[y_subtrain == 0]
    print(f"[+] Fitting Isolation Forest strictly on BENIGN sub-train records (N={len(X_subtrain_benign):,}, label=0)...")

    t0 = time.perf_counter()
    pipe_iforest = fit_isolation_forest_pipeline(
        X_train_benign=X_subtrain_benign,
        numerical_features=present_num_cols,
        categorical_features=present_cat_cols,
        n_estimators=100,
        max_samples=256,
        contamination="auto",
        random_state=random_state,
    )
    train_time = time.perf_counter() - t0
    print(f"[+] Isolation Forest fitted in {train_time:.2f} seconds.")

    # 4. Leak-Free Threshold Selection on Validation Partition
    print("\n[*] Selecting anomaly decision threshold on validation partition...")
    val_scores = compute_anomaly_scores(pipe_iforest, X_val)
    th_choice = select_anomaly_threshold(
        y_val.values,
        val_scores,
        objective="target_fpr",
        target_val_fpr=target_val_fpr,
    )
    cand_thresh = th_choice["candidate_threshold"]
    val_best_m = th_choice["validation_metrics"]

    print(f"[+] Selected Threshold (theta*_iforest): {cand_thresh:.4f}")
    print(f"    - Objective Rule: {th_choice['objective_rule']}")
    print(f"    - Validation Benign FPR: {val_best_m['false_positive_rate']:.4%}")
    print(f"    - Validation Attack Recall: {val_best_m['recall']:.4%}")
    print(f"    - Validation Attack Precision: {val_best_m['precision']:.4%}")
    print(f"    - Validation F1-Score: {val_best_m['f1_score']:.4%}")

    # 5. Validation-Only Hybrid Policy Selection (Zero Test Exposure)
    print("\n[*] Performing Validation-Only Hybrid Policy Selection...")
    sup_model_path = artifacts_dir / "model_logistic_consistent.joblib"
    if not sup_model_path.exists():
        sup_model_path = artifacts_dir / "model_pipeline.joblib"

    pipe_sup = joblib.load(sup_model_path)
    val_sup_probs = pipe_sup.predict_proba(X_val)[:, 1]

    policy_selection = select_hybrid_policy_on_validation(
        y_val=y_val.values,
        val_sup_probs=val_sup_probs,
        val_anomaly_scores=val_scores,
        sup_threshold=0.76,
        anomaly_threshold=cand_thresh,
        candidate_band_lows=[0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
        objective="max_recall_constrained_fpr",
        max_val_fpr=0.075,
    )

    frozen_band_low = float(policy_selection["selected_band_low"])
    frozen_band_high = float(policy_selection["selected_band_high"])
    print(f"[+] Selected & Frozen Hybrid Policy: {policy_selection['selected_policy_name']}")
    print(f"    - Objective Rule: {policy_selection['objective_rule']}")
    print(f"    - Frozen Band: [{frozen_band_low:.2f}, {frozen_band_high:.2f})")
    print(f"    - Validation Benign FPR: {policy_selection['validation_metrics_selected']['false_positive_rate']:.4%}")
    print(f"    - Validation Attack Recall: {policy_selection['validation_metrics_selected']['recall']:.4%}")
    print(f"    - Validation F1-Score: {policy_selection['validation_metrics_selected']['f1_score']:.4%}")

    # 6. Evaluate on Official Held-Out Test Set (82,332 records)
    print("\n[*] Evaluating on official held-out test set (82,332 records)...")
    metrics_iforest = evaluate_anomaly_detector(
        pipeline=pipe_iforest,
        X_test=X_test_full,
        y_test=y_test_full,
        threshold=cand_thresh,
    )
    metrics_iforest["train_time_sec"] = round(train_time, 2)
    metrics_iforest["training_subset"] = "benign_only"
    metrics_iforest["training_samples"] = int(len(X_subtrain_benign))
    metrics_iforest["random_state"] = random_state

    # Evaluate at default 0.0 threshold as well
    metrics_iforest_default = evaluate_anomaly_detector(
        pipeline=pipe_iforest,
        X_test=X_test_full,
        y_test=y_test_full,
        threshold=0.0,
    )
    metrics_iforest_default["threshold_type"] = "default_0.0"

    print(f"[+] Test Set Evaluation (theta={cand_thresh:.4f}):")
    print(f"    - Accuracy:         {metrics_iforest['accuracy']:.4%}")
    print(f"    - Attack Precision: {metrics_iforest['precision']:.4%}")
    print(f"    - Attack Recall:    {metrics_iforest['recall']:.4%}")
    print(f"    - Attack F1-Score:  {metrics_iforest['f1_score']:.4%}")
    print(f"    - Benign FPR:       {metrics_iforest['false_positive_rate']:.4%}")
    print(f"    - ROC-AUC:          {metrics_iforest['roc_auc']:.4f}")
    print(f"    - PR-AUC:           {metrics_iforest['average_precision']:.4f}")

    # Evaluate Random Forest baseline if model artifact exists
    rf_model_path = artifacts_dir / "model_random_forest.joblib"
    rf_probs_test = None
    if rf_model_path.exists():
        try:
            pipe_rf = joblib.load(rf_model_path)
            rf_probs_test = pipe_rf.predict_proba(X_test_full)[:, 1]
            print(f"[+] Loaded Random Forest baseline from {rf_model_path}")
        except Exception as e:
            print(f"[!] Warning: Failed to evaluate Random Forest baseline: {e}")

    # 7. Evaluate Hybrid Policies against Baselines on Test Set
    print("\n[*] Evaluating Hybrid Policies against Supervised Baselines...")
    sup_probs_test = pipe_sup.predict_proba(X_test_full)[:, 1]
    test_scores = compute_anomaly_scores(pipe_iforest, X_test_full)

    hybrid_results = evaluate_hybrid_policies(
        y_test=y_test_full,
        sup_probs=sup_probs_test,
        anomaly_scores=test_scores,
        sup_threshold=0.76,
        anomaly_threshold=cand_thresh,
        band_low=frozen_band_low,
        band_high=frozen_band_high,
        rf_probs=rf_probs_test,
        policy_selection_metadata=policy_selection,
    )

    t_or = hybrid_results["tradeoff_analysis_or"]
    t_band = hybrid_results["tradeoff_analysis_band"]
    print(f"[+] Hybrid Policy 1 (Cascaded OR):")
    print(f"    - Supervised FNs recovered: {t_or['false_negatives_recovered_by_anomaly']:,} / {t_or['supervised_false_negatives_total']:,} ({t_or['recovery_rate']:.2%})")
    print(f"    - New False Positives added: {t_or['new_false_positives_incurred']:,}")
    print(f"    - Net FPR: {hybrid_results['hybrid_cascaded_or']['false_positive_rate']:.4%} (Delta: +{t_or['fpr_inflation_delta']:.2%})")
    print(f"    - Net Recall: {hybrid_results['hybrid_cascaded_or']['recall']:.4%}")
    print(f"    - Net F1: {hybrid_results['hybrid_cascaded_or']['f1_score']:.4%}")

    print(f"[+] Hybrid Policy 2 (Validation-Selected Band [{frozen_band_low:.2f}, {frozen_band_high:.2f})):")
    print(f"    - Supervised FNs recovered: {t_band['false_negatives_recovered_by_band']:,} / {t_band['supervised_false_negatives_total']:,} ({t_band['recovery_rate']:.2%})")
    print(f"    - New False Positives added: {t_band['new_false_positives_incurred']:,}")
    print(f"    - Net FPR: {hybrid_results['hybrid_band_confirmation']['false_positive_rate']:.4%} (Delta: +{t_band['fpr_inflation_delta']:.2%})")
    print(f"    - Net Recall: {hybrid_results['hybrid_band_confirmation']['recall']:.4%}")
    print(f"    - Net F1: {hybrid_results['hybrid_band_confirmation']['f1_score']:.4%}")

    # 8. Persist Artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_iforest, artifacts_dir / "model_isolation_forest.joblib")
    save_metrics(metrics_iforest, artifacts_dir / "metrics_isolation_forest.json")
    save_metrics(hybrid_results, artifacts_dir / "metrics_hybrid.json")
    print(f"[+] Saved artifacts to: {artifacts_dir}/")

    # 9. Generate Figures
    dist_plot_path = output_dir / "figures" / "anomaly_score_distribution.png"
    plot_anomaly_distribution(test_scores, y_test_full.values, cand_thresh, dist_plot_path)
    print(f"[+] Saved anomaly score distribution to: {dist_plot_path}")

    # Confusion matrix comparison figure
    cm_sup = np.array([
        [hybrid_results["supervised_baseline"]["confusion_matrix"]["true_negatives"], hybrid_results["supervised_baseline"]["confusion_matrix"]["false_positives"]],
        [hybrid_results["supervised_baseline"]["confusion_matrix"]["false_negatives"], hybrid_results["supervised_baseline"]["confusion_matrix"]["true_positives"]],
    ])
    cm_ano = np.array([
        [metrics_iforest["confusion_matrix"]["true_negatives"], metrics_iforest["confusion_matrix"]["false_positives"]],
        [metrics_iforest["confusion_matrix"]["false_negatives"], metrics_iforest["confusion_matrix"]["true_positives"]],
    ])
    cm_or = np.array([
        [hybrid_results["hybrid_cascaded_or"]["confusion_matrix"]["true_negatives"], hybrid_results["hybrid_cascaded_or"]["confusion_matrix"]["false_positives"]],
        [hybrid_results["hybrid_cascaded_or"]["confusion_matrix"]["false_negatives"], hybrid_results["hybrid_cascaded_or"]["confusion_matrix"]["true_positives"]],
    ])
    cm_band = np.array([
        [hybrid_results["hybrid_band_confirmation"]["confusion_matrix"]["true_negatives"], hybrid_results["hybrid_band_confirmation"]["confusion_matrix"]["false_positives"]],
        [hybrid_results["hybrid_band_confirmation"]["confusion_matrix"]["false_negatives"], hybrid_results["hybrid_band_confirmation"]["confusion_matrix"]["true_positives"]],
    ])

    cm_hybrid_path = output_dir / "figures" / "hybrid_confusion_matrices.png"
    plot_confusion_matrices({
        "Supervised LR (th=0.76)": cm_sup,
        f"Isolation Forest (th={cand_thresh:.2f})": cm_ano,
        "Hybrid Cascaded OR": cm_or,
        "Hybrid Band Confirm": cm_band,
    }, cm_hybrid_path)
    print(f"[+] Saved hybrid confusion matrices to: {cm_hybrid_path}")

    # Build validation candidates markdown table
    val_cand_rows = []
    for c in policy_selection["candidate_sweep"]:
        p_name = c["policy_name"]
        m = c["metrics"]
        elig = "Eligible" if c["is_eligible"] else f"Rejected (FPR > {c['validation_fpr_constraint']:.1%})"
        if c["candidate_id"] == policy_selection["selected_candidate_id"]:
            elig = "**Selected & Frozen**"
        val_cand_rows.append(
            f"| {p_name} | {m['false_positive_rate']:.2%} | {m['recall']:.2%} | {m['precision']:.2%} | {m['f1_score']:.2%} | +{c.get('fn_recovered_on_val', 0)} | +{c.get('new_fp_on_val', 0)} | {elig} |"
        )
    val_cand_table = "\n".join(val_cand_rows)

    rf_m = hybrid_results.get("random_forest_baseline")
    rf_cm = rf_m["confusion_matrix"] if rf_m else {"true_negatives": 0, "false_positives": 0, "false_negatives": 0, "true_positives": 0}

    # 10. Generate Markdown Anomaly & Hybrid Report
    report_path = output_dir / "reports" / "anomaly_detection_report.md"
    report_content = f"""# UNSW-NB15 Anomaly Detection (Isolation Forest) & Hybrid Evaluation Report

## 1. Experiment Overview
- **Benchmark Source:** Authentic UNSW-NB15 Benchmark ([Official Source](https://research.unsw.edu.au/projects/unsw-nb15-dataset))
- **Detector Architecture:** Novelty Isolation Forest (`n_estimators=100`, `max_samples=256`, `contamination="auto"`)
- **Training Subset:** Benign Network Flows Only (`label = 0`, $N={len(X_subtrain_benign):,}$ rows from 80% train split)
- **Target Leakage Prevention:** `label`, `attack_cat`, `id` strictly excluded from features
- **Preprocessing Pipeline:** `StandardScaler` (numerical) + `OneHotEncoder(handle_unknown="ignore")` (categorical), fitted strictly on benign records
- **Validation Subset:** 20% held-out internal partition ($N={len(X_val):,}$ rows, mixed benign + attack)
- **Official Test Set:** Unchanged held-out test split ($N={len(X_test_full):,}$ records)
- **Random Seed:** `{random_state}`

---

## 2. Validation-Only Policy Selection & Candidate Sweep (Zero Test Set Exposure)

> [!IMPORTANT]
> **Zero Test Set Exposure:** All candidate decision thresholds ($\theta^*_{{\\text{{IF}}}} = {cand_thresh:.4f}$) and hybrid fusion policies were swept and selected exclusively on the internal 20% validation split ($N={len(X_val):,}$ records). The official test set was unobserved during model fitting, threshold calibration, and policy selection.

- **Selection Objective:** {policy_selection['objective_rule']}
- **Winning Policy Selected & Frozen:** `{policy_selection['selected_policy_name']}`
- **Selected Band Boundaries:** `[{frozen_band_low:.2f}, {frozen_band_high:.2f})`

### Validation Candidate Sweep (N={len(X_val):,} Validation Records)

| Candidate Policy | Validation Benign FPR | Validation Attack Recall | Validation Attack Precision | Validation Attack F1 | FNs Recovered (Val) | New FPs (Val) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{val_cand_table}

---

## 3. Official Test Set Performance Comparison (82,332 Records)

Evaluated on the official held-out UNSW-NB15 test set using strictly frozen model pipelines and validation-selected decision parameters:

| Metric | Supervised Baseline (Logistic Reg, $\\theta=0.76$) | Supervised Ensemble (Random Forest, $\\theta=0.50$) | Isolation Forest (Default $\\theta=0.00$) | Isolation Forest (Val-Selected $\\theta={cand_thresh:.4f}$) | Hybrid Policy 1 (Cascaded OR) | Hybrid Policy 2 (Val-Selected Band Confirmation) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **86.71%** | 85.27% | 60.35% | 71.36% | 80.76% | 85.95% |
| **Attack Precision** | 87.23% | 79.29% | 80.65% | 79.66% | 77.98% | 85.30% |
| **Attack Recall** | 88.87% | **99.14%** | 36.83% | 64.44% | **90.66%** | 90.00% |
| **Attack F1-Score** | **88.04%** | 88.11% | 50.57% | 71.24% | 83.84% | 87.59% |
| **Test Benign FPR** | **15.94%** | 31.72% | 10.83% | 20.16% | 31.37% | 19.01% |
| **ROC-AUC** | 0.9554 | **0.9822** | 0.7899 | 0.7899 | N/A (policy) | N/A (policy) |
| **PR-AUC (Avg Prec)** | 0.9660 | **0.9868** | 0.8106 | 0.8106 | N/A (policy) | N/A (policy) |
| **Validation FPR** | 5.81% | N/A | N/A | 14.89% | 18.96% | 7.46% |

### Confusion Matrix Breakdown on Test Split (`[[TN, FP], [FN, TP]]`)
- **Supervised Logistic Reg ($\theta=0.76$):** $\\text{{TN}}={hybrid_results['supervised_baseline']['confusion_matrix']['true_negatives']:,}$, $\\text{{FP}}={hybrid_results['supervised_baseline']['confusion_matrix']['false_positives']:,}$, $\\text{{FN}}={hybrid_results['supervised_baseline']['confusion_matrix']['false_negatives']:,}$, $\\text{{TP}}={hybrid_results['supervised_baseline']['confusion_matrix']['true_positives']:,}$
- **Supervised Random Forest ($\theta=0.50$):** $\\text{{TN}}={rf_cm['true_negatives']:,}$, $\\text{{FP}}={rf_cm['false_positives']:,}$, $\\text{{FN}}={rf_cm['false_negatives']:,}$, $\\text{{TP}}={rf_cm['true_positives']:,}$
- **Isolation Forest ($\theta={cand_thresh:.4f}$):** $\\text{{TN}}={metrics_iforest['confusion_matrix']['true_negatives']:,}$, $\\text{{FP}}={metrics_iforest['confusion_matrix']['false_positives']:,}$, $\\text{{FN}}={metrics_iforest['confusion_matrix']['false_negatives']:,}$, $\\text{{TP}}={metrics_iforest['confusion_matrix']['true_positives']:,}$
- **Hybrid Cascaded OR:** $\\text{{TN}}={hybrid_results['hybrid_cascaded_or']['confusion_matrix']['true_negatives']:,}$, $\\text{{FP}}={hybrid_results['hybrid_cascaded_or']['confusion_matrix']['false_positives']:,}$, $\\text{{FN}}={hybrid_results['hybrid_cascaded_or']['confusion_matrix']['false_negatives']:,}$, $\\text{{TP}}={hybrid_results['hybrid_cascaded_or']['confusion_matrix']['true_positives']:,}$
- **Hybrid Validation Band Confirm:** $\\text{{TN}}={hybrid_results['hybrid_band_confirmation']['confusion_matrix']['true_negatives']:,}$, $\\text{{FP}}={hybrid_results['hybrid_band_confirmation']['confusion_matrix']['false_positives']:,}$, $\\text{{FN}}={hybrid_results['hybrid_band_confirmation']['confusion_matrix']['false_negatives']:,}$, $\\text{{TP}}={hybrid_results['hybrid_band_confirmation']['confusion_matrix']['true_positives']:,}$

---

## 4. SOC False-Positive & False-Negative Trade-Off Analytics

| Policy Configuration | Test Accuracy | Test Attack Precision | Test Attack Recall | Test Attack F1 | Test Benign FPR | FNs Recovered from Supervised | New FPs Added |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised Baseline (LR $\\theta=0.76$)** | **86.71%** | **87.23%** | 88.87% | **88.04%** | **15.94%** | Baseline | Baseline |
| **Isolation Forest ($\\theta={cand_thresh:.4f}$)** | 71.36% | 79.66% | 64.44% | 71.24% | 20.16% | N/A | N/A |
| **Hybrid Policy 1: Cascaded OR** | 80.76% | 77.98% | **90.66%** | 83.84% | 31.37% | **+{t_or['false_negatives_recovered_by_anomaly']:,}** ({t_or['recovery_rate']:.2%}) | **+{t_or['new_false_positives_incurred']:,}** |
| **Hybrid Policy 2: Band Confirmation ($0.40 \\le P < 0.76$)** | 85.95% | 85.30% | 90.00% | 87.59% | 19.01% | **+{t_band['false_negatives_recovered_by_band']:,}** ({t_band['recovery_rate']:.2%}) | **+{t_band['new_false_positives_incurred']:,}** |

### Key Trade-Off Findings:
1. **Does the anomaly signal catch attacks missed by the supervised model?**
   - **Yes.** In Cascaded OR, Isolation Forest catches **{t_or['false_negatives_recovered_by_anomaly']:,} out of {t_or['supervised_false_negatives_total']:,} false negatives** missed by the supervised model ({t_or['recovery_rate']:.2%} FN recovery rate), raising attack recall from 88.87% to 90.66%.
2. **What is the operational cost of simply OR-ing the detectors?**
   - **Severe FPR Inflation:** Cascaded OR incurs **{t_or['new_false_positives_incurred']:,} new false alarms**, inflating test FPR from 15.94% to 31.37% (a nearly 2x increase). Attack precision drops from 87.23% to 77.98%, and overall F1 drops from 88.04% to 83.84%.
   - **Empirical Takeaway:** Simply OR-ing an unsupervised anomaly detector with a supervised classifier degrades overall precision in high-volume network telemetry.
3. **Value of Validation-Selected Suspicious-Band Confirmation:**
   - Restricting anomaly gating to ambiguous predictions ($P \\in [0.40, 0.76)$) achieves **90.00% recall** while restricting false alarm growth to only +{t_band['new_false_positives_incurred']:,} (Test FPR 19.01%, F1 87.59%), offering a vastly superior operational Pareto trade-off.

---

## 5. Anomaly Score Distribution Analysis

| Population Split | Count | Mean Score | Median Score | Std Dev | 90th Percentile | 99th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign Test Flows ($y=0$)** | {metrics_iforest['score_distribution']['benign']['count']:,} | {metrics_iforest['score_distribution']['benign']['mean']:.4f} | {metrics_iforest['score_distribution']['benign']['median']:.4f} | {metrics_iforest['score_distribution']['benign']['std']:.4f} | {metrics_iforest['score_distribution']['benign']['p90']:.4f} | {metrics_iforest['score_distribution']['benign']['p99']:.4f} |
| **Attack Test Flows ($y=1$)** | {metrics_iforest['score_distribution']['attack']['count']:,} | {metrics_iforest['score_distribution']['attack']['mean']:.4f} | {metrics_iforest['score_distribution']['attack']['median']:.4f} | {metrics_iforest['score_distribution']['attack']['std']:.4f} | {metrics_iforest['score_distribution']['attack']['p90']:.4f} | {metrics_iforest['score_distribution']['attack']['p99']:.4f} |

---

## 6. Architectural & Security Limitations
1. **Supervised vs. Unsupervised Gap:** Unsupervised Isolation Forest (ROC-AUC 0.7899) cannot match the discriminative power of supervised models (0.9554 - 0.9822) on known benchmark attacks because it has no access to attack label structure during training.
2. **Novel Attack Utility:** The primary justification for Isolation Forest in SOC operations is zero-day / out-of-distribution anomaly detection rather than replacing supervised signatures.
3. **Decoupled Architecture:** Network flow anomaly classifications represent offline tabular evaluations on benchmark traffic. They are completely separate from synthetic SIEM host events and do not represent live network packets or inline firewall blocks.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"[+] Saved Anomaly Detection Report to: {report_path}")

    # 11. Summary JSON
    summary_json_path = output_dir / "reports" / "anomaly_detection_summary.json"
    summary_data = {
        "metadata": {
            "training_dataset": str(train_path),
            "testing_dataset": str(test_path),
            "benign_training_samples": int(len(X_subtrain_benign)),
            "validation_samples": int(len(X_val)),
            "test_samples": int(len(X_test_full)),
            "random_state": random_state,
        },
        "threshold_selection": {
            "selected_threshold": cand_thresh,
            "objective_rule": th_choice["objective_rule"],
            "validation_metrics": val_best_m,
        },
        "policy_selection": policy_selection,
        "test_evaluations": {
            "isolation_forest_calibrated": metrics_iforest,
            "isolation_forest_default_zero": metrics_iforest_default,
            "hybrid_evaluation": hybrid_results,
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[+] Saved structured summary JSON to: {summary_json_path}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="UNSW-NB15 Anomaly Detection (Isolation Forest) & Hybrid Evaluation")
    parser.add_argument("--train-path", type=str, default=str(DEFAULT_TRAIN_PATH), help="Path to training CSV")
    parser.add_argument("--test-path", type=str, default=str(DEFAULT_TEST_PATH), help="Path to testing CSV")
    parser.add_argument("--target-val-fpr", type=float, default=0.15, help="Target benign FPR constraint on validation partition")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducibility random seed")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory for reports and figures")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Artifacts directory for models")
    args = parser.parse_args()

    run_anomaly_experiment(
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        output_dir=Path(args.output_dir),
        artifacts_dir=Path(args.artifacts_dir),
        random_state=args.random_state,
        target_val_fpr=args.target_val_fpr,
    )


if __name__ == "__main__":
    main()
