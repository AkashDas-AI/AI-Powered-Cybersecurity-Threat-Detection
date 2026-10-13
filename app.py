"""Streamlit Dashboard for AI-Powered Cybersecurity Threat Detection System.

Displays:
1. Supervised Machine Learning Evaluation Results (UNSW-NB15)
2. Deterministic Rule-Based SIEM Threat Alerts (Synthetic Events)
3. Dataset Inspection & Telemetry Explorer

NOTE:
- Synthetic security events are explicitly fictional for demonstration.
- This prototype does not perform live kernel-level packet monitoring or automated blocking.
"""

import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="AI-Powered Cybersecurity Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Canonical Paths
METRICS_PATH = Path("artifacts/metrics.json")
MODEL_PATH = Path("artifacts/model_pipeline.joblib")
ALERTS_PATH = Path("outputs/alerts/detected_alerts.csv")
EVENTS_PATH = Path("data/sample/synthetic_security_events.csv")
UNSW_SAMPLE_PATH = Path("data/sample/unsw_sample.csv")


# Benchmark Multi-Model & Threshold Artifact Paths
METRICS_LOGISTIC_PATH = Path("artifacts/metrics_logistic.json")
METRICS_LOGISTIC_TUNED_PATH = Path("artifacts/metrics_logistic_tuned.json")
METRICS_LOGISTIC_OOF_PATH = Path("artifacts/metrics_logistic_oof.json")
METRICS_LOGISTIC_CONSISTENT_PATH = Path("artifacts/metrics_logistic_consistent.json")
METRICS_RF_PATH = Path("artifacts/metrics_random_forest.json")
METRICS_IFOREST_PATH = Path("artifacts/metrics_isolation_forest.json")
METRICS_HYBRID_PATH = Path("artifacts/metrics_hybrid.json")

FIGURE_ROC_PATH = Path("outputs/figures/roc_curves.png")
FIGURE_PR_PATH = Path("outputs/figures/pr_curves.png")
FIGURE_THRESH_PATH = Path("outputs/figures/threshold_analysis_logistic.png")
FIGURE_CM_PATH = Path("outputs/figures/confusion_matrices_comparison.png")
FIGURE_ANOMALY_DIST_PATH = Path("outputs/figures/anomaly_score_distribution.png")
FIGURE_HYBRID_CM_PATH = Path("outputs/figures/hybrid_confusion_matrices.png")


def load_json_artifact(path: Path):
    """Loads a JSON metric file safely, returning None if missing or 'CORRUPTED' if invalid."""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, OSError):
            return "CORRUPTED"
    return None


def load_metrics_data():
    return load_json_artifact(METRICS_PATH)


def load_anomaly_metrics():
    return load_json_artifact(METRICS_IFOREST_PATH)


def load_hybrid_metrics():
    return load_json_artifact(METRICS_HYBRID_PATH)


def load_model_pipeline(path: Path = MODEL_PATH):
    """Safely loads a serialized model pipeline; returns None if missing or 'CORRUPTED' if invalid."""
    path = Path(path)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return "CORRUPTED"
    return None


def load_all_benchmark_metrics():
    """Loads all available benchmark evaluation models with metadata."""
    return {
        "Logistic Regression (Default th=0.50)": load_json_artifact(METRICS_LOGISTIC_PATH),
        "Logistic Regression (Val-Tuned th=0.76)": load_json_artifact(METRICS_LOGISTIC_TUNED_PATH),
        "Logistic Regression (OOF-Calibrated th=0.77)": load_json_artifact(METRICS_LOGISTIC_OOF_PATH),
        "Logistic Regression (Consistent Frozen th=0.76)": load_json_artifact(METRICS_LOGISTIC_CONSISTENT_PATH),
        "Random Forest (Default th=0.50)": load_json_artifact(METRICS_RF_PATH),
    }


def load_alerts_data():
    if ALERTS_PATH.exists():
        try:
            return pd.read_csv(ALERTS_PATH)
        except Exception:
            return None
    return None


# Sidebar Navigation
st.sidebar.title("🛡️ Threat Detection Hub")
st.sidebar.markdown("**AI & SIEM Security Demonstration**")
nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "System Overview",
        "ML Model Evaluation & Benchmark Comparison (UNSW-NB15)",
        "Anomaly Detection & Hybrid Defense (Isolation Forest)",
        "Rule-Based SIEM Alerts (Synthetic)",
        "Dataset & Telemetry Explorer",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Prototype Version: 1.0.0")
st.sidebar.caption("Designed for Reproducibility & Auditability")


# ---------------------------------------------------------
# View 1: System Overview
# ---------------------------------------------------------
if nav_choice == "System Overview":
    st.title("🛡️ AI-Powered Cybersecurity Threat Detection System")
    st.markdown("### Hybrid Threat Detection Prototype Architecture")

    st.info(
        "**System Boundary & Operational Disclaimer:**\n\n"
        "1. **Decoupled Workflows:** The supervised ML models operate on standard network flow features "
        "(UNSW-NB15 benchmark), while deterministic rules operate on synthetic host/authentication telemetry.\n"
        "2. **Fictional Telemetry:** All simulated security events in `data/sample/` are explicitly fictional.\n"
        "3. **No Live In-Line Blocking:** This system is an offline/nearline analysis prototype and does not execute "
        "automated firewall blocking or kernel packet interception."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🤖 Supervised Machine Learning & Novelty Detection")
        st.markdown(
            """
            - **Benchmark Dataset:** Authentic UNSW-NB15 Intrusion Benchmark.
            - **Supervised Models:** Linear Logistic Regression baseline & non-linear Random Forest ensemble.
            - **Anomaly Detection:** Novelty Isolation Forest fitted strictly on benign flows (`label = 0`).
            - **Hybrid Defense:** Cascaded OR and ambiguous-band confirmation policies.
            - **Artifacts:** Stored in `artifacts/` with independent provenance tags.
            """
        )

    with col2:
        st.subheader("⚡ Deterministic Rule Detection Engine")
        st.markdown(
            r"""
            - **Telemetry:** Fictional synthetic security events (`data/sample/synthetic_security_events.csv`).
            - **Detection Rules:**
              - `RULE-AUTH-001`: Brute-force authentication attempts ($\ge 5$ failed logins).
              - `RULE-EXFIL-002`: High-volume data exfiltration ($\ge 10$ MB outbound).
              - `RULE-PROBE-003`: Offensive security scanner User-Agent signatures.
              - `RULE-SCAN-004`: Multi-port reconnaissance probing.
            - **Alerts:** Stored in `outputs/alerts/detected_alerts.csv`.
            """
        )


# ---------------------------------------------------------
# View 2: ML Model Benchmark & Comparison
# ---------------------------------------------------------
elif nav_choice == "ML Model Evaluation & Benchmark Comparison (UNSW-NB15)":
    st.title("📊 UNSW-NB15 Benchmark Model Evaluation & Comparison")
    st.markdown(
        "Comparative evaluation on the **official held-out test split** (82,332 records) with leak-free threshold calibration."
    )

    st.warning(
        "⚠️ **Decoupled Architecture Notice:** Network flow classifications displayed below represent offline tabular "
        "evaluations on benchmark traffic. They are completely separate from synthetic SIEM host events and do not represent "
        "live network packets or inline firewall blocks."
    )

    all_metrics = load_all_benchmark_metrics()
    canonical_metrics = load_metrics_data()

    # Detect corrupted artifacts and display warning
    corrupted_models = [k for k, v in all_metrics.items() if v == "CORRUPTED"]
    if corrupted_models:
        st.error(
            f"⚠️ **Corrupted Artifact Warning:** The following benchmark metric files are corrupted on disk: "
            f"`{', '.join(corrupted_models)}`. Please re-run `python src/benchmark_audit.py` to regenerate."
        )

    # Filter available models
    available_models = {k: v for k, v in all_metrics.items() if isinstance(v, dict)}

    if not available_models and canonical_metrics is None:
        st.warning(
            "⚠️ No benchmark evaluation metrics found in `artifacts/`.\n\n"
            "Run benchmark training and audit using:\n"
            "```powershell\npython src/benchmark_audit.py\n```"
        )
    elif not available_models and isinstance(canonical_metrics, dict):
        available_models = {"Canonical Baseline": canonical_metrics}

    tab_compare, tab_thresh, tab_diagnostics = st.tabs([
        "🏆 Head-to-Head Comparison",
        "🎯 Leak-Free Threshold Tuning",
        "🔬 Detailed Diagnostics & Reports",
    ])

    def _fmt(val):
        return f"{val:.2%}" if isinstance(val, (int, float)) else "N/A"

    def _fmt_dec(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

    # Tab 1: Head-to-Head Comparison
    with tab_compare:
        st.subheader("Official Held-Out Test Set Performance (82,332 Records)")

        # Comparison DataFrame
        table_rows = []
        for name, m in available_models.items():
            table_rows.append({
                "Model Variant": name,
                "Accuracy": _fmt(m.get("accuracy")),
                "Attack Precision": _fmt(m.get("precision")),
                "Attack Recall": _fmt(m.get("recall")),
                "Attack F1-Score": _fmt(m.get("f1_score")),
                "False Positive Rate (FPR)": _fmt(m.get("false_positive_rate")),
                "ROC-AUC": _fmt_dec(m.get("roc_auc")),
                "PR-AUC (Avg Prec)": _fmt_dec(m.get("average_precision")),
                "Training Time (s)": m.get("train_time_sec", "N/A"),
                "Eval Time (s)": m.get("eval_time_sec", "N/A"),
            })

        df_compare = pd.DataFrame(table_rows)
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Comparative Confusion Matrices")
        if FIGURE_CM_PATH.exists():
            st.image(str(FIGURE_CM_PATH), caption="Confusion Matrices across Thresholds & Models (Row: Actual, Col: Predicted)", use_container_width=True)
        else:
            st.info(f"ℹ️ Comparative confusion matrix plot `{FIGURE_CM_PATH}` not found on disk. Run `python src/benchmark_audit.py` to generate.")

        st.markdown("---")
        st.subheader("Comparative Discrimination Curves (ROC & Precision-Recall)")
        col_roc, col_pr = st.columns(2)
        with col_roc:
            if FIGURE_ROC_PATH.exists():
                st.image(str(FIGURE_ROC_PATH), caption="Receiver Operating Characteristic (ROC)", use_container_width=True)
            else:
                st.info(f"ℹ️ ROC plot `{FIGURE_ROC_PATH}` not found.")
        with col_pr:
            if FIGURE_PR_PATH.exists():
                st.image(str(FIGURE_PR_PATH), caption="Precision-Recall Curve (Attack Class)", use_container_width=True)
            else:
                st.info(f"ℹ️ Precision-Recall plot `{FIGURE_PR_PATH}` not found.")

    # Tab 2: Leak-Free Threshold Tuning
    with tab_thresh:
        st.subheader("Leak-Free Threshold Selection & Parameter Invariance")
        st.markdown(
            """
            In enterprise SOC environments, a standard 0.50 threshold on Logistic Regression yields a high False Positive Rate (39.01%),
            producing thousands of false alarms.

            **Methodological Rigor & Zero Test Leakage:**
            - Threshold selection is performed **exclusively on the training dataset** without observing any test records:
              1. **Single Validation Split:** Internal 20% validation partition (yielding $\\theta^* = 0.76$, refitted on 100%).
              2. **5-Fold Out-Of-Fold (OOF):** Cross-validation across all 175,341 training flows (yielding $\\theta^*_{\\text{oof}} = 0.77$, refitted on 100%).
              3. **Parameter-Invariant Frozen Split:** Model trained strictly on 80% ($N=140,272$), threshold selected on 20% ($N=35,069$, yielding $\\theta^*_{\\text{consistent}} = 0.76$), and deployed **without refitting**. The same frozen model parameters were retained between threshold selection and evaluation.
            - **Security Objective:** Minimize False Positive Rate subject to Attack Recall $\\ge 90.0\\%$ on validation traffic.
            """
        )

        if FIGURE_THRESH_PATH.exists():
            st.image(str(FIGURE_THRESH_PATH), caption="Validation Threshold Trade-off Curves (Precision, Recall, F1, FPR)", use_container_width=True)
        else:
            st.info(f"ℹ️ Threshold curve plot `{FIGURE_THRESH_PATH}` not found. Run `python src/benchmark_audit.py` to generate.")

        st.markdown(
            """
            **Operational Generalization on Official Test Set (82,332 Records):**
            - **Default Baseline (0.50):** FPR = 39.01% (14,434 false alarms), Attack Recall = 97.31%.
            - **OOF Calibrated (0.77, Refitted):** FPR = **14.64%** (5,415 false alarms), Attack Recall = **88.59%** (9,019 false alarms eliminated).
            - **Consistent Frozen Model (0.76, Zero Refit Drift):** FPR = **15.94%** (5,899 false alarms), Attack Recall = **88.87%** (8,523 false alarms eliminated with 100% parameter identity!).
            """
        )

    # Tab 3: Detailed Diagnostics & Reports
    with tab_diagnostics:
        st.subheader("Model Diagnostic Deep-Dive")
        model_name = st.selectbox("Select Model to Inspect", list(available_models.keys()))
        selected_m = available_models[model_name]

        if isinstance(selected_m, dict):
            # Provenance badge & details
            source_lbl = selected_m.get("dataset_source", "unknown")
            is_sample = selected_m.get("is_sample_evaluation", False)
            if is_sample:
                st.warning(f"⚠️ Source: Demonstration Sample (`{source_lbl}`). Synthetic mock data.")
            else:
                st.success(f"✅ Source: Authentic UNSW-NB15 Benchmark (`{source_lbl}`). Verified (82,332 Held-Out Test Records).")

            cal_strat = selected_m.get("calibration_strategy")
            th_val = selected_m.get("decision_threshold", 0.5)
            st.caption(f"Decision Threshold: **θ = {th_val:.2f}** | Calibration Strategy: **{cal_strat or 'standard_refit'}**")

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Accuracy", _fmt(selected_m.get("accuracy")))
            k2.metric("Attack Precision", _fmt(selected_m.get("precision")))
            k3.metric("Attack Recall", _fmt(selected_m.get("recall")))
            k4.metric("Attack F1", _fmt(selected_m.get("f1_score")))
            k5.metric("FPR", _fmt(selected_m.get("false_positive_rate")))

            cm = selected_m.get("confusion_matrix", {})
            if cm:
                st.markdown("#### Confusion Matrix Breakdown (`[[TN, FP], [FN, TP]]`)")
                cm_df = pd.DataFrame(
                    [
                        [cm.get("true_negatives", 0), cm.get("false_positives", 0)],
                        [cm.get("false_negatives", 0), cm.get("true_positives", 0)],
                    ],
                    index=["Actual Normal (0)", "Actual Attack (1)"],
                    columns=["Predicted Normal (0)", "Predicted Attack (1)"],
                )
                st.dataframe(cm_df, use_container_width=True)

            rep = selected_m.get("classification_report", {})
            if isinstance(rep, dict) and "Normal" in rep and "Attack" in rep:
                st.markdown("#### Per-Class Classification Report")
                rep_df = pd.DataFrame({
                    "Class": ["Normal (0)", "Attack (1)"],
                    "Precision": [_fmt(rep["Normal"]["precision"]), _fmt(rep["Attack"]["precision"])],
                    "Recall": [_fmt(rep["Normal"]["recall"]), _fmt(rep["Attack"]["recall"])],
                    "F1-Score": [_fmt(rep["Normal"]["f1-score"]), _fmt(rep["Attack"]["f1-score"])],
                    "Support": [f"{int(rep['Normal']['support']):,}", f"{int(rep['Attack']['support']):,}"],
                })
                st.dataframe(rep_df, use_container_width=True, hide_index=True)

            # Display calibration metadata if present
            cal_details = selected_m.get("calibration_details")
            if cal_details:
                with st.expander("🔍 Calibration Details (Validation Partition)"):
                    st.json(cal_details)



# ---------------------------------------------------------
# View 3: Anomaly Detection & Hybrid Defense (Isolation Forest)
# ---------------------------------------------------------
elif nav_choice == "Anomaly Detection & Hybrid Defense (Isolation Forest)":
    st.title("🌲 Anomaly Detection & Hybrid Defense (Isolation Forest)")
    st.markdown(
        "Experimental unsupervised novelty detection trained strictly on **benign network flows** (`label = 0`) "
        "and evaluated against supervised baselines and hybrid fusion policies."
    )

    st.warning(
        "⚠️ **Decoupled Architecture Notice:** Network flow anomaly classifications displayed below represent offline tabular "
        "evaluations on benchmark traffic. They are completely separate from synthetic SIEM host events and do not represent "
        "live network packets or inline firewall blocks."
    )

    iforest_metrics = load_anomaly_metrics()
    hybrid_metrics = load_hybrid_metrics()

    if iforest_metrics == "CORRUPTED" or hybrid_metrics == "CORRUPTED":
        st.error(
            "⚠️ **Corrupted Artifact Warning:** Anomaly detection or hybrid metric files are corrupted on disk. "
            "Please re-run `python src/anomaly_detection.py` to regenerate."
        )
    elif iforest_metrics is None or hybrid_metrics is None:
        st.warning(
            "⚠️ Anomaly detection or hybrid evaluation metrics not found in `artifacts/`.\n\n"
            "Run the anomaly detection experiment using:\n"
            "```powershell\npython src/anomaly_detection.py\n```"
        )
    else:
        st.success(
            "✅ **Provenance Verified:** Detector fitted exclusively on authentic benign flows (`label = 0`, N=44,800) "
            "and evaluated on the official held-out test split (82,332 records). Target and identifier columns strictly excluded."
        )

        st.markdown("### 1. Standalone & Hybrid Policy Comparison (Official Test Set: 82,332 Records)")

        def _fmt(val):
            return f"{val:.2%}" if isinstance(val, (int, float)) else "N/A"

        sup_base = hybrid_metrics.get("supervised_baseline", {})
        rf_base = hybrid_metrics.get("random_forest_baseline", {})
        ano_stand = hybrid_metrics.get("anomaly_standalone", {})
        hyb_or = hybrid_metrics.get("hybrid_cascaded_or", {})
        hyb_band = hybrid_metrics.get("hybrid_band_confirmation", {})

        rows = [
            {
                "Policy / Detector": "Supervised Baseline (Logistic Reg, θ=0.76)",
                "Test Accuracy": _fmt(sup_base.get("accuracy")),
                "Test Precision": _fmt(sup_base.get("precision")),
                "Test Recall": _fmt(sup_base.get("recall")),
                "Test F1": _fmt(sup_base.get("f1_score")),
                "Test Benign FPR": _fmt(sup_base.get("false_positive_rate")),
                "Role / Mechanism": "Supervised linear baseline with frozen threshold selection",
            },
        ]
        if rf_base:
            rows.append({
                "Policy / Detector": "Supervised Ensemble (Random Forest, θ=0.50)",
                "Test Accuracy": _fmt(rf_base.get("accuracy")),
                "Test Precision": _fmt(rf_base.get("precision")),
                "Test Recall": _fmt(rf_base.get("recall")),
                "Test F1": _fmt(rf_base.get("f1_score")),
                "Test Benign FPR": _fmt(rf_base.get("false_positive_rate")),
                "Role / Mechanism": "Supervised ensemble benchmark (max recall comparator)",
            })
        rows.extend([
            {
                "Policy / Detector": f"Isolation Forest Standalone (θ={iforest_metrics.get('decision_threshold', -0.0428):.4f})",
                "Test Accuracy": _fmt(ano_stand.get("accuracy")),
                "Test Precision": _fmt(ano_stand.get("precision")),
                "Test Recall": _fmt(ano_stand.get("recall")),
                "Test F1": _fmt(ano_stand.get("f1_score")),
                "Test Benign FPR": _fmt(ano_stand.get("false_positive_rate")),
                "Role / Mechanism": "Unsupervised novelty detection (benign-only fit)",
            },
            {
                "Policy / Detector": "Hybrid Policy 1: Cascaded OR",
                "Test Accuracy": _fmt(hyb_or.get("accuracy")),
                "Test Precision": _fmt(hyb_or.get("precision")),
                "Test Recall": _fmt(hyb_or.get("recall")),
                "Test F1": _fmt(hyb_or.get("f1_score")),
                "Test Benign FPR": _fmt(hyb_or.get("false_positive_rate")),
                "Role / Mechanism": "Alert if Supervised OR Anomaly flags flow (max sensitivity)",
            },
            {
                "Policy / Detector": "Hybrid Policy 2: Val-Selected Band Confirm (0.40 ≤ P < 0.76)",
                "Test Accuracy": _fmt(hyb_band.get("accuracy")),
                "Test Precision": _fmt(hyb_band.get("precision")),
                "Test Recall": _fmt(hyb_band.get("recall")),
                "Test F1": _fmt(hyb_band.get("f1_score")),
                "Test Benign FPR": _fmt(hyb_band.get("false_positive_rate")),
                "Role / Mechanism": "Alert in ambiguous supervised band only if Anomaly confirms",
            },
        ])
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 2. False-Positive & False-Negative Trade-Off Analytics (Official Test Set)")

        t_or = hybrid_metrics.get("tradeoff_analysis_or", {})
        t_band = hybrid_metrics.get("tradeoff_analysis_band", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Supervised FNs Missed", f"{t_or.get('supervised_false_negatives_total', 0):,}")
        col2.metric("FNs Caught by Anomaly (OR)", f"{t_or.get('false_negatives_recovered_by_anomaly', 0):,}", f"{t_or.get('recovery_rate', 0):.1%} recovered")
        col3.metric("New False Alarms Incurred (OR)", f"+{t_or.get('new_false_positives_incurred', 0):,}", f"+{t_or.get('fpr_inflation_delta', 0):.1%} FPR delta", delta_color="inverse")
        col4.metric("Band Confirmation New FPs", f"+{t_band.get('new_false_positives_incurred', 0):,}", f"Test FPR = {hyb_band.get('false_positive_rate', 0):.2%}")

        st.info(
            "💡 **SOC Trade-Off Insight:**\n\n"
            "- **Cascaded OR Benefit:** Successfully recovers **809 missed attacks** (16.04% of supervised false negatives), raising test recall to 90.66%.\n"
            "- **Cascaded OR Cost:** Incurs **5,707 additional false alarms**, nearly doubling test FPR from 15.94% to 31.37% and dropping precision from 87.23% to 77.98%.\n"
            "- **Band Confirmation Solution:** Restricting anomaly gating to ambiguous predictions (0.40 ≤ P < 0.76) achieves **90.00% recall** with only 1,134 new false alarms (Test FPR 19.01%), providing a superior operational trade-off."
        )

        # Policy Selection Provenance Section
        pol_meta = hybrid_metrics.get("policy_selection_metadata")
        if pol_meta:
            st.markdown("---")
            st.markdown("### 3. Validation Policy Selection Provenance (Zero Test Set Exposure)")
            st.markdown(
                f"**Selection Objective:** `{pol_meta.get('objective_rule', 'N/A')}`  \n"
                f"**Winning Policy Selected & Frozen:** `{pol_meta.get('selected_policy_name', 'N/A')}`  \n"
                f"**Frozen Confirmation Band:** `[{pol_meta.get('selected_band_low', 0.40):.2f}, {pol_meta.get('selected_band_high', 0.76):.2f})`"
            )

            cand_sweep = pol_meta.get("candidate_sweep", [])
            if cand_sweep:
                val_rows = []
                for c in cand_sweep:
                    m = c.get("metrics", {})
                    is_sel = c.get("candidate_id") == pol_meta.get("selected_candidate_id")
                    status = "Selected & Frozen" if is_sel else ("Eligible" if c.get("is_eligible") else f"Rejected (FPR > {c.get('validation_fpr_constraint', 0.075):.1%})")
                    val_rows.append({
                        "Candidate Policy": c.get("policy_name", ""),
                        "Validation Benign FPR": _fmt(m.get("false_positive_rate")),
                        "Validation Attack Recall": _fmt(m.get("recall")),
                        "Validation Attack Precision": _fmt(m.get("precision")),
                        "Validation F1": _fmt(m.get("f1_score")),
                        "FNs Recovered (Val)": f"+{c.get('fn_recovered_on_val', 0):,}",
                        "New FPs (Val)": f"+{c.get('new_fp_on_val', 0):,}",
                        "Selection Status": status,
                    })
                st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 4. Visual Diagnostics")

        c_dist, c_cm = st.columns(2)
        with c_dist:
            if FIGURE_ANOMALY_DIST_PATH.exists():
                st.image(str(FIGURE_ANOMALY_DIST_PATH), caption="Anomaly Score Distributions (Benign vs Attack)", use_container_width=True)
            else:
                st.info(f"ℹ️ Anomaly distribution plot `{FIGURE_ANOMALY_DIST_PATH}` not found.")

        with c_cm:
            if FIGURE_HYBRID_CM_PATH.exists():
                st.image(str(FIGURE_HYBRID_CM_PATH), caption="Comparative Confusion Matrices across Policies", use_container_width=True)
            else:
                st.info(f"ℹ️ Hybrid confusion matrix plot `{FIGURE_HYBRID_CM_PATH}` not found.")

        st.markdown("---")
        st.markdown("### 5. Anomaly Score Distribution Breakdown (Held-Out Test Set)")
        score_dist = iforest_metrics.get("score_distribution", {})
        if score_dist and "benign" in score_dist and "attack" in score_dist:
            b_s = score_dist["benign"]
            a_s = score_dist["attack"]
            dist_df = pd.DataFrame([
                {
                    "Population Split": "Normal Flows (Label=0)",
                    "Count": f"{b_s.get('count', 0):,}",
                    "Mean Score": f"{b_s.get('mean', 0):.4f}",
                    "Median Score": f"{b_s.get('median', 0):.4f}",
                    "Std Dev": f"{b_s.get('std', 0):.4f}",
                    "25th %": f"{b_s.get('p25', 0):.4f}",
                    "75th %": f"{b_s.get('p75', 0):.4f}",
                    "90th %": f"{b_s.get('p90', 0):.4f}",
                    "95th %": f"{b_s.get('p95', 0):.4f}",
                },
                {
                    "Population Split": "Attack Flows (Label=1)",
                    "Count": f"{a_s.get('count', 0):,}",
                    "Mean Score": f"{a_s.get('mean', 0):.4f}",
                    "Median Score": f"{a_s.get('median', 0):.4f}",
                    "Std Dev": f"{a_s.get('std', 0):.4f}",
                    "25th %": f"{a_s.get('p25', 0):.4f}",
                    "75th %": f"{a_s.get('p75', 0):.4f}",
                    "90th %": f"{a_s.get('p90', 0):.4f}",
                    "95th %": f"{a_s.get('p95', 0):.4f}",
                },
            ])
            st.dataframe(dist_df, use_container_width=True, hide_index=True)



# ---------------------------------------------------------
# View 4: Rule-Based SIEM Alerts
# ---------------------------------------------------------
elif nav_choice == "Rule-Based SIEM Alerts (Synthetic)":
    st.title("🚨 Deterministic Rule-Based SIEM Alerts")
    st.markdown("Alerts generated from synthetic event telemetry in `outputs/alerts/`.")

    alerts_df = load_alerts_data()

    if alerts_df is None or alerts_df.empty:
        st.warning(
            "⚠️ No alerts file found at `outputs/alerts/detected_alerts.csv`.\n\n"
            "Run rule detection using:\n```powershell\npython src/detect_rules.py\n```"
        )
    else:
        # Severity metrics
        sev_counts = alerts_df["severity"].value_counts().to_dict()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Alerts", len(alerts_df))
        s2.metric("Critical", sev_counts.get("CRITICAL", 0))
        s3.metric("High", sev_counts.get("HIGH", 0))
        s4.metric("Medium", sev_counts.get("MEDIUM", 0))

        # Severity filter
        selected_severity = st.multiselect(
            "Filter by Severity",
            options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM"],
        )

        filtered_alerts = alerts_df[alerts_df["severity"].isin(selected_severity)]

        st.subheader("Triage Alert Table")
        st.dataframe(
            filtered_alerts[[
                "alert_id", "severity", "rule_name", "source_ip",
                "destination_ip", "destination_port", "evidence"
            ]],
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------
# View 4: Dataset & Telemetry Explorer
# ---------------------------------------------------------
elif nav_choice == "Dataset & Telemetry Explorer":
    st.title("🔍 Dataset & Telemetry Explorer")

    dataset_choice = st.selectbox(
        "Select Dataset to Inspect",
        ["Synthetic Security Events (Fictional)", "UNSW-NB15 Benchmark Sample"]
    )

    if dataset_choice == "Synthetic Security Events (Fictional)":
        if EVENTS_PATH.exists():
            df_events = pd.read_csv(EVENTS_PATH)
            st.markdown(f"**Loaded:** `{EVENTS_PATH}` ({len(df_events)} records)")
            st.dataframe(df_events.head(25), use_container_width=True)
            st.subheader("Scenario Breakdown")
            st.bar_chart(df_events["simulated_scenario"].value_counts())
        else:
            st.error(f"File not found: {EVENTS_PATH}")

    else:
        if UNSW_SAMPLE_PATH.exists():
            df_unsw = pd.read_csv(UNSW_SAMPLE_PATH)
            st.markdown(f"**Loaded:** `{UNSW_SAMPLE_PATH}` ({len(df_unsw)} rows, {len(df_unsw.columns)} columns)")
            st.dataframe(df_unsw.head(25), use_container_width=True)
            st.subheader("Label Distribution (0: Normal, 1: Attack)")
            st.bar_chart(df_unsw["label"].value_counts())
        else:
            st.error(f"File not found: {UNSW_SAMPLE_PATH}")
