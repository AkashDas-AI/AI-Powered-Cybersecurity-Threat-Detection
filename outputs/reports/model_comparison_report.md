# UNSW-NB15 Authentic Benchmark Model Comparison & Audit Report

## 1. Experiment Overview
- **Benchmark Source:** Authentic UNSW-NB15 Benchmark ([Official Source](https://research.unsw.edu.au/projects/unsw-nb15-dataset))
- **Training Dataset:** `data\raw\UNSW_NB15_training-set.csv` (175,341 rows)
- **Testing Dataset:** `data\raw\UNSW_NB15_testing-set.csv` (82,332 rows)
- **Target Variable:** Binary `label` (`0 = Normal`, `1 = Attack`)
- **Random Seed:** `42`

---

## 2. Head-to-Head Performance Comparison (Official Test Set)

| Evaluation Metric | Logistic Reg (Default th=0.50) | Logistic Reg (Val-Tuned th=0.76) | Logistic Reg (OOF-Calibrated th=0.77) | Logistic Reg (Consistent Frozen th=0.76) | Random Forest (Default th=0.50) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **80.9892%** | **86.8387%** | **87.1387%** | **86.7087%** | **85.2694%** |
| **Attack Precision** | 75.3467% | 87.1917% | 88.1180% | 87.2280% | 79.2928% |
| **Attack Recall** | 97.3132% | 89.1997% | 88.5864% | 88.8732% | 99.1353% |
| **Attack F1-Score** | 84.9326% | 88.1842% | 88.3516% | 88.0429% | 88.1107% |
| **False Positive Rate (FPR)** | 39.0108% | 16.0541% | 14.6351% | 15.9432% | 31.7189% |
| **ROC-AUC** | 0.9561 | 0.9561 | 0.9561 | 0.9554 | 0.9822 |
| **Average Precision (PR-AUC)** | 0.9664 | 0.9664 | 0.9664 | 0.9660 | 0.9868 |
| **Training Time** | 8.38 s | 8.38 s | 8.38 s | 8.15 s | 9.73 s |
| **Evaluation Time** | 0.20 s | 0.20 s | 0.20 s | 0.26 s | 0.49 s |

---

## 3. Confusion Matrix Breakdown

### Scikit-Learn Class Ordering & Axes
- **Rows:** Ground Truth Class (`Row 0: Normal (0)`, `Row 1: Attack (1)`)
- **Columns:** Model Prediction (`Col 0: Normal (0)`, `Col 1: Attack (1)`)
- **Formulation:** `[[TN, FP], [FN, TP]]`

| Model Variant | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (th=0.50)** | 22,566 | 14,434 | 1,218 | 44,114 |
| **Logistic Regression (Validation th=0.76)** | 31,060 | 5,940 | 4,896 | 40,436 |
| **Logistic Regression (OOF th=0.77)** | 31,585 | 5,415 | 5,174 | 40,158 |
| **Logistic Regression Consistent (th=0.76)** | 31,101 | 5,899 | 5,044 | 40,288 |
| **Random Forest (th=0.50)** | 25,264 | 11,736 | 392 | 44,940 |

---

## 4. Threshold Selection Rigor & Parameter Invariance

### A. OOF vs. Refit Threshold Selection Analysis
In Phase 6.5, Out-Of-Fold (OOF) cross-validation computed probabilities across 5 folds (each fold trained on 80% of data, $N=140,272$). The threshold $\theta^*_{\text{oof}} = 0.77$ was selected on those aggregated fold probabilities.
However, the test-time model was refitted on 100% of the training set ($N=175,341$).
**Why OOF threshold selection does NOT guarantee identical probability scale on full refit:**
1. **Model Parameter Non-Identity:** The 5 fold models produced the threshold-selection probabilities, but an entirely different 6th model (fitted on 100% of data) was deployed to the test set.
2. **Sample Size Scaling:** Increasing training sample size shifts the empirical log-loss landscape, altering coefficient norms and the intercept $b$, which can shift the sigmoid inflection point.
3. **Empirical Evidence:** On the validation fold predictions, the selected threshold achieved a validation recall of ~90.04% and FPR of ~13.97%. When the 100% refitted model was evaluated on the test set at that threshold, recall was 88.59% and FPR was 14.64%. The test labels were never observed during threshold selection, but subtle scale shifts occurred.

### B. The Parameter-Invariant Solution: Frozen 80/20 Train/Validation Split (Strategy A)
To guarantee exact parameter invariance:
1. The training split is partitioned into 80% training ($N=140,272$) and 20% validation ($N=35,069$).
2. The pipeline is fitted strictly on the 80% partition (`pipe_consistent`).
3. Threshold $\theta^*_{\text{consistent}} = 0.76$ is selected on the held-out 20% validation split.
4. **Deployment Rule:** That **exact same fitted pipeline** is serialized to `artifacts/model_logistic_consistent.joblib` and evaluated on the official test set.
5. **Guarantee:** Because no refit occurs after threshold selection, the same frozen model parameters were retained between threshold selection and evaluation.

---

## 5. Architectural Comparison Insights
1. **Linear Baseline (Logistic Regression):**
   - High attack sensitivity (97.31% recall at default threshold) but elevated false alarm rate (39.01% FPR).
   - Rapid convergence (8.38s) and high interpretability.
   - Threshold selection to 0.76 cuts FPR down from 39.01% to 15.94%, preventing **8,523 false alarms**.
2. **Non-linear Ensemble (Random Forest):**
   - Superior overall discrimination with complex non-linear feature combinations (ROC-AUC 0.9822).
   - Exceptional attack recall (99.14%) catching almost all attacks in the test split.
   - Requires additional training time (9.73s) and memory footprint.
