# UNSW-NB15 Anomaly Detection (Isolation Forest) & Hybrid Evaluation Report

## 1. Experiment Overview
- **Benchmark Source:** Authentic UNSW-NB15 Benchmark ([Official Source](https://research.unsw.edu.au/projects/unsw-nb15-dataset))
- **Detector Architecture:** Novelty Isolation Forest (`n_estimators=100`, `max_samples=256`, `contamination="auto"`)
- **Training Subset:** Benign Network Flows Only (`label = 0`, $N=44,800$ rows from 80% train split)
- **Target Leakage Prevention:** `label`, `attack_cat`, `id` strictly excluded from features
- **Preprocessing Pipeline:** `StandardScaler` (numerical) + `OneHotEncoder(handle_unknown="ignore")` (categorical), fitted strictly on benign records
- **Validation Subset:** 20% held-out internal partition ($N=35,069$ rows, mixed benign + attack)
- **Official Test Set:** Unchanged held-out test split ($N=82,332$ records)
- **Random Seed:** `42`

---

## 2. Validation-Only Policy Selection & Candidate Sweep (Zero Test Set Exposure)

> [!IMPORTANT]
> **Zero Test Set Exposure:** All candidate decision thresholds ($	heta^*_{\text{IF}} = -0.0428$) and hybrid fusion policies were swept and selected exclusively on the internal 20% validation split ($N=35,069$ records). The official test set was unobserved during model fitting, threshold calibration, and policy selection.

- **Selection Objective:** Maximize attack recall subject to validation benign FPR <= 7.50% (fallback=False)
- **Winning Policy Selected & Frozen:** `Hybrid Policy: Band Confirmation (0.40 <= P < 0.76)`
- **Selected Band Boundaries:** `[0.40, 0.76)`

### Validation Candidate Sweep (N=35,069 Validation Records)

| Candidate Policy | Validation Benign FPR | Validation Attack Recall | Validation Attack Precision | Validation Attack F1 | FNs Recovered (Val) | New FPs (Val) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Supervised Baseline (LR, th=0.76) | 5.81% | 90.69% | 97.08% | 93.77% | +0 | +0 | Eligible |
| Hybrid Policy: Cascaded OR | 18.96% | 92.17% | 91.20% | 91.68% | +353 | +1472 | Rejected (FPR > 7.5%) |
| Hybrid Policy: Band Confirmation (0.20 <= P < 0.76) | 8.03% | 92.00% | 96.07% | 93.99% | +314 | +248 | Rejected (FPR > 7.5%) |
| Hybrid Policy: Band Confirmation (0.30 <= P < 0.76) | 7.71% | 91.90% | 96.21% | 94.01% | +289 | +212 | Rejected (FPR > 7.5%) |
| Hybrid Policy: Band Confirmation (0.40 <= P < 0.76) | 7.46% | 91.81% | 96.33% | 94.01% | +267 | +184 | **Selected & Frozen** |
| Hybrid Policy: Band Confirmation (0.50 <= P < 0.76) | 7.17% | 91.69% | 96.46% | 94.02% | +240 | +152 | Eligible |
| Hybrid Policy: Band Confirmation (0.60 <= P < 0.76) | 6.76% | 91.52% | 96.65% | 94.02% | +199 | +106 | Eligible |
| Hybrid Policy: Band Confirmation (0.70 <= P < 0.76) | 6.38% | 91.13% | 96.82% | 93.89% | +106 | +63 | Eligible |

---

## 3. Official Test Set Performance Comparison (82,332 Records)

Evaluated on the official held-out UNSW-NB15 test set using strictly frozen model pipelines and validation-selected decision parameters:

| Metric | Supervised Baseline (Logistic Reg, $\theta=0.76$) | Supervised Ensemble (Random Forest, $\theta=0.50$) | Isolation Forest (Default $\theta=0.00$) | Isolation Forest (Val-Selected $\theta=-0.0428$) | Hybrid Policy 1 (Cascaded OR) | Hybrid Policy 2 (Val-Selected Band Confirmation) |
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
- **Supervised Logistic Reg ($	heta=0.76$):** $\text{TN}=31,101$, $\text{FP}=5,899$, $\text{FN}=5,044$, $\text{TP}=40,288$
- **Supervised Random Forest ($	heta=0.50$):** $\text{TN}=25,264$, $\text{FP}=11,736$, $\text{FN}=392$, $\text{TP}=44,940$
- **Isolation Forest ($	heta=-0.0428$):** $\text{TN}=29,541$, $\text{FP}=7,459$, $\text{FN}=16,121$, $\text{TP}=29,211$
- **Hybrid Cascaded OR:** $\text{TN}=25,394$, $\text{FP}=11,606$, $\text{FN}=4,235$, $\text{TP}=41,097$
- **Hybrid Validation Band Confirm:** $\text{TN}=29,967$, $\text{FP}=7,033$, $\text{FN}=4,531$, $\text{TP}=40,801$

---

## 4. SOC False-Positive & False-Negative Trade-Off Analytics

| Policy Configuration | Test Accuracy | Test Attack Precision | Test Attack Recall | Test Attack F1 | Test Benign FPR | FNs Recovered from Supervised | New FPs Added |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Supervised Baseline (LR $\theta=0.76$)** | **86.71%** | **87.23%** | 88.87% | **88.04%** | **15.94%** | Baseline | Baseline |
| **Isolation Forest ($\theta=-0.0428$)** | 71.36% | 79.66% | 64.44% | 71.24% | 20.16% | N/A | N/A |
| **Hybrid Policy 1: Cascaded OR** | 80.76% | 77.98% | **90.66%** | 83.84% | 31.37% | **+809** (16.04%) | **+5,707** |
| **Hybrid Policy 2: Band Confirmation ($0.40 \le P < 0.76$)** | 85.95% | 85.30% | 90.00% | 87.59% | 19.01% | **+513** (10.17%) | **+1,134** |

### Key Trade-Off Findings:
1. **Does the anomaly signal catch attacks missed by the supervised model?**
   - **Yes.** In Cascaded OR, Isolation Forest catches **809 out of 5,044 false negatives** missed by the supervised model (16.04% FN recovery rate), raising attack recall from 88.87% to 90.66%.
2. **What is the operational cost of simply OR-ing the detectors?**
   - **Severe FPR Inflation:** Cascaded OR incurs **5,707 new false alarms**, inflating test FPR from 15.94% to 31.37% (a nearly 2x increase). Attack precision drops from 87.23% to 77.98%, and overall F1 drops from 88.04% to 83.84%.
   - **Empirical Takeaway:** Simply OR-ing an unsupervised anomaly detector with a supervised classifier degrades overall precision in high-volume network telemetry.
3. **Value of Validation-Selected Suspicious-Band Confirmation:**
   - Restricting anomaly gating to ambiguous predictions ($P \in [0.40, 0.76)$) achieves **90.00% recall** while restricting false alarm growth to only +1,134 (Test FPR 19.01%, F1 87.59%), offering a vastly superior operational Pareto trade-off.

---

## 5. Anomaly Score Distribution Analysis

| Population Split | Count | Mean Score | Median Score | Std Dev | 90th Percentile | 99th Percentile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benign Test Flows ($y=0$)** | 37,000 | -0.0841 | -0.1003 | 0.0544 | 0.0041 | 0.0659 |
| **Attack Test Flows ($y=1$)** | 45,332 | -0.0144 | -0.0184 | 0.0675 | 0.0839 | 0.1231 |

---

## 6. Architectural & Security Limitations
1. **Supervised vs. Unsupervised Gap:** Unsupervised Isolation Forest (ROC-AUC 0.7899) cannot match the discriminative power of supervised models (0.9554 - 0.9822) on known benchmark attacks because it has no access to attack label structure during training.
2. **Novel Attack Utility:** The primary justification for Isolation Forest in SOC operations is zero-day / out-of-distribution anomaly detection rather than replacing supervised signatures.
3. **Decoupled Architecture:** Network flow anomaly classifications represent offline tabular evaluations on benchmark traffic. They are completely separate from synthetic SIEM host events and do not represent live network packets or inline firewall blocks.
