# Quickstart & Demo Checklist

This checklist guides reviewers, recruiters, and developers through validating the **AI-Powered Cybersecurity Threat Detection** prototype end-to-end.

---

## 📋 Prerequisites
- Python 3.10+ installed.
- PowerShell or bash terminal.

---

## 🚀 Step-by-Step Execution Checklist

### 1. Environment & Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### 2. Run Test Suite
Verify that all unit tests, rule assertions, and pipeline integrity checks pass:
```powershell
pytest -q
```
*Expected Result:* `63 passed in ~5s`.

---

### 3. Inspect Datasets
Inspect dataset properties and distributions via CLI:
```powershell
# Inspect synthetic security event telemetry:
python src/inspect_data.py data/sample/synthetic_security_events.csv

# Inspect UNSW-NB15 sample dataset:
python src/inspect_data.py data/sample/unsw_sample.csv
```

---

### 4. Train Supervised Threat Classifier
Train the Logistic Regression baseline model pipeline (or optional Random Forest) and evaluate metrics:
```powershell
# Fast demonstration run on sample data (Logistic Regression baseline):
python src/train.py --sample

# Optional: Train Random Forest on sample data:
python src/train.py --sample --classifier random_forest

# Full run (after downloading full UNSW-NB15 dataset into data/raw/):
python src/train.py
```
*Outputs generated:*
- Fast demonstration run: `artifacts/model_sample.joblib`, `artifacts/metrics_sample.json`
- Full benchmark run: `artifacts/model_pipeline.joblib`, `artifacts/metrics.json`

---

### 5. Generate Synthetic Security Events
Simulate controlled fictional security event telemetry:
```powershell
python src/simulate.py --count 150 --output data/sample/synthetic_security_events.csv
```
*Output generated:* [`data/sample/synthetic_security_events.csv`](file:///d:/Projects/Artificial%20Intelligence/AI-Powered%20Cybersecurity%20Threat%20Detection/data/sample/synthetic_security_events.csv)

---

### 6. Run Deterministic Rule-Based Detection
Evaluate deterministic security rules against the simulated events:
```powershell
python src/detect_rules.py --input data/sample/synthetic_security_events.csv --output outputs/alerts/detected_alerts.csv
```
*Output generated:* [`outputs/alerts/detected_alerts.csv`](file:///d:/Projects/Artificial%20Intelligence/AI-Powered%20Cybersecurity%20Threat%20Detection/outputs/alerts/detected_alerts.csv)

---

### 7. Launch Interactive Streamlit Dashboard
Launch the visual SOC triage dashboard:
```powershell
streamlit run app.py
```
- Open local URL (typically `http://localhost:8501`).
- Navigate between **System Overview**, **ML Model Evaluation**, **Rule-Based SIEM Alerts**, and **Dataset & Telemetry Explorer**.
