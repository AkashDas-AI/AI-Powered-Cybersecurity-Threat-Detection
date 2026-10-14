# Quickstart and Demonstration Checklist

This checklist guides reviewers, recruiters, and engineering interviewers through reproducing and verifying the AI-Powered Cybersecurity Threat Detection prototype end-to-end.

---

## 1. Prerequisites
- Python 3.10 or higher installed.
- PowerShell (Windows) or bash (macOS/Linux) terminal.
- Git command-line client.

---

## 2. Step-by-Step Execution Checklist

### Step 1: Environment Setup
```powershell
# Create isolated virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt
```

---

### Step 2: Automated Test Suite Execution
Verify that all 63 unit and integration tests pass cleanly:
```powershell
pytest -q
```
Expected result: `63 passed in ~8s`.

---

### Step 3: Dataset Inspection via CLI
Inspect dataset properties, schema types, and class balance:
```powershell
# Inspect synthetic host security event telemetry:
python src/inspect_data.py data/sample/synthetic_security_events.csv

# Inspect UNSW-NB15 sample demonstration dataset:
python src/inspect_data.py data/sample/unsw_sample.csv
```

---

### Step 4: Supervised Model Pipeline Training

```powershell
# Fast demonstration run on sample data (Logistic Regression baseline):
python src/train.py --sample

# Optional: Fast demonstration run with Random Forest:
python src/train.py --sample --classifier random_forest

# Full benchmark run (requires UNSW-NB15 CSVs in data/raw/):
python src/train.py
```
Generated artifacts:
- Demonstration run: `artifacts/model_sample.joblib`, `artifacts/metrics_sample.json`.
- Full benchmark run: `artifacts/model_pipeline.joblib`, `artifacts/metrics.json`.

---

### Step 5: Synthetic Host Event Simulation
Simulate controlled fictional security event telemetry:
```powershell
python src/simulate.py --count 150 --output data/sample/synthetic_security_events.csv
```
Generated output: `data/sample/synthetic_security_events.csv`.

---

### Step 6: Deterministic SIEM Rule Evaluation
Evaluate deterministic security rules against the simulated host telemetry:
```powershell
python src/detect_rules.py --input data/sample/synthetic_security_events.csv --output outputs/alerts/detected_alerts.csv
```
Generated output: `outputs/alerts/detected_alerts.csv`.

---

### Step 7: Launch Interactive Streamlit Dashboard
Launch the visual SOC triage dashboard:
```powershell
streamlit run app.py
```
- Open browser to `http://localhost:8501`.
- Inspect the five interface views:
  1. System Overview: Pipeline architecture and security boundaries.
  2. ML Model Evaluation: Head-to-head metrics, confusion matrices, ROC curves, and PR curves.
  3. Anomaly Detection & Hybrid Defense: Isolation Forest trade-off analytics and candidate sweep results.
  4. Rule-Based SIEM Alerts: Multi-attribute alert filtering and evidence inspection.
  5. Dataset & Telemetry Explorer: Interactive data table viewer.
