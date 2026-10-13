# System Modules & Component Specifications

## 1. Modular Directory Structure

The project follows a clean, decoupled modular layout under `src/`:

```
src/
├── config/              # Central configuration schemas & hyperparameters
│   ├── __init__.py
│   └── settings.py
├── data/                # Ingestion, validation, and stream simulation
│   ├── __init__.py
│   ├── ingestion.py
│   └── validator.py
├── features/            # Preprocessing, transformations, scaling, encoders
│   ├── __init__.py
│   ├── preprocessor.py
│   └── feature_engineer.py
├── models/              # Threat detection model implementations
│   ├── __init__.py
│   ├── base.py
│   ├── supervised.py
│   ├── anomaly.py
│   └── hybrid.py
├── training/            # Training routines, loss tracking, checkpoints
│   ├── __init__.py
│   ├── trainer.py
│   └── loss.py
├── inference/           # Offline and batch scoring pipelines
│   ├── __init__.py
│   ├── engine.py
│   └── explainer.py
├── alerting/            # Threat event dispatching and notification
│   ├── __init__.py
│   └── alert_manager.py
└── evaluation/          # Metrics, confusion matrices, evaluation reports
    ├── __init__.py
    └── metrics.py
```

---

## 2. Module Responsibilities & Key Interfaces

### Module 1: Data Ingestion (`src/data/`)
- **`ingestion.py`:**
  - Loads streaming telemetry, PCAPs, or CSV flow dumps.
  - Implements batch chunking for memory-efficient processing of multi-gigabyte datasets.
- **`validator.py`:**
  - Enforces schema consistency, detects corrupted rows, and handles out-of-bounds metrics.

### Module 2: Feature Engineering & Preprocessing (`src/features/`)
- **`preprocessor.py`:**
  - Cleans infinite values, imputes missing statistics, and filters high-cardinality noise identifiers (e.g. source IPs).
- **`feature_engineer.py`:**
  - Computes secondary traffic metrics (packet size ratios, burst rates, flag frequency).
  - Serializes fitted `StandardScaler` / `RobustScaler` and encoders to prevent distribution mismatch during production inference.

### Module 3: Models Core (`src/models/`)
- **`base.py`:** Abstract base class defining common methods: `fit()`, `predict()`, `predict_proba()`, `save()`, and `load()`.
- **`supervised.py`:** Implements XGBoost / LightGBM / MLP multi-class threat classifiers.
- **`anomaly.py`:** Implements PyTorch Deep Autoencoder and Isolation Forest anomaly detectors.
- **`hybrid.py`:** Combines supervised threat classification with unsupervised anomaly detection into a unified scoring mechanism.

### Module 4: Training Pipeline (`src/training/`)
- **`trainer.py`:** Orchestrates training pipelines, handles stratified train/val splits, cross-validation, learning rate schedules, and saves model checkpoints.
- **`loss.py`:** Custom loss functions (e.g., Focal Loss, Weighted Cross-Entropy).

### Module 5: Inference & Explainability (`src/inference/`)
- **`engine.py`:** Batch scoring loop that processes feature vectors and outputs threat predictions.
- **`explainer.py`:** Computes feature contributions (SHAP values or feature attribution weights) explaining *why* a flow was flagged.

### Module 6: Alerting & Dispatch (`src/alerting/`)
- **`alert_manager.py`:**
  - Evaluates threat severity (`INFO`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`).
  - Generates standardized security events formatted in JSON/CEF with remediation suggestions.

### Module 7: Evaluation & Benchmarking (`src/evaluation/`)
- **`metrics.py`:** Generates classification reports, confusion matrices, ROC/PR curves, and false-positive rate audits.
