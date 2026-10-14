# System Modules and Component Specifications

## 1. Source Code Organization

The codebase follows a clean, modular Python structure under `src/`, accompanied by automated tests in `tests/` and the presentation application `app.py`:

```text
AI-Powered Cybersecurity Threat Detection/
|-- src/
|   |-- __init__.py                     # Package initialization
|   |-- preprocessing.py                # Data cleaning, column imputation, ColumnTransformer
|   |-- train.py                        # Model training, dataset preflight, artifact generation
|   |-- evaluate.py                     # Metric calculation (Precision, Recall, F1, FPR, ROC/PR)
|   |-- benchmark_audit.py              # Threshold calibration, OOF analysis, model comparison
|   |-- anomaly_detection.py            # Isolation Forest, benign fitting, hybrid policies
|   |-- detect_rules.py                 # Deterministic SIEM rule engine and alert generation
|   |-- simulate.py                     # Synthetic host and authentication event generator
|   `-- inspect_data.py                 # CLI dataset inspection and distribution summary
|-- tests/                              # Automated test suite (63 unit and integration tests)
|   |-- test_pipeline.py                # Preprocessing, training, and artifact loading tests
|   |-- test_benchmark_audit.py         # Threshold calibration, OOF, and leakage tests
|   |-- test_anomaly_detection.py       # Benign-only fit, score conventions, and hybrid tests
|   |-- test_rules.py                   # Deterministic rule trigger and evidence tests
|   |-- test_simulate.py                # Synthetic event generation and schema tests
|   |-- test_inspect.py                 # CLI inspection tool tests
|   `-- test_dashboard_states.py        # Streamlit dashboard state and error fallback tests
|-- app.py                              # Interactive Streamlit SOC triage dashboard
`-- requirements.txt                    # Python dependencies
```

---

## 2. Granular Module Specifications

### Module 1: Preprocessing (`src/preprocessing.py`)

- **Primary Responsibility:** Prepares tabular network flow datasets for machine learning models through deterministic, reproducible cleaning, feature extraction, and encoding.
- **Key Functions:**
  - `clean_dataframe(df: pd.DataFrame, target_columns: Optional[List[str]] = None) -> pd.DataFrame`:
    - Normalizes column names to lowercase and strips whitespace.
    - Maps `-` and empty values in categorical columns to `unknown`.
    - Imputes `inf` and `-inf` with `NaN`, then fills numeric missing values with feature medians.
  - `prepare_features_and_labels(df: pd.DataFrame, target_column: str = "label", drop_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]`:
    - Validates presence of target column and coerces values to binary integers (`0` or `1`).
    - Drops sequence identifiers (`id`) and multi-class labels (`attack_cat`) to prevent identifier memorization.
    - Returns feature matrix `X` and target series `y`.
  - `build_column_transformer(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer`:
    - Constructs a scikit-learn `ColumnTransformer` with `StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` for categorical columns.

---

### Module 2: Model Training and Preflight (`src/train.py`)

- **Primary Responsibility:** Orchestrates training and evaluation of baseline and ensemble models, enforcing dataset preflight checks to prevent accidental fallback to sample data.
- **Key Functions:**
  - `validate_preflight(train_path: Path, test_path: Optional[Path] = None, is_sample: bool = False) -> None`:
    - Asserts existence of training and testing files.
    - Raises informative `FileNotFoundError` if authentic benchmark files are missing, instructing users on how to acquire them.
    - Prevents silent fallback from benchmark datasets to synthetic sample data.
  - `train_pipeline(train_path: Path, test_path: Optional[Path] = None, model_output: Optional[Path] = None, metrics_output: Optional[Path] = None, random_state: int = 42, classifier_type: str = "logistic") -> dict`:
    - Instantiates `LogisticRegression(max_iter=1000)` or `RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=5)`.
    - Aligns test feature schemas to training feature sets.
    - Fits the end-to-end `Pipeline` (`ColumnTransformer` + classifier).
    - Serializes model binaries (`.joblib`) and metric summaries (`.json`).
- **CLI Interface:** Supports `--sample`, `--classifier [logistic|random_forest]`, `--train-path`, `--test-path`.

---

### Module 3: Metric Evaluation (`src/evaluate.py`)

- **Primary Responsibility:** Calculates comprehensive classification and security domain metrics on binary predictions and probabilistic outputs.
- **Key Functions:**
  - `compute_metrics(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray], y_prob: Optional[Union[pd.Series, np.ndarray]] = None) -> dict`:
    - Computes confusion matrix components: True Negatives (`TN`), False Positives (`FP`), False Negatives (`FN`), True Positives (`TP`).
    - Calculates operational rates:
      - Precision: $\frac{TP}{TP + FP}$
      - Recall (Detection Rate): $\frac{TP}{TP + FN}$
      - F1-Score: $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
      - False Positive Rate (FPR): $\frac{FP}{TN + FP}$
    - Computes Receiver Operating Characteristic AUC (ROC-AUC) and Precision-Recall AUC (PR-AUC) when probabilities are supplied.
  - `save_metrics(metrics: dict, output_path: Path) -> None`:
    - Serializes metric dictionaries to formatted JSON with indented indentation.

---

### Module 4: Benchmark Audit and Threshold Calibration (`src/benchmark_audit.py`)

- **Primary Responsibility:** Implements leak-free threshold calibration, cross-validation sweeps, and head-to-head model comparisons.
- **Key Functions:**
  - `compute_threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[np.ndarray] = None) -> List[Dict[str, Any]]`:
    - Computes precision, recall, F1, and FPR across 91 threshold points spanning 0.05 to 0.95.
  - `select_candidate_threshold(threshold_metrics: List[Dict[str, Any]], min_recall: float = 0.90, objective: str = "min_fpr") -> Dict[str, Any]`:
    - Selects the optimal threshold meeting a minimum recall constraint (default $\ge 90\%$) while minimizing false alarm rate.
  - `run_oof_threshold_calibration(...)`:
    - Executes 5-fold Stratified K-Fold cross-validation across all 175,341 training records to generate out-of-fold probability distributions without test leakage.
  - `run_consistent_calibration_pipeline(...)`:
    - Splits the official training set into an 80% subtrain and 20% validation partition.
    - Fits the model on 80% subtrain, selects threshold $\theta = 0.76$ on 20% validation, and deploys the identical frozen pipeline directly to the test set, guaranteeing zero refit drift and full parameter invariance.
  - `plot_model_comparison_curves(...)`, `plot_threshold_curves(...)`:
    - Generates publication-ready comparative ROC, Precision-Recall, threshold trade-off, and confusion matrix figures.

---

### Module 5: Novelty Anomaly Detection and Hybrid Gating (`src/anomaly_detection.py`)

- **Primary Responsibility:** Implements unsupervised novelty detection and evaluates hybrid confirmation policies.
- **Key Functions:**
  - `fit_isolation_forest_pipeline(X_train_benign, numerical_features, categorical_features, n_estimators=100, max_samples=256, contamination="auto", random_state=42) -> Pipeline`:
    - Fits `IsolationForest` strictly on benign traffic (`label = 0`, N=44,800 records) to establish normal traffic baselines.
  - `compute_anomaly_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray`:
    - Returns negated decision function: $\text{anomaly\_score} = -\text{decision\_function}(X)$. Inliers yield lower/negative scores; anomalies yield higher/positive scores.
  - `select_hybrid_policy_on_validation(...)`:
    - Sweeps policy thresholds strictly on the validation partition (N=35,069).
    - Evaluates Cascaded OR vs. Suspicious-Band Confirmation policies across lower boundary candidates (0.20 to 0.70).
  - `evaluate_hybrid_policies(...)`:
    - Applies the frozen validation-selected policy ($0.40 \le P < 0.76$) to the official held-out test split (82,332 records) and reports false positive trade-offs.

---

### Module 6: Deterministic SIEM Rule Engine (`src/detect_rules.py`)

- **Primary Responsibility:** Evaluates deterministic security rules on structured host event telemetry and persists actionable alert records.
- **Class:** `RuleDetectionEngine(failed_login_threshold=5, exfiltration_byte_threshold=10_000_000)`
- **Key Methods:**
  - `evaluate_events(df: pd.DataFrame) -> pd.DataFrame`:
    - Validates required event columns and drops duplicate `event_id` rows.
    - Sanitizes data types and replaces missing counts with zeros.
    - Evaluates four deterministic rules:
      1. `RULE-AUTH-001`: Brute-Force Authentication (failed logins $\ge 5$). Emits HIGH severity.
      2. `RULE-EXFIL-002`: Data Exfiltration (bytes $\ge 10\text{ MB}$). Emits CRITICAL severity.
      3. `RULE-PROBE-003`: Offensive Tool Signatures (`sqlmap`, `nikto`, `masscan`, `malicious-probe` in User-Agent). Emits HIGH severity.
      4. `RULE-SCAN-004`: Multi-Port Reconnaissance ($\ge 5$ distinct destination ports with packet count $\le 5$). Emits MEDIUM severity.
    - Returns DataFrame of formatted alerts containing alert IDs, timestamps, IP addresses, severities, and explicit trigger evidence.

---

### Module 7: Event Simulation (`src/simulate.py`)

- **Primary Responsibility:** Generates synthetic host and authentication telemetry records for reproducible rule testing and offline demonstrations.
- **Key Functions:**
  - `generate_security_events(count: int = 150, attack_ratio: float = 0.35, random_seed: int = 42) -> pd.DataFrame`:
    - Simulates normal background traffic and controlled attack scenarios (brute-force login loops, bulk file transfers, vulnerability scans, and port probes).
    - Outputs timestamps formatted in ISO 8601 UTC.

---

### Module 8: Dataset Inspection (`src/inspect_data.py`)

- **Primary Responsibility:** CLI utility providing fast summaries of dataset schemas, missing value counts, memory usage, and class balance distributions.
- **Key Functions:**
  - `inspect_dataset(file_path: Path) -> dict`: Reads CSV, prints column data types, missing value percentages, and binary/categorical distribution metrics.

---

### Module 9: Presentation Dashboard (`app.py`)

- **Primary Responsibility:** Interactive Streamlit web application for reviewing benchmark results, threshold trade-offs, and SIEM alerts.
- **Key Features:**
  - Multi-tab navigation: System Overview, Benchmark Model Comparison, Anomaly Detection & Hybrid Defense, SIEM Threat Alerts, and Telemetry Explorer.
  - Graceful degradation: Detects missing or corrupted artifacts and displays actionable guidance rather than throwing unhandled exceptions.
  - Interactive triage table: Allows multi-attribute filtering by alert severity, rule ID, and source IP.
