# Project Technical Documentation

Welcome to the technical documentation repository for the **AI-Powered Cybersecurity Threat Detection System**. This directory provides architectural specifications, data lifecycle diagrams, algorithmic learning mechanics, module interface contracts, and validation protocols.

---

## Documentation Structure

| Document | Description |
| :--- | :--- |
| [**System Architecture**](architecture.md) | Multi-tiered decoupled architectural design, system boundaries, technology stack, and engineering decisions. |
| [**Data Flow and Pipeline**](data_flow.md) | End-to-end data lifecycle across preprocessing, partition splitting, feature transformation, inference scoring, and SIEM alerting. |
| [**Dataset Specification**](dataset_reference.md) | Complete reference for the authentic UNSW-NB15 benchmark, 49-feature dictionary, 9 attack categories, synthetic event schemas, and academic citations. |
| [**Model Architecture and Learning**](model_learning.md) | Mathematical formulations for Logistic Regression, Random Forest, Isolation Forest (benign-only fit), leak-free threshold calibration, and hybrid confirmation gating. |
| [**Detection Logic**](detection-logic.md) | Specifications for the four deterministic SIEM security rules, trigger thresholds, operational rationales, and alert schemas. |
| [**Module Specifications**](modules_specification.md) | Granular breakdown of each Python module in `src/`, key function signatures, inputs, outputs, and CLI options. |
| [**Validation and Verification Protocol**](validation_guide.md) | Multi-layer quality assurance protocol verifying the 63 automated unit and integration tests across 7 test suites. |
| [**Demonstration Checklist**](demo-checklist.md) | Step-by-step checklist to set up environments, run tests, train models, simulate events, evaluate rules, and launch the Streamlit dashboard. |

---

## Architectural Principles

Every technical component in this repository adheres to three engineering standards:
1. Methodological Rigor: Strict zero-leakage protocols preventing test-set exposure during preprocessing, threshold calibration, and policy sweeps.
2. Operational Reality: Optimization for SOC triage capacity by balancing attack recall against false alarm suppression rather than chasing raw accuracy vanity metrics.
3. Decoupled Modularity: Clean separation between statistical network flow analysis and deterministic host-level SIEM telemetry.
