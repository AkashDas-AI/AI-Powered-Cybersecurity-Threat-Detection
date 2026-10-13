# Deterministic Rule Detection Logic

This document specifies the deterministic rule engine implemented in [`src/detect_rules.py`](file:///d:/Projects/Artificial%20Intelligence/AI-Powered%20Cybersecurity%20Threat%20Detection/src/detect_rules.py) for analyzing synthetic security event telemetry.

---

## 1. Architectural Role

In modern Security Operations Centers (SOC) and SIEM architectures, deterministic rules serve as the frontline triage layer for:
- Detecting well-known, high-confidence Indicators of Attack (IoAs).
- Instant alerting without statistical model latency.
- Direct operational explainability with explicit evidence.

This engine processes synthetic security events (`data/sample/synthetic_security_events.csv`) and writes actionable alerts to [`outputs/alerts/detected_alerts.csv`](file:///d:/Projects/Artificial%20Intelligence/AI-Powered%20Cybersecurity%20Threat%20Detection/outputs/alerts/detected_alerts.csv).

> [!NOTE]
> This deterministic rule pipeline is decoupled from the machine learning model pipeline (which runs on UNSW-NB15 network flow features).

---

## 2. Specification of Detection Rules

### Rule 1: Brute-Force Authentication Attempt
- **Rule ID:** `RULE-AUTH-001`
- **Severity:** `HIGH`
- **Target Event:** Authentication / Login attempts (SSH, RDP, Web).
- **Condition:**
  $$\text{failed\_logins} \ge 5$$
- **Evidence Emitted:** String specifying observed count vs. threshold (e.g. `failed_logins=12 (threshold=5)`).
- **Operational Rationale:** Repeated sequential authentication failures indicate credential stuffing or dictionary attacks.

---

### Rule 2: Suspicious High-Volume Data Exfiltration
- **Rule ID:** `RULE-EXFIL-002`
- **Severity:** `CRITICAL`
- **Target Event:** Data transfer sessions.
- **Condition:**
  $$\text{byte\_count} \ge 10{,}000{,}000 \text{ bytes } (\sim 10 \text{ MB})$$
- **Evidence Emitted:** Exact byte count observed vs. threshold.
- **Operational Rationale:** Abnormally large single-session egress to external addresses suggests unauthorized data staging or exfiltration.

---

### Rule 3: Offensive Security Scanner User-Agent
- **Rule ID:** `RULE-PROBE-003`
- **Severity:** `HIGH`
- **Target Event:** Web requests and HTTP probes.
- **Condition:** Case-insensitive match on known attack tool signatures within the `user_agent` header:
  - `sqlmap` (Automated SQL injection tool)
  - `nikto` (Vulnerability scanner)
  - `masscan` (High-rate port scanner)
  - `malicious-probe` (Generic offensive probe)
- **Evidence Emitted:** Matched tool name and full User-Agent header string.

---

### Rule 4: Horizontal / Vertical Port Reconnaissance Probing
- **Rule ID:** `RULE-SCAN-004`
- **Severity:** `MEDIUM`
- **Target Event:** Connection requests aggregated by source IP.
- **Condition:**
  - $\text{packet\_count} \le 5$ (SYN-only or probe-level interaction), AND
  - Number of distinct destination ports probed by the same `source_ip` $\ge 5$.
- **Evidence Emitted:** Total distinct ports scanned and port list sample.
- **Operational Rationale:** Scanning multiple ports with minimal packet interaction indicates active reconnaissance and service enumeration.

---

## 3. Standard Alert Schema

All alerts written to [`outputs/alerts/detected_alerts.csv`](file:///d:/Projects/Artificial%20Intelligence/AI-Powered%20Cybersecurity%20Threat%20Detection/outputs/alerts/detected_alerts.csv) follow a consistent schema:

| Field | Type | Description |
| :--- | :--- | :--- |
| `alert_id` | String | Unique identifier (e.g., `ALT-0001`). |
| `rule_id` | String | Static rule ID (e.g., `RULE-AUTH-001`). |
| `rule_name` | String | Human-readable threat description. |
| `severity` | Enum | `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW`. |
| `timestamp` | String | ISO 8601 UTC timestamp of trigger event. |
| `source_ip` | String | Originating IP address. |
| `destination_ip` | String | Target host IP address. |
| `destination_port` | Integer | Target port. |
| `triggering_event_id` | String | Pointer to raw event ID (`EVT-XXXX`). |
| `evidence` | String | Specific values that exceeded thresholds. |
