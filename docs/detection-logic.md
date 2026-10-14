# Deterministic Rule Detection Logic

This document specifies the deterministic rule engine implemented in `src/detect_rules.py` for analyzing synthetic security event telemetry.

---

## 1. Architectural Role

In modern Security Operations Centers (SOC) and SIEM architectures, deterministic rules serve as the frontline triage layer for:
- Detecting well-known, high-confidence Indicators of Attack (IoAs).
- Instant alerting without statistical model latency.
- Direct operational explainability with explicit evidence.

This engine processes synthetic security events (`data/sample/synthetic_security_events.csv`) and writes actionable alerts to `outputs/alerts/detected_alerts.csv`.

> [!NOTE]
> This deterministic rule pipeline is decoupled from the machine learning model pipeline (which runs on UNSW-NB15 network flow features).

---

## 2. Specification of Detection Rules

### Rule 1: Brute-Force Authentication Attempt
- Rule ID: `RULE-AUTH-001`
- Severity: `HIGH`
- Target Event: Authentication and login attempts (SSH, RDP, Web).
- Trigger Condition:
  $$\text{failed\_logins} \ge 5$$
- Evidence Emitted: String specifying observed count vs. threshold (e.g., `failed_logins=12 (threshold=5)`).
- Operational Rationale: Repeated sequential authentication failures indicate credential stuffing, password spraying, or dictionary attacks.

---

### Rule 2: Suspicious High-Volume Data Exfiltration
- Rule ID: `RULE-EXFIL-002`
- Severity: `CRITICAL`
- Target Event: Data transfer sessions.
- Trigger Condition:
  $$\text{byte\_count} \ge 10{,}000{,}000 \text{ bytes } (\sim 10 \text{ MB})$$
- Evidence Emitted: Exact byte count observed vs. threshold (e.g., `byte_count=15,482,900 bytes (threshold=10,000,000)`).
- Operational Rationale: Abnormally large single-session egress to external addresses suggests unauthorized data staging or exfiltration.

---

### Rule 3: Offensive Security Scanner User-Agent
- Rule ID: `RULE-PROBE-003`
- Severity: `HIGH`
- Target Event: Web requests and HTTP probes.
- Trigger Condition: Case-insensitive match on known attack tool signatures within the `user_agent` header:
  - `sqlmap` (Automated SQL injection tool)
  - `nikto` (Web server vulnerability scanner)
  - `masscan` (High-rate TCP port scanner)
  - `malicious-probe` (Generic offensive scanner probe)
- Evidence Emitted: Matched tool name and full User-Agent header string.

---

### Rule 4: Multi-Port Reconnaissance Probing
- Rule ID: `RULE-SCAN-004`
- Severity: `MEDIUM`
- Target Event: Connection requests aggregated by source IP.
- Trigger Condition:
  - $\text{packet\_count} \le 5$ (SYN-only or probe-level interaction), AND
  - Number of distinct destination ports probed by the same `source_ip` $\ge 5$.
- Evidence Emitted: Total distinct ports scanned and port list sample.
- Operational Rationale: Probing multiple ports with minimal packet interaction indicates active reconnaissance and service enumeration.

---

## 3. Standard Alert Schema

All alerts written to `outputs/alerts/detected_alerts.csv` follow a standardized schema:

| Field | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `alert_id` | String | Unique identifier | `ALT-0001` |
| `rule_id` | String | Static rule ID | `RULE-AUTH-001` |
| `rule_name` | String | Human-readable threat description | `Brute-Force Authentication Attempt` |
| `severity` | Enum | `CRITICAL`, `HIGH`, `MEDIUM`, or `LOW` | `HIGH` |
| `timestamp` | String | ISO 8601 UTC timestamp of trigger event | `2026-10-13T10:00:00Z` |
| `source_ip` | String | Originating IP address | `192.168.1.105` |
| `destination_ip` | String | Target host IP address | `10.0.0.5` |
| `destination_port` | Integer | Target port | `22` |
| `triggering_event_id` | String | Pointer to raw event ID | `EVT-0042` |
| `evidence` | String | Specific values that exceeded thresholds | `failed_logins=8 (threshold=5)` |
