"""Deterministic Rule-Based Detection Engine.

Applies deterministic security rules to synthetic security event records
and persists structured alerts to outputs/alerts/ for triage and SIEM review.

NOTE: This workflow operates on the synthetic security event schema.
It is decoupled from the UNSW-NB15 machine learning pipeline.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


DEFAULT_EVENTS_INPUT = Path("data/sample/synthetic_security_events.csv")
DEFAULT_ALERTS_OUTPUT = Path("outputs/alerts/detected_alerts.csv")

# Known offensive reconnaissance tools in User-Agents
KNOWN_TOOL_SIGNATURES = ["sqlmap", "nikto", "masscan", "malicious-probe"]


class RuleDetectionEngine:
    """Evaluates deterministic security rules against security event streams."""

    def __init__(self, failed_login_threshold: int = 5, exfiltration_byte_threshold: int = 10_000_000):
        self.failed_login_threshold = failed_login_threshold
        self.exfiltration_byte_threshold = exfiltration_byte_threshold

    def evaluate_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies detection rules across an event DataFrame.

        Args:
            df: DataFrame containing synthetic security event records.

        Returns:
            DataFrame of generated security alerts.
        """
        alerts: List[Dict[str, Any]] = []
        alert_counter = 1

        # Check required columns
        required_cols = ["event_id", "timestamp", "source_ip", "destination_ip", "destination_port",
                         "failed_logins", "byte_count", "packet_count", "user_agent"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns in events DataFrame: {missing}")

        df = df.copy()
        # Drop duplicate event rows to prevent unintended duplicate alert dispatch
        df = df.drop_duplicates(subset=["event_id"])

        # Sanitize data types and handle missing or corrupted values gracefully
        df["failed_logins"] = pd.to_numeric(df["failed_logins"], errors="coerce").fillna(0)
        df["byte_count"] = pd.to_numeric(df["byte_count"], errors="coerce").fillna(0)
        df["packet_count"] = pd.to_numeric(df["packet_count"], errors="coerce").fillna(0)
        df["destination_port"] = pd.to_numeric(df["destination_port"], errors="coerce").fillna(0).astype(int)
        df["user_agent"] = df["user_agent"].fillna("").astype(str)

        # Rule 1: Brute Force Authentication
        for _, row in df[df["failed_logins"] >= self.failed_login_threshold].iterrows():
            alerts.append({
                "alert_id": f"ALT-{alert_counter:04d}",
                "rule_id": "RULE-AUTH-001",
                "rule_name": "Brute-Force Authentication Attempt",
                "severity": "HIGH",
                "timestamp": row["timestamp"],
                "source_ip": row["source_ip"],
                "destination_ip": row["destination_ip"],
                "destination_port": int(row["destination_port"]),
                "triggering_event_id": row["event_id"],
                "evidence": f"failed_logins={row['failed_logins']} (threshold={self.failed_login_threshold})",
            })
            alert_counter += 1

        # Rule 2: High-Volume Data Exfiltration
        for _, row in df[df["byte_count"] >= self.exfiltration_byte_threshold].iterrows():
            alerts.append({
                "alert_id": f"ALT-{alert_counter:04d}",
                "rule_id": "RULE-EXFIL-002",
                "rule_name": "Suspicious High-Volume Data Exfiltration",
                "severity": "CRITICAL",
                "timestamp": row["timestamp"],
                "source_ip": row["source_ip"],
                "destination_ip": row["destination_ip"],
                "destination_port": int(row["destination_port"]),
                "triggering_event_id": row["event_id"],
                "evidence": f"byte_count={row['byte_count']:,} bytes (threshold={self.exfiltration_byte_threshold:,})",
            })
            alert_counter += 1

        # Rule 3: Malicious Scanner / Tool User-Agent Signature
        for _, row in df.iterrows():
            ua = str(row["user_agent"]).lower()
            matching_tool = next((sig for sig in KNOWN_TOOL_SIGNATURES if sig in ua), None)
            if matching_tool:
                alerts.append({
                    "alert_id": f"ALT-{alert_counter:04d}",
                    "rule_id": "RULE-PROBE-003",
                    "rule_name": "Offensive Security Scanner User-Agent",
                    "severity": "HIGH",
                    "timestamp": row["timestamp"],
                    "source_ip": row["source_ip"],
                    "destination_ip": row["destination_ip"],
                    "destination_port": int(row["destination_port"]),
                    "triggering_event_id": row["event_id"],
                    "evidence": f"User-Agent matched '{matching_tool}': {row['user_agent']}",
                })
                alert_counter += 1

        # Rule 4: Multi-Port Reconnaissance Scanning (Aggregated by Source IP)
        small_packets_df = df[df["packet_count"] <= 5]
        ports_per_source = small_packets_df.groupby("source_ip")["destination_port"].nunique()
        scanning_sources = ports_per_source[ports_per_source >= 5].index.tolist()

        for src in scanning_sources:
            subset = small_packets_df[small_packets_df["source_ip"] == src]
            first_event = subset.iloc[0]
            scanned_ports = subset["destination_port"].unique().tolist()
            alerts.append({
                "alert_id": f"ALT-{alert_counter:04d}",
                "rule_id": "RULE-SCAN-004",
                "rule_name": "Horizontal/Vertical Port Reconnaissance Probing",
                "severity": "MEDIUM",
                "timestamp": first_event["timestamp"],
                "source_ip": src,
                "destination_ip": first_event["destination_ip"],
                "destination_port": int(first_event["destination_port"]),
                "triggering_event_id": first_event["event_id"],
                "evidence": f"Source scanned {len(scanned_ports)} distinct ports: {scanned_ports[:8]}...",
            })
            alert_counter += 1

        return pd.DataFrame(alerts)


def detect_and_save_alerts(
    input_path: Path = DEFAULT_EVENTS_INPUT,
    output_path: Path = DEFAULT_ALERTS_OUTPUT,
) -> pd.DataFrame:
    """Loads events, applies rules, and writes alerts to CSV.

    Args:
        input_path: Path to events CSV.
        output_path: Destination path for alerts CSV.

    Returns:
        DataFrame of generated alerts.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input events file not found at: {input_path}")

    events_df = pd.read_csv(input_path)
    engine = RuleDetectionEngine()
    alerts_df = engine.evaluate_events(events_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    alerts_df.to_csv(output_path, index=False)
    return alerts_df


def main():
    parser = argparse.ArgumentParser(description="Run deterministic rule detection on security events")
    parser.add_argument("--input", type=str, default=str(DEFAULT_EVENTS_INPUT), help="Path to events CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_ALERTS_OUTPUT), help="Path to alerts output CSV")
    args = parser.parse_args()

    print(f"[*] Running rule detection on: {args.input}")
    try:
        alerts = detect_and_save_alerts(Path(args.input), Path(args.output))
        print(f"[+] Detection complete. Generated {len(alerts)} alerts saved to: {args.output}")
        if not alerts.empty:
            severity_counts = alerts["severity"].value_counts().to_dict()
            print(f"[*] Severity Breakdown: {severity_counts}")
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")


if __name__ == "__main__":
    main()
