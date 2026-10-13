"""Tests for deterministic rule-based threat detection engine."""

import pandas as pd
import pytest
from src.detect_rules import RuleDetectionEngine


@pytest.fixture
def sample_events_df():
    """Provides a synthetic DataFrame covering each rule scenario."""
    return pd.DataFrame([
        {
            "event_id": "EVT-001",
            "timestamp": "2026-10-12T20:00:00Z",
            "source_ip": "192.168.1.10",
            "destination_ip": "10.0.0.5",
            "destination_port": 80,
            "protocol": "TCP",
            "failed_logins": 0,
            "byte_count": 1200,
            "packet_count": 10,
            "duration_sec": 1.0,
            "user_agent": "Mozilla/5.0",
            "simulated_scenario": "BENIGN",
        },
        {
            "event_id": "EVT-002",
            "timestamp": "2026-10-12T20:01:00Z",
            "source_ip": "198.51.100.22",
            "destination_ip": "10.0.0.5",
            "destination_port": 22,
            "protocol": "TCP",
            "failed_logins": 8,  # Triggers RULE-AUTH-001
            "byte_count": 4000,
            "packet_count": 25,
            "duration_sec": 30.0,
            "user_agent": "OpenSSH_8.4",
            "simulated_scenario": "BRUTE_FORCE",
        },
        {
            "event_id": "EVT-003",
            "timestamp": "2026-10-12T20:02:00Z",
            "source_ip": "10.0.0.12",
            "destination_ip": "198.51.100.99",
            "destination_port": 443,
            "protocol": "TCP",
            "failed_logins": 0,
            "byte_count": 15_000_000,  # Triggers RULE-EXFIL-002
            "packet_count": 12000,
            "duration_sec": 120.0,
            "user_agent": "curl/7.81.0",
            "simulated_scenario": "DATA_EXFILTRATION",
        },
        {
            "event_id": "EVT-004",
            "timestamp": "2026-10-12T20:03:00Z",
            "source_ip": "203.0.113.15",
            "destination_ip": "10.0.0.25",
            "destination_port": 80,
            "protocol": "TCP",
            "failed_logins": 0,
            "byte_count": 2200,
            "packet_count": 6,
            "duration_sec": 0.5,
            "user_agent": "sqlmap/1.6.4#stable",  # Triggers RULE-PROBE-003
            "simulated_scenario": "WEB_EXPLOIT_PROBE",
        },
    ])


def test_rule_brute_force_detection(sample_events_df):
    engine = RuleDetectionEngine(failed_login_threshold=5)
    alerts = engine.evaluate_events(sample_events_df)

    auth_alerts = alerts[alerts["rule_id"] == "RULE-AUTH-001"]
    assert len(auth_alerts) == 1
    assert auth_alerts.iloc[0]["source_ip"] == "198.51.100.22"
    assert auth_alerts.iloc[0]["severity"] == "HIGH"


def test_rule_exfiltration_detection(sample_events_df):
    engine = RuleDetectionEngine(exfiltration_byte_threshold=10_000_000)
    alerts = engine.evaluate_events(sample_events_df)

    exfil_alerts = alerts[alerts["rule_id"] == "RULE-EXFIL-002"]
    assert len(exfil_alerts) == 1
    assert exfil_alerts.iloc[0]["severity"] == "CRITICAL"
    assert "15,000,000" in exfil_alerts.iloc[0]["evidence"]


def test_rule_scanner_user_agent(sample_events_df):
    engine = RuleDetectionEngine()
    alerts = engine.evaluate_events(sample_events_df)

    probe_alerts = alerts[alerts["rule_id"] == "RULE-PROBE-003"]
    assert len(probe_alerts) == 1
    assert "sqlmap" in probe_alerts.iloc[0]["evidence"]


def test_rule_port_scan_detection():
    # Build 6 probing events to distinct ports from same source
    probe_records = [
        {
            "event_id": f"SCAN-{p}",
            "timestamp": f"2026-10-12T20:10:{p:02d}Z",
            "source_ip": "203.0.113.99",
            "destination_ip": "10.0.0.5",
            "destination_port": 1000 + p,
            "protocol": "TCP",
            "failed_logins": 0,
            "byte_count": 60,
            "packet_count": 2,
            "duration_sec": 0.05,
            "user_agent": "masscan/1.3.2",
            "simulated_scenario": "PORT_SCAN",
        }
        for p in range(6)
    ]
    df = pd.DataFrame(probe_records)
    engine = RuleDetectionEngine()
    alerts = engine.evaluate_events(df)

    scan_alerts = alerts[alerts["rule_id"] == "RULE-SCAN-004"]
    assert len(scan_alerts) == 1
    assert scan_alerts.iloc[0]["source_ip"] == "203.0.113.99"
    assert scan_alerts.iloc[0]["severity"] == "MEDIUM"


def test_benign_event_produces_no_alerts():
    df_benign = pd.DataFrame([{
        "event_id": "EVT-BENIGN-01",
        "timestamp": "2026-10-12T20:00:00Z",
        "source_ip": "10.0.0.5",
        "destination_ip": "10.0.0.12",
        "destination_port": 80,
        "protocol": "TCP",
        "failed_logins": 0,
        "byte_count": 1200,
        "packet_count": 10,
        "duration_sec": 1.0,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "simulated_scenario": "BENIGN",
    }])
    engine = RuleDetectionEngine()
    alerts = engine.evaluate_events(df_benign)
    assert len(alerts) == 0


def test_rules_handle_missing_and_corrupt_values():
    df_corrupt = pd.DataFrame([{
        "event_id": "EVT-CORRUPT-01",
        "timestamp": "2026-10-12T20:00:00Z",
        "source_ip": "10.0.0.5",
        "destination_ip": "10.0.0.12",
        "destination_port": "invalid_port",  # corrupted port string
        "protocol": "TCP",
        "failed_logins": None,                # missing login count
        "byte_count": "not_a_number",        # corrupted byte string
        "packet_count": None,                 # missing packet count
        "duration_sec": None,
        "user_agent": None,                   # missing user agent
        "simulated_scenario": "BENIGN",
    }])
    engine = RuleDetectionEngine()
    alerts = engine.evaluate_events(df_corrupt)
    # Should complete without error and produce no alerts
    assert len(alerts) == 0


def test_duplicate_events_do_not_produce_duplicate_alerts():
    event = {
        "event_id": "EVT-DUP-01",
        "timestamp": "2026-10-12T20:00:00Z",
        "source_ip": "198.51.100.22",
        "destination_ip": "10.0.0.5",
        "destination_port": 22,
        "protocol": "TCP",
        "failed_logins": 10,
        "byte_count": 2000,
        "packet_count": 15,
        "duration_sec": 5.0,
        "user_agent": "OpenSSH_8.4",
        "simulated_scenario": "BRUTE_FORCE",
    }
    df_with_dups = pd.DataFrame([event, event])  # duplicate identical event
    engine = RuleDetectionEngine()
    alerts = engine.evaluate_events(df_with_dups)
    assert len(alerts) == 1
