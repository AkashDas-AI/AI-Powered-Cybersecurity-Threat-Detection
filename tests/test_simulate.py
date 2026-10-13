"""Tests for synthetic security event simulation module."""

import pandas as pd
from src.simulate import generate_synthetic_events


def test_generate_synthetic_events_shape_and_columns():
    df = generate_synthetic_events(count=50, seed=123)
    assert len(df) == 50

    expected_cols = [
        "event_id", "timestamp", "source_ip", "destination_ip", "destination_port",
        "protocol", "failed_logins", "byte_count", "packet_count", "duration_sec",
        "user_agent", "simulated_scenario"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_generate_synthetic_events_reproducibility():
    df1 = generate_synthetic_events(count=30, seed=99)
    df2 = generate_synthetic_events(count=30, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_synthetic_events_contains_scenarios():
    df = generate_synthetic_events(count=100, seed=42)
    scenarios = set(df["simulated_scenario"].unique())
    assert "BENIGN" in scenarios
    assert "BRUTE_FORCE" in scenarios
    assert "DATA_EXFILTRATION" in scenarios
