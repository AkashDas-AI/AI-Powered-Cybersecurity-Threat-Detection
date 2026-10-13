"""Tests for inspect_data utility."""

from pathlib import Path
from src.inspect_data import inspect_dataset


def test_inspect_dataset_returns_valid_metadata():
    path = Path("data/sample/synthetic_security_events.csv")
    summary = inspect_dataset(path)

    assert summary["rows"] > 0
    assert summary["columns"] > 0
    assert "event_id" in summary["column_names"]
    assert "simulated_scenario_distribution" in summary
    assert summary["total_missing_values"] == 0
