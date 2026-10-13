"""Dataset Inspection Utility.

Provides a clean CLI utility to inspect dataset dimensions, column data types,
missing/infinite values, and class balance for any project CSV file.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd


def inspect_dataset(filepath: Path) -> Dict[str, Any]:
    """Inspects a CSV dataset and returns structured metadata.

    Args:
        filepath: Path to CSV dataset.

    Returns:
        Dictionary containing dataset summary information.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    df = pd.read_csv(filepath)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    missing_counts = df.isna().sum().to_dict()
    missing_total = int(df.isna().sum().sum())

    inf_count = 0
    for col in num_cols:
        inf_count += int(np.isinf(df[col]).sum())

    summary: Dict[str, Any] = {
        "filepath": str(filepath),
        "rows": len(df),
        "columns": len(df.columns),
        "numerical_column_count": len(num_cols),
        "categorical_column_count": len(cat_cols),
        "total_missing_values": missing_total,
        "total_infinite_values": inf_count,
        "column_names": list(df.columns),
    }

    # Label distribution check if common targets exist
    for target in ["label", "attack_cat", "simulated_scenario"]:
        if target in df.columns:
            summary[f"{target}_distribution"] = df[target].value_counts(dropna=False).to_dict()

    return summary


def print_inspection_report(filepath: Path) -> None:
    """Formats and prints the dataset inspection report to stdout.

    Args:
        filepath: Path to CSV dataset.
    """
    try:
        report = inspect_dataset(filepath)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return

    print("=" * 60)
    print(f" DATASET INSPECTION: {Path(report['filepath']).name}")
    print("=" * 60)
    print(f"Path:                   {report['filepath']}")
    print(f"Total Rows:             {report['rows']:,}")
    print(f"Total Columns:          {report['columns']}")
    print(f"Numerical Columns:      {report['numerical_column_count']}")
    print(f"Categorical Columns:    {report['categorical_column_count']}")
    print(f"Missing Values:         {report['total_missing_values']}")
    print(f"Infinite Values:        {report['total_infinite_values']}")

    for key, dist in report.items():
        if key.endswith("_distribution"):
            col_name = key.replace("_distribution", "")
            print(f"\nDistribution for '{col_name}':")
            for label, count in dist.items():
                pct = (count / report['rows']) * 100
                print(f"  - {label}: {count:,} ({pct:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect CSV dataset structure and label distributions")
    parser.add_argument("path", type=str, help="Path to CSV dataset")
    args = parser.parse_args()

    print_inspection_report(Path(args.path))


if __name__ == "__main__":
    main()
