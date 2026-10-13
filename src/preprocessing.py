"""Preprocessing module for the UNSW-NB15 Cybersecurity Dataset.

Handles data cleaning, missing value handling, column standardization,
and feature transformation using scikit-learn pipelines.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Canonical non-feature or identifier columns to exclude from training
DEFAULT_DROP_COLUMNS = ["id", "attack_cat"]

# Canonical categorical columns in UNSW-NB15
CATEGORICAL_FEATURES = ["proto", "service", "state"]


def clean_dataframe(df: pd.DataFrame, target_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Standardizes column names, cleans whitespace, and handles missing/infinite values.

    Args:
        df: Raw input DataFrame.
        target_columns: Target columns to exclude from numerical median imputation.

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()
    # Strip whitespace and lowercase column headers
    df.columns = df.columns.str.strip().str.lower()

    if target_columns is None:
        target_columns = ["label", "attack_cat"]
    targets_lower = [t.lower() for t in target_columns]

    # Replace '-' placeholder with 'unknown' for categorical columns
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str).str.strip().replace({"-": "unknown", "": "unknown"})

    # Fill numerical missing or infinite values (excluding target columns)
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in targets_lower]
    for col in num_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        median_val = df[col].median() if not df[col].isna().all() else 0.0
        df[col] = df[col].fillna(median_val)

    return df


def prepare_features_and_labels(
    df: pd.DataFrame,
    target_column: str = "label",
    drop_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Splits DataFrame into feature matrix X and target vector y.

    Args:
        df: Input DataFrame.
        target_column: Name of the target label column (default: 'label').
        drop_columns: List of columns to drop from feature set.

    Returns:
        Tuple of (X, y).
    """
    cleaned = clean_dataframe(df)

    if drop_columns is None:
        drop_columns = DEFAULT_DROP_COLUMNS

    # Ensure target column exists
    target_lower = target_column.lower()
    if target_lower not in cleaned.columns:
        raise KeyError(f"Target column '{target_column}' not found in DataFrame.")

    # Explicit validation and coercion of binary target (0 or 1)
    target_series = pd.to_numeric(cleaned[target_lower], errors="coerce")
    valid_mask = target_series.notna() & target_series.isin([0, 1])

    if not valid_mask.all():
        cleaned = cleaned[valid_mask].copy()
        target_series = target_series[valid_mask]
        if len(cleaned) == 0:
            raise ValueError(f"No valid binary target values (0 or 1) found in column '{target_column}'.")

    y = target_series.astype(int)

    # Columns to drop from features
    to_drop = [c.lower() for c in drop_columns if c.lower() in cleaned.columns]
    if target_lower not in to_drop:
        to_drop.append(target_lower)

    X = cleaned.drop(columns=to_drop)
    return X, y


def build_column_transformer(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for numerical and categorical features.

    Args:
        numerical_features: List of continuous feature names.
        categorical_features: List of categorical feature names.

    Returns:
        Configured ColumnTransformer instance.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor
