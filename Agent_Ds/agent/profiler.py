"""
profiler.py — Step 1: Static Analysis MVP

This module takes a pandas DataFrame and produces a structured, JSON-serializable
"data profile" dictionary. No AI/LLM calls happen here — this is pure statistics.

Later steps (insight_generator.py) will feed this profile's output into an LLM
to turn it into natural-language insights.

Beginner notes are left in as comments — delete them once you're comfortable.
"""

import pandas as pd
import numpy as np
from scipy import stats


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Main entry point. Takes a DataFrame, returns a dict profile.
    This dict is what gets saved, embedded into memory, and shown to the LLM later.
    """
    profile = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "dtypes": _get_dtypes(df),
        "missing_values": _get_missing_values(df),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": _profile_numeric_columns(df),
        "categorical_summary": _profile_categorical_columns(df),
        "correlations": _get_correlations(df),
    }
    return profile


def _get_dtypes(df: pd.DataFrame) -> dict:
    """Column name -> dtype as a string (so it's JSON-safe)."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def _get_missing_values(df: pd.DataFrame) -> dict:
    """
    Returns count AND percentage of missing values per column.
    Percentage matters more than raw count — 5 missing out of 10 rows
    is a very different problem than 5 missing out of 1,000,000.
    """
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    result = {}
    for col in df.columns:
        if missing_counts[col] > 0:
            result[col] = {
                "count": int(missing_counts[col]),
                "percent": float(missing_pct[col]),
            }
    return result


def _profile_numeric_columns(df: pd.DataFrame) -> dict:
    """
    For every numeric column: basic stats, skewness, and outlier count.

    Skewness interpretation (rule of thumb):
      -0.5 to 0.5   -> fairly symmetric
      0.5 to 1 or -0.5 to -1 -> moderately skewed
      beyond that   -> highly skewed (consider a log/sqrt transform before modeling)

    Outliers are flagged using the IQR method:
      anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier.
      This is a classic, simple, and widely-used rule (not the only one, but a
      solid first pass any interviewer will recognize).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    result = {}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]

        result[col] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "skew": float(stats.skew(series)),
            "outlier_count": int(len(outliers)),
            "outlier_percent": round(float(len(outliers) / len(series) * 100), 2),
        }

    return result


def _profile_categorical_columns(df: pd.DataFrame) -> dict:
    """
    For every non-numeric column: how many unique values (cardinality),
    and the top few most common values.

    Cardinality matters because:
      - Very low cardinality (e.g., 2-3 values) -> good candidate for one-hot encoding
      - Very high cardinality (e.g., unique IDs, free text) -> not usable as a
        categorical feature directly; needs different handling (or should be dropped)
    """
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    result = {}

    for col in categorical_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        value_counts = series.value_counts().head(5)
        result[col] = {
            "unique_count": int(series.nunique()),
            "unique_ratio": round(float(series.nunique() / len(series)), 3),
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
        }

    return result


def _get_correlations(df: pd.DataFrame, threshold: float = 0.5) -> list:
    """
    Returns only the *notable* correlations (above threshold), not the full matrix.
    A full correlation matrix on a 50-column dataset is noise; an agent (and a human)
    cares about the handful of strong relationships.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return []

    corr_matrix = numeric_df.corr()
    notable = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_a = corr_matrix.columns[i]
            col_b = corr_matrix.columns[j]
            value = corr_matrix.iloc[i, j]

            if pd.notna(value) and abs(value) >= threshold:
                notable.append({
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": round(float(value), 3),
                })

    # Strongest relationships first
    notable.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return notable


if __name__ == "__main__":
    # Quick manual test — run this file directly to see it work on a sample dataset.
    import json

    sample_df = pd.DataFrame({
        "age": [25, 32, 47, 29, np.nan, 200, 31, 28, 45, 33],
        "salary": [50000, 62000, 85000, 58000, 60000, 61000, 59000, 300000, 82000, 63000],
        "department": ["Sales", "Eng", "Eng", "Sales", "HR", "Eng", "Sales", "Eng", "HR", "Eng"],
    })

    result = profile_dataset(sample_df)
    print(json.dumps(result, indent=2))
