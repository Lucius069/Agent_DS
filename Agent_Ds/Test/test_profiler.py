"""
Basic sanity tests for profiler.py.
Run with: python -m pytest tests/test_profiler.py -v
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.profiler import profile_dataset


def test_shape_is_correct():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    profile = profile_dataset(df)
    assert profile["shape"]["rows"] == 3
    assert profile["shape"]["columns"] == 2


def test_missing_values_detected():
    df = pd.DataFrame({"a": [1, np.nan, 3, np.nan]})
    profile = profile_dataset(df)
    assert profile["missing_values"]["a"]["count"] == 2
    assert profile["missing_values"]["a"]["percent"] == 50.0


def test_no_missing_values_key_when_column_is_complete():
    df = pd.DataFrame({"a": [1, 2, 3]})
    profile = profile_dataset(df)
    assert "a" not in profile["missing_values"]


def test_duplicates_counted():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    profile = profile_dataset(df)
    assert profile["duplicates"] == 1


def test_outlier_detection():
    # 9 normal values + 1 extreme outlier
    df = pd.DataFrame({"a": [10, 11, 12, 9, 10, 11, 10, 9, 11, 1000]})
    profile = profile_dataset(df)
    assert profile["numeric_summary"]["a"]["outlier_count"] >= 1


def test_categorical_cardinality():
    df = pd.DataFrame({"cat": ["x", "y", "x", "z", "x"]})
    profile = profile_dataset(df)
    assert profile["categorical_summary"]["cat"]["unique_count"] == 3


def test_correlation_detected():
    # perfectly correlated columns
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
    profile = profile_dataset(df)
    assert len(profile["correlations"]) == 1
    assert profile["correlations"][0]["correlation"] == 1.0


def test_low_correlation_not_flagged():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 1, 4, 2, 3]})
    profile = profile_dataset(df)
    # weak/random relationship shouldn't clear the 0.5 threshold
    for corr in profile["correlations"]:
        assert abs(corr["correlation"]) >= 0.5
