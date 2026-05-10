"""Unit tests for eval/metrics.py — focus on edge cases found in smoke testing."""
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from eval.metrics import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pytest

from eval.metrics import (
    compute_roc_auc,
    compute_classification_report,
    compute_latency_stats,
    save_results_json,
)


class TestComputeRocAuc:
    def test_normal_case(self):
        """Standard multiclass ROC computation."""
        labels = ["a", "b", "c"]
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])
        y_scores = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
        ])
        result = compute_roc_auc(y_true, y_scores, labels)
        assert "a" in result
        assert result["a"]["auc"] >= 0

    def test_class_with_zero_samples(self):
        """Class with 0 true positives should not crash, AUC should be handled."""
        labels = ["a", "b", "c"]
        y_true = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])  # "c" has 0 samples
        y_scores = np.array([
            [0.9, 0.05, 0.05],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
        ])
        result = compute_roc_auc(y_true, y_scores, labels)
        # Should not crash; "c" entry should exist with AUC = None
        assert "c" in result
        assert result["c"]["auc"] is None
        # macro_avg should not be NaN
        if "macro_avg" in result:
            macro = result["macro_avg"]["auc"]
            assert macro is None or not (isinstance(macro, float) and np.isnan(macro))

    def test_single_class(self):
        """All samples are the same class — b has 0 positives."""
        labels = ["a", "b"]
        y_true = np.array([[1, 0], [1, 0], [1, 0]])
        y_scores = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        result = compute_roc_auc(y_true, y_scores, labels)
        assert "a" in result
        # "b" has 0 positives so AUC should be None
        assert result["b"]["auc"] is None

    def test_micro_and_macro_present(self):
        """Verify micro_avg and macro_avg keys are always returned."""
        labels = ["x", "y"]
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
        result = compute_roc_auc(y_true, y_scores, labels)
        assert "micro_avg" in result
        assert "macro_avg" in result
        assert isinstance(result["micro_avg"]["auc"], float)
        assert isinstance(result["macro_avg"]["auc"], float)


class TestComputeClassificationReport:
    def test_basic_report(self):
        labels = ["a", "b", "c"]
        y_true = ["a", "b", "c", "a"]
        y_pred = ["a", "b", "b", "a"]
        report = compute_classification_report(y_true, y_pred, labels)
        assert "a" in report
        assert report["a"]["precision"] == 1.0
        assert "macro avg" in report

    def test_missing_class_in_predictions(self):
        """A class that never appears in predictions should get zero precision."""
        labels = ["a", "b", "c"]
        y_true = ["a", "a", "a"]
        y_pred = ["a", "a", "a"]
        report = compute_classification_report(y_true, y_pred, labels)
        # "b" and "c" never predicted and never true -> zero_division=0 -> 0.0
        assert report["b"]["precision"] == 0.0
        assert report["c"]["recall"] == 0.0


class TestComputeLatencyStats:
    def test_empty_list(self):
        """Empty latency list should return zeros, not crash."""
        result = compute_latency_stats([])
        assert result["mean_ms"] == 0 or np.isnan(result["mean_ms"])
        assert result["total_samples"] == 0

    def test_single_value(self):
        result = compute_latency_stats([100.0])
        assert result["mean_ms"] == 100.0
        assert result["total_samples"] == 1
        assert result["median_ms"] == 100.0

    def test_multiple_values(self):
        result = compute_latency_stats([10.0, 20.0, 30.0, 40.0, 50.0])
        assert result["mean_ms"] == 30.0
        assert result["median_ms"] == 30.0
        assert result["min_ms"] == 10.0
        assert result["max_ms"] == 50.0
        assert result["total_samples"] == 5

    def test_p95_p99(self):
        """Percentiles should be within the data range."""
        values = list(range(1, 101))  # 1..100
        result = compute_latency_stats([float(v) for v in values])
        assert result["p95_ms"] >= 95.0
        assert result["p99_ms"] >= 99.0
        assert result["p95_ms"] <= 100.0
        assert result["p99_ms"] <= 100.0


class TestSaveResultsJson:
    def test_nan_handling(self, tmp_path):
        """NaN values should serialize to null, not crash."""
        data = {
            "accuracy": 0.5,
            "auc": float("nan"),
            "nested": {"val": float("nan")},
        }
        path = str(tmp_path / "test.json")
        save_results_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["auc"] is None  # NaN -> null
        assert loaded["nested"]["val"] is None

    def test_inf_handling(self, tmp_path):
        """Inf values should serialize to null."""
        data = {"value": float("inf"), "neg": float("-inf")}
        path = str(tmp_path / "test_inf.json")
        save_results_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["value"] is None
        assert loaded["neg"] is None

    def test_numpy_types(self, tmp_path):
        """Numpy integers, floats, and arrays should serialize correctly."""
        data = {
            "int_val": np.int64(42),
            "float_val": np.float32(3.14),
            "array_val": np.array([1, 2, 3]),
        }
        path = str(tmp_path / "test_numpy.json")
        save_results_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["int_val"] == 42
        assert abs(loaded["float_val"] - 3.14) < 0.01
        assert loaded["array_val"] == [1, 2, 3]

    def test_metadata_injected(self, tmp_path):
        """Metadata block should be auto-added if not present."""
        data = {"accuracy": 0.95}
        path = str(tmp_path / "test_meta.json")
        save_results_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert "metadata" in loaded
        assert "timestamp" in loaded["metadata"]
        assert "python_version" in loaded["metadata"]

    def test_existing_metadata_preserved(self, tmp_path):
        """Existing metadata fields should not be overwritten."""
        data = {"metadata": {"custom_field": "keep_me", "timestamp": "2025-01-01"}}
        path = str(tmp_path / "test_meta_keep.json")
        save_results_json(data, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["metadata"]["custom_field"] == "keep_me"
        assert loaded["metadata"]["timestamp"] == "2025-01-01"

    def test_creates_parent_dirs(self, tmp_path):
        """Should create missing parent directories."""
        path = str(tmp_path / "deep" / "nested" / "dir" / "results.json")
        save_results_json({"test": True}, path)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded["test"] is True
