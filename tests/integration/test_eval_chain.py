"""
Integration tests for the evaluation pipeline.
Runs each eval script with --limit 5 to verify the chain works.

These tests require:
- FER2013 dataset at eval/datasets/FER2013/test/
- RAF-DB dataset at eval/datasets/RAF-DB/DATASET/test/
- DeepFace and all detector backends installed
- GPU or CPU fallback available
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"

# Use a separate output dir so we don't overwrite real results
TEST_OUTPUT = EVAL_DIR / "results" / "_test_integration"

# Dataset paths for skip conditions
FER2013_TEST = EVAL_DIR / "datasets" / "FER2013" / "test"
RAFDB_TEST = EVAL_DIR / "datasets" / "RAF-DB" / "DATASET" / "test"

has_fer2013 = pytest.mark.skipif(
    not FER2013_TEST.is_dir(),
    reason=f"FER2013 test set not found at {FER2013_TEST}",
)
has_rafdb = pytest.mark.skipif(
    not RAFDB_TEST.is_dir(),
    reason=f"RAF-DB test set not found at {RAFDB_TEST}",
)


@pytest.fixture(scope="module", autouse=True)
def setup_test_output():
    TEST_OUTPUT.mkdir(parents=True, exist_ok=True)
    yield
    # Don't clean up — useful for debugging


def _run_eval_script(script_name: str, extra_args: list = None) -> subprocess.CompletedProcess:
    """Run an eval script as a subprocess and return the result."""
    cmd = [
        sys.executable,
        str(EVAL_DIR / script_name),
        "--limit", "5",
        "--output-dir", str(TEST_OUTPUT),
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300, cwd=str(PROJECT_ROOT),
    )
    return result


def _load_json(filename: str) -> dict:
    path = TEST_OUTPUT / filename
    assert path.exists(), f"Expected output file not found: {path}"
    with open(path) as f:
        return json.load(f)


@has_fer2013
class TestFER2013Baseline:
    def test_benchmark_runs(self):
        result = _run_eval_script("benchmark_fer2013.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr[-500:]}"

    def test_json_output_valid(self):
        data = _load_json("fer2013_baseline_results.json")
        assert "overall_accuracy" in data
        assert "classification_report" in data
        assert "confusion_matrix" in data
        assert "latency" in data
        assert "metadata" in data
        assert data["total_images"] == 5

    def test_plots_generated(self):
        for suffix in [
            "confusion_matrix.png",
            "roc_curves.png",
            "per_class_metrics.png",
            "latency_histogram.png",
        ]:
            path = TEST_OUTPUT / f"fer2013_{suffix}"
            assert path.exists(), f"Missing plot: {suffix}"
            assert path.stat().st_size > 1000, f"Plot too small: {suffix}"


@has_rafdb
class TestRAFDBBaseline:
    def test_benchmark_runs(self):
        result = _run_eval_script("benchmark_rafdb.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr[-500:]}"

    def test_json_output_valid(self):
        data = _load_json("rafdb_baseline_results.json")
        assert "overall_accuracy" in data
        assert data["dataset"] == "RAF-DB"


@has_fer2013
class TestAblationPreprocess:
    def test_ablation_runs(self):
        result = _run_eval_script("ablation_preprocess.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr[-500:]}"

    def test_all_configs_present(self):
        data = _load_json("ablation_preprocess.json")
        expected = {"none", "sr_only", "clahe_only", "sr_clahe", "full_preprocess"}
        assert expected.issubset(set(data["configs"].keys()))


@has_fer2013
class TestAblationDetector:
    def test_ablation_runs(self):
        result = _run_eval_script("ablation_detector.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr[-500:]}"

    def test_detectors_present(self):
        data = _load_json("ablation_detector.json")
        # At least retinaface should have results
        assert "retinaface" in data["detectors"]


@has_fer2013
class TestAblationPostprocess:
    def test_ablation_runs(self):
        # Need at least 40 images for batches of 20
        result = _run_eval_script("ablation_postprocess.py", ["--limit", "40"])
        assert result.returncode == 0, f"Script failed:\n{result.stderr[-500:]}"

    def test_temporal_metrics(self):
        data = _load_json("ablation_postprocess.json")
        assert "raw" in data["configs"]
        raw = data["configs"]["raw"]
        assert "consistency_score" in raw
        assert "flicker_rate" in raw


@has_fer2013
@has_rafdb
class TestPipelineVsBaseline:
    def test_pipeline_runs(self):
        result = _run_eval_script("pipeline_vs_baseline.py")
        assert result.returncode == 0, f"Script failed:\n{result.stderr[-500:]}"

    def test_comparison_data(self):
        data = _load_json("pipeline_vs_baseline.json")
        assert "fer2013" in data or "FER2013" in str(data)
