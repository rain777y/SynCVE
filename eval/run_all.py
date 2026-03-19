"""
eval/run_all.py — Master Runner for All Evaluation Scripts

Orchestrates the full evaluation suite: baselines, ablation studies,
ensemble optimization, and pipeline comparison.

Usage
-----
    python -m eval.run_all --limit 50              # quick smoke test
    python -m eval.run_all                         # full evaluation
    python -m eval.run_all --skip-baseline         # skip baseline benchmarks
    python -m eval.run_all --skip-slow             # skip slow studies (ensemble, pipeline)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all SynCVE evaluation scripts",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max images per script (0 = all). Passed to all sub-scripts.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline benchmark scripts (FER2013, RAF-DB).",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow scripts (ensemble optimization, pipeline comparison).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Output directory for all results.",
    )
    return parser.parse_args()


def run_script(module: str, extra_args: list, label: str) -> bool:
    """Run a Python module as a subprocess and report status.

    Returns True on success, False on failure.
    """
    cmd = [sys.executable, "-m", module] + extra_args

    print(f"\n{'#' * 70}")
    print(f"# {label}")
    print(f"# Command: {' '.join(cmd)}")
    print(f"{'#' * 70}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            check=False,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"\n  [{label}] PASSED in {elapsed:.1f}s")
            return True
        else:
            print(f"\n  [{label}] FAILED (exit code {result.returncode}) in {elapsed:.1f}s")
            return False

    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\n  [{label}] ERROR: {exc}  ({elapsed:.1f}s)")
        return False


def main() -> None:
    args = parse_args()

    limit_args = ["--limit", str(args.limit)] if args.limit > 0 else []
    output_args = ["--output-dir", args.output_dir]

    # Track results
    results = {}
    overall_t0 = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Baseline benchmarks
    # ------------------------------------------------------------------
    if not args.skip_baseline:
        results["FER2013 Baseline"] = run_script(
            "eval.benchmark_fer2013",
            limit_args + output_args,
            "FER2013 Baseline",
        )
        results["RAF-DB Baseline"] = run_script(
            "eval.benchmark_rafdb",
            limit_args + output_args,
            "RAF-DB Baseline",
        )
    else:
        print("\n  Skipping baseline benchmarks (--skip-baseline)")

    # ------------------------------------------------------------------
    # Phase 2: Ablation studies
    # ------------------------------------------------------------------
    results["Preprocessing Ablation"] = run_script(
        "eval.ablation_preprocess",
        limit_args + ["--output-dir", args.output_dir],
        "Preprocessing Ablation",
    )

    results["Detector Ablation"] = run_script(
        "eval.ablation_detector",
        limit_args + ["--output-dir", args.output_dir],
        "Detector Ablation",
    )

    results["Post-Processing Ablation"] = run_script(
        "eval.ablation_postprocess",
        limit_args + ["--output-dir", args.output_dir],
        "Post-Processing Ablation",
    )

    # ------------------------------------------------------------------
    # Phase 3: Slow studies (ensemble optimization, pipeline comparison)
    # ------------------------------------------------------------------
    if not args.skip_slow:
        ensemble_args = ["--output-dir", args.output_dir]
        if args.limit > 0:
            ensemble_args += ["--train-limit", str(args.limit), "--limit", str(args.limit)]

        results["Ensemble Weight Optimization"] = run_script(
            "eval.optimize_ensemble_weights",
            ensemble_args,
            "Ensemble Weight Optimization",
        )

        results["Pipeline vs Baseline"] = run_script(
            "eval.pipeline_vs_baseline",
            limit_args + ["--output-dir", args.output_dir],
            "Pipeline vs Baseline",
        )
    else:
        print("\n  Skipping slow studies (--skip-slow)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - overall_t0

    print(f"\n\n{'='*70}")
    print("EVALUATION SUITE SUMMARY")
    print(f"{'='*70}")

    passed = 0
    failed = 0
    for label, success in results.items():
        status = "PASS" if success else "FAIL"
        marker = "  " if success else ">>"
        print(f"  {marker} [{status}]  {label}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed + failed}  |  Passed: {passed}  |  Failed: {failed}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")

    # Exit with non-zero if any script failed
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
