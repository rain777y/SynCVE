# SynCVE Evidence Index

This directory is the external-facing map for validation evidence. It does not replace automated tests, evaluation scripts, or raw experiment outputs. It points reviewers to the artifacts that show how SynCVE is being made credible for industrial deployment and later academic output.

## Purpose

Use this index when the project needs to show process, not only final claims:

- Industrial reviewers can inspect deployability, reliability, reporting behavior, and operational tests.
- Research reviewers can inspect methodology, event detection, ablation evidence, and reusable figures.
- Internal contributors can see which files are source tests, which files are curated evidence, and which files should stay as raw local data.

Academic papers and slide decks should be built later from these sources. They should not be mixed into the test suite or raw evaluation cache.

## Repository Zones

| Zone | Role | External use |
| --- | --- | --- |
| `tests/` | Source-of-truth automated verification. | Show coverage structure and key behavior tests. |
| `tests/artifacts/` | Reusable fixtures, videos, screenshots, and generated reports used by tests. | Show concrete inputs and outputs, but avoid treating fixtures as final benchmark results. |
| `e2e/` | Browser-level product workflow tests. | Show that the frontend can exercise backend capabilities through user-visible flows. |
| `scripts/` | Smoke, flood, demo, and operational validation scripts. | Show reproducible checks for backend health, pipeline behavior, and stress paths. |
| `eval/` | Evaluation harness for experiments, ablations, metrics, and figures. | Show research and product validation process. |
| `eval/reports/` | Curated summaries and selected structured run records. | Primary place to pull evidence for demos, handoffs, papers, and slides. |
| `eval/figures/` | Reusable visual outputs from evaluation. | Candidate visuals for future academic papers and presentations. |

## Current Evidence Trail

| Theme | Start here | Supporting files |
| --- | --- | --- |
| Industrial landing potential | `docs/use_cases.md` | `eval/reports/handoff_context.md`, `scripts/backend_smoke.py`, `scripts/flood_e2e.py`, `eval/reports/flood_e2e_1778393619.json` |
| Clinical and report pipeline | `docs/clinical_metrics.md` | `eval/reports/demo_clinical_report.md`, `eval/reports/demo_session_summary.json`, `eval/reports/system_evaluation_wave12.md` |
| Real-time detection workflow | `docs/methodology_realtime_clinical_ui.md` | `tests/test_event_detector.py`, `eval/event_eval.py`, `e2e/tests/` |
| Research novelty and ablation | `eval/reports/ablation_findings.md` | `eval/reports/full_ablation.json`, `eval/reports/phase2_analysis.md`, `eval/figures/fig1_pipeline_overview.png`, `eval/figures/fig2_worked_example.png`, `eval/figures/fig3_wave12_key_results.png` |
| Engineering reliability | `tests/README.md` | `tests/unit/`, `tests/integration/`, `tests/e2e/`, `e2e/tests/`, `eval/reports/flood_e2e_*.json` |

## Recommended Demo Narrative

1. Product problem and landing path: start with `docs/use_cases.md`.
2. Detection and event pipeline: use `docs/methodology_realtime_clinical_ui.md`.
3. Backend and UI evidence: show `scripts/backend_smoke.py`, `scripts/flood_e2e.py`, and selected E2E tests.
4. Output quality: show `eval/reports/demo_clinical_report.md` and `eval/reports/system_evaluation_wave12.md`.
5. Research path: show ablation reports and figures from `eval/reports/` and `eval/figures/`.

## Boundary Rules

- Keep automated test source in `tests/` and `e2e/`.
- Keep experiment code, metrics, and generated figures in `eval/`.
- Keep curated evaluation summaries and selected run records in `eval/reports/`.
- Keep raw cache, datasets, and bulky local outputs out of git.
- Build future papers and slides from this evidence trail instead of duplicating raw artifacts.
