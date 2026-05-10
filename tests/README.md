# Test Suite Map

This directory is the source-of-truth automated verification area. It should stay focused on tests and reusable test artifacts, while `docs/evidence/` explains which test and evaluation outputs are useful for external review.

## Structure

| Path | Role |
| --- | --- |
| `unit/` | Focused behavior checks for individual modules and helpers. |
| `integration/` | Cross-module checks for backend, pipeline, and persistence behavior. |
| `e2e/` | End-to-end checks that exercise broader workflows from the test runner. |
| `artifacts/` | Fixtures and generated media used by tests, including screenshots, videos, and reports. |
| top-level `test_*.py` | Legacy or focused regression tests that have not been split into subfolders. |

## Operating Rules

- Keep runnable tests here even when their outputs are useful for demos.
- Do not move test files into `docs/` or `eval/reports/`; link to them from the evidence index instead.
- Keep bulky raw outputs and caches ignored unless they are intentionally small, stable fixtures.
- When adding a new user-visible behavior, add the closest lightweight test first, then add broader integration or E2E coverage if the behavior crosses frontend, backend, database, or model boundaries.

## External Evidence

For demos or collaborator handoff, start from `docs/evidence/README.md`. That index explains which tests, scripts, reports, and figures currently support industrial deployment claims and future academic output.
