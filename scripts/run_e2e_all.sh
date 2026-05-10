#!/usr/bin/env bash
# Unified E2E entrypoint — backend flood + frontend Playwright.
#
# Usage:
#   ./scripts/run_e2e_all.sh
#
# Prerequisites:
#   - SynCVE conda env active (or run.sh-equivalent for the Python interpreter)
#   - Backend running at 127.0.0.1:5005 (./run.sh -m src.backend.app)
#   - Frontend running at localhost:3000 (cd src/frontend && npm start)
#   - Playwright deps installed (cd e2e && npm install && npx playwright install chromium)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-E:/conda/envs/SynCVE/python.exe}"

echo "=== 1. Backend flood test ==="
"$PY" scripts/flood_e2e.py
echo

echo "=== 2. Frontend Playwright suite ==="
( cd e2e && npm test -- --reporter=list )
echo

echo "=== 3. Regenerate validation figure ==="
"$PY" eval/figures/make_wave12_figure.py
echo

echo "=== Done. Reports: ==="
ls -lt eval/reports/flood_e2e_*.json | head -1
echo "  Playwright HTML: e2e/playwright-report/index.html"
echo "  Figure: eval/figures/fig3_wave12_results.png"
