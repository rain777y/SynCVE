import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for SynCVE E2E tests.
 *
 * Assumptions:
 *   - Frontend (CRA dev server) is already running at http://localhost:3000
 *   - Backend (Flask) is already running at http://127.0.0.1:5005
 *
 * We do NOT spawn either server from here.
 */
export default defineConfig({
  testDir: './tests',
  outputDir: './test-results',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 1,
  workers: 4,
  timeout: 30_000,
  reporter: [
    ['list'],
    ['html', { outputFolder: './playwright-report', open: 'never' }],
  ],
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'on-first-retry',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'], headless: true },
    },
  ],
});
