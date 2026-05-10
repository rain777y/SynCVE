import { test } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const SHOTS_DIR = path.join(__dirname, '..', 'test-results', 'screenshots');
fs.mkdirSync(SHOTS_DIR, { recursive: true });

const PAGES: Array<{ url: string; name: string }> = [
  { url: '/', name: 'home' },
  { url: '/detection', name: 'detection' },
  { url: '/history', name: 'history' },
];

test.describe('Visual snapshots (no auth)', () => {
  for (const p of PAGES) {
    test(`fullpage screenshot of ${p.url}`, async ({ page }) => {
      await page.goto(p.url);
      await page.waitForLoadState('networkidle');
      await page.waitForTimeout(300);
      await page.screenshot({
        path: path.join(SHOTS_DIR, `visual-${p.name}.png`),
        fullPage: true,
      });
    });
  }
});
