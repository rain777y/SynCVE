import { test, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

const SHOTS_DIR = path.join(__dirname, '..', 'test-results', 'screenshots');
fs.mkdirSync(SHOTS_DIR, { recursive: true });

test.describe('Empty / unauthenticated states', () => {
  test('home page renders without console errors', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    page.on('console', (msg) => {
      if (msg.type() === 'error') errors.push(`console.error: ${msg.text()}`);
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.screenshot({
      path: path.join(SHOTS_DIR, 'home-empty.png'),
      fullPage: true,
    });

    const filtered = errors.filter(
      (e) => !/favicon|Manifest|Supabase|supabase/i.test(e),
    );
    expect(filtered, `unexpected errors:\n${filtered.join('\n')}`).toEqual([]);
  });

  test('detection page shows sign-in empty state', async ({ page }) => {
    await page.goto('/detection');
    await page.waitForLoadState('networkidle');

    const title = page.locator('.ed-empty-card__title');
    await expect(title).toBeVisible();
    await expect(title).toContainText(/sign in/i);
    await expect(title).toContainText(/begin a session/i);

    await page.screenshot({
      path: path.join(SHOTS_DIR, 'detection-empty.png'),
      fullPage: true,
    });
  });

  test('history page shows sign-in empty state', async ({ page }) => {
    await page.goto('/history');
    await page.waitForLoadState('networkidle');

    const title = page.locator('.ed-empty-card__title');
    await expect(title).toBeVisible();
    await expect(title).toContainText(/sign in/i);
    await expect(title).toContainText(/your sessions/i);

    await page.screenshot({
      path: path.join(SHOTS_DIR, 'history-empty.png'),
      fullPage: true,
    });
  });
});
