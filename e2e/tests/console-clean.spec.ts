import { test, expect } from '@playwright/test';

const PAGES = ['/', '/detection', '/history'];

// Errors we expect/tolerate in dev: missing favicons, Supabase no-creds,
// CRA manifest probes, ResizeObserver loop noise, etc.
const BENIGN = [
  /favicon/i,
  /manifest/i,
  /supabase/i,
  /ResizeObserver loop/i,
  // 401 from auth/v1 if helper is bypassed
  /auth\/v1/i,
];

function isBenign(text: string): boolean {
  return BENIGN.some((rx) => rx.test(text));
}

test.describe('Console hygiene (no auth)', () => {
  for (const p of PAGES) {
    test(`page ${p} reports no console errors`, async ({ page }) => {
      const errors: string[] = [];

      page.on('pageerror', (e) => {
        if (!isBenign(e.message)) errors.push(`pageerror: ${e.message}`);
      });
      page.on('console', (msg) => {
        if (msg.type() === 'error') {
          const text = msg.text();
          if (!isBenign(text)) errors.push(`console.error: ${text}`);
        }
      });

      await page.goto(p);
      await page.waitForLoadState('networkidle');
      // Give React effects a beat to settle.
      await page.waitForTimeout(300);

      expect(
        errors,
        `Found unexpected errors on ${p}:\n${errors.join('\n')}`,
      ).toEqual([]);
    });
  }
});
