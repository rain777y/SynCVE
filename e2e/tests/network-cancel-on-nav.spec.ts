import { test, expect } from '@playwright/test';
import { injectFakeSession } from './_helpers/auth';

/**
 * Goal: confirm that navigating away cancels in-flight fetches.
 *
 * The webcam / /analyze flow can't be exercised headless without a real
 * camera, so we verify the simpler invariant: navigating between
 * /detection and /history produces no orphan network errors and no
 * console errors after the page settles.
 *
 * If the app correctly aborts pending requests on unmount we expect:
 *   - no console.error rings after navigation
 *   - the new route renders cleanly
 */
test.describe('Navigation cancels in-flight work', () => {
  test('round-trip /detection -> /history -> /detection is clean', async ({
    page,
  }) => {
    await injectFakeSession(page);

    const errors: string[] = [];
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        const t = msg.text();
        if (!/favicon|manifest|supabase/i.test(t)) {
          errors.push(`console.error: ${t}`);
        }
      }
    });

    // Stub /analyze to delay 5s so we have an in-flight request to cancel.
    await page.route('**/analyze**', async (route) => {
      await new Promise((r) => setTimeout(r, 5000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ emotions: {}, dominant_emotion: 'neutral' }),
      });
    });

    await page.goto('/detection');
    await page.waitForLoadState('networkidle');

    // Click through consent if visible — gets us to the control bench.
    const consent = page.getByRole('button', { name: /i understand/i });
    if (await consent.isVisible().catch(() => false)) {
      await consent.click();
    }

    // Navigate away while fetches are likely in-flight.
    await page.goto('/history');
    await page.waitForLoadState('networkidle');

    // And back.
    await page.goto('/detection');
    await page.waitForLoadState('networkidle');

    // Allow any abort fallout to surface.
    await page.waitForTimeout(1000);

    expect(
      errors,
      `Unexpected errors during navigation:\n${errors.join('\n')}`,
    ).toEqual([]);
  });
});
