import { test, expect } from '@playwright/test';
import { injectFakeSession } from './_helpers/auth';

test.describe('Auth bypass via injected Supabase session', () => {
  test('detection page mounts EmotionDetector when session is injected', async ({
    page,
  }) => {
    await injectFakeSession(page);

    await page.goto('/detection');
    await page.waitForLoadState('networkidle');

    // The unauthenticated empty-state card must NOT be visible.
    await expect(page.locator('.ed-empty-card__title')).toHaveCount(0);

    // EmotionDetector first renders the consent gate; once consent is given
    // it renders the .ed-bench control panel. The .ed-shell wrapper is
    // present in both states, so its presence is the strongest "component
    // mounted" signal we can assert without webcam access.
    await expect(page.locator('.ed-shell')).toBeVisible();

    // Either consent screen ("I Understand") or post-consent control bench
    // ("Begin Acquisition") should be present.
    const beginBtn = page.getByRole('button', { name: /begin acquisition/i });
    const consentBtn = page.getByRole('button', { name: /i understand/i });

    // Click consent if shown to reach the control bench.
    if (await consentBtn.isVisible().catch(() => false)) {
      await consentBtn.click();
    }

    await expect(beginBtn).toBeVisible({ timeout: 10_000 });
    await expect(page.locator('.ed-bench')).toBeVisible();
  });
});
