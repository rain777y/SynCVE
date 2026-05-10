import { test, expect } from '@playwright/test';
import { injectFakeSession, FAKE_USER_ID } from './_helpers/auth';

const BACKEND_RE = /127\.0\.0\.1:5005|localhost:5005/;

/**
 * Slider debounce test for EventSensitivityPanel.
 *
 * Wave 1C claim: dragging the sensitivity slider should fire ONE
 * /events request after the user stops dragging (200ms debounce on input),
 * NOT one per slider tick.
 *
 * To get the panel to mount we stub the relevant backend endpoints rather
 * than driving a real session through /analyze:
 *   - /session/history returns a synthetic session with temporal_summary
 *     populated (so buildReportFromSession returns a non-null report and
 *     SessionReport mounts EventSensitivityPanel).
 *   - /session/<id>/clinical_metrics returns a stable empty payload.
 *   - /session/<id>/events records every request and returns a small fake.
 *
 * The test then opens the session in the modal, drags the slider through
 * 10 positions, and asserts that the burst produced <= 3 /events calls.
 */

const FAKE_SESSION_ID = '11111111-2222-3333-4444-555555555555';

const TEMPORAL_SUMMARY = {
  fps_estimate: 1.0,
  frame_count: 60,
  stability_score: 0.72,
  smoothed_timeline: Array.from({ length: 60 }, (_, i) => ({
    frame: i,
    timestamp: i,
    happy: 0.4 + 0.05 * Math.sin(i / 4),
    sad: 0.1,
    angry: 0.05,
    fear: 0.05,
    surprise: 0.05,
    disgust: 0.05,
    neutral: 0.3,
    dominant: 'happy',
  })),
  events: [],
  transitions: [],
  transition_count: 0,
  trends: [],
  duration_sec: 60,
};

const FAKE_HISTORY = {
  sessions: [
    {
      id: FAKE_SESSION_ID,
      user_id: FAKE_USER_ID,
      created_at: '2026-05-09T08:00:00Z',
      ended_at: '2026-05-09T08:01:00Z',
      last_event_at: '2026-05-09T08:01:00Z',
      status: 'completed',
      summary: 'E2E debounce fixture session.',
      recommendations: null,
      metadata: {},
      temporal_summary: TEMPORAL_SUMMARY,
    },
  ],
};

const FAKE_CLINICAL_METRICS = {
  session_id: FAKE_SESSION_ID,
  events: [],
  valence_trace: [],
  affect_blunting: { value: 0, confidence: 0 },
  reactivity: null,
  suppression: null,
  incongruence: null,
  per_detector_reliability: {},
};

test.describe('EventSensitivityPanel debounces slider input', () => {
  test('drag fires <= 3 /events requests', async ({ page }) => {
    await injectFakeSession(page);

    // Stub /session/history to return a session with temporal_summary so the
    // report modal renders a non-fallback SessionReport.
    await page.route(BACKEND_RE, async (route) => {
      const url = route.request().url();
      const method = route.request().method();

      if (url.includes('/session/history')) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FAKE_HISTORY),
        });
      }
      if (url.includes(`/session/${FAKE_SESSION_ID}/clinical_metrics`)) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(FAKE_CLINICAL_METRICS),
        });
      }
      if (url.includes(`/session/${FAKE_SESSION_ID}/events`)) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            events: [],
            event_count: 0,
            samples: 60,
            method: 'ensemble',
            z_threshold: 2.5,
            min_magnitude: 0.1,
            consensus_min_methods: 2,
            fps_estimate: 1.0,
          }),
        });
      }
      // Pass through anything else (e.g. /health).
      return route.fallback();
    });

    // Count /events requests AFTER history is rendered.
    const eventsCalls: { url: string; t: number }[] = [];
    const t0 = Date.now();
    page.on('request', (req) => {
      const u = req.url();
      if (/\/session\/[^/]+\/events/.test(u)) {
        eventsCalls.push({ url: u, t: Date.now() - t0 });
      }
    });

    await page.goto('/history');
    await page.waitForLoadState('networkidle');

    // Should see exactly one row from the fake history.
    const row = page.locator('.sessions-table__row').first();
    await expect(row).toBeVisible({ timeout: 5000 });
    await row.click();

    // Wait for the SessionReport modal AND the EventSensitivityPanel to mount.
    const slider = page
      .locator('.event-sensitivity-panel input[type=range]')
      .first();
    await expect(slider).toBeVisible({ timeout: 10000 });

    // Wait for the panel's initial fetch to settle.
    await page.waitForTimeout(600);
    const initialCount = eventsCalls.length;

    // Drag through 10 positions in ~800ms.
    const box = await slider.boundingBox();
    expect(box).not.toBeNull();
    if (!box) return;

    const y = box.y + box.height / 2;
    await page.mouse.move(box.x + 5, y);
    await page.mouse.down();
    for (let i = 0; i < 10; i++) {
      const fx = box.x + ((box.width - 10) * (i + 1)) / 10;
      await page.mouse.move(fx, y);
      await page.waitForTimeout(80);
    }
    await page.mouse.up();

    // Wait debounce window + a margin.
    await page.waitForTimeout(500);

    const dragCalls = eventsCalls.length - initialCount;
    expect(
      dragCalls,
      `Slider drag fired ${dragCalls} /events requests; expected <=3 (debounce should collapse ~10 onChange events into 1).`,
    ).toBeLessThanOrEqual(3);
  });
});
