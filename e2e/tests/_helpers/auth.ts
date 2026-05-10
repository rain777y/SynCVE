import type { Page, Route } from '@playwright/test';

/**
 * Auth bypass helper.
 *
 * SynCVE uses Supabase v2 client, which persists the active session in
 * localStorage under the key `sb-<project-ref>-auth-token`. The Supabase
 * client's getSession() reads localStorage synchronously and (when refresh
 * is needed) calls https://<ref>.supabase.co/auth/v1/...
 *
 * We do two things:
 *   1) Inject a far-future, well-formed session into localStorage BEFORE
 *      navigation via page.addInitScript. This hydrates the AuthContext
 *      without any network round-trip.
 *   2) Stub out *.supabase.co/auth/v1/** so any background refresh calls
 *      return a valid-looking response instead of 401.
 *
 * Project ref is hard-coded to match the running frontend.
 */

const SUPABASE_REF = 'gainvlutpnhyommbkadk';
const STORAGE_KEY = `sb-${SUPABASE_REF}-auth-token`;

const FAKE_USER = {
  id: 'e2e-test-user-uuid',
  email: 'e2e@test.local',
  aud: 'authenticated',
  role: 'authenticated',
  app_metadata: {},
  user_metadata: { full_name: 'E2E Test', avatar_url: '' },
  created_at: '2024-01-01T00:00:00Z',
};

function makeSession() {
  // Year 2099-01-01 in unix seconds.
  const expiresAt = 4070908800;
  return {
    access_token: 'fake-token',
    refresh_token: 'fake-refresh',
    expires_at: expiresAt,
    expires_in: 3600,
    token_type: 'bearer',
    user: FAKE_USER,
  };
}

export async function injectFakeSession(page: Page): Promise<void> {
  const session = makeSession();
  const payload = JSON.stringify(session);

  // 1. Pre-seed localStorage before any app code runs.
  await page.addInitScript(
    ({ key, value }) => {
      try {
        window.localStorage.setItem(key, value);
      } catch (_) {
        // ignored — page may not have storage yet (about:blank)
      }
    },
    { key: STORAGE_KEY, value: payload },
  );

  // 2. Stub Supabase auth network calls so any refresh / getUser succeeds.
  await page.route('**/auth/v1/**', (route: Route) => {
    const url = route.request().url();
    // /auth/v1/token returns a token-shaped response.
    if (url.includes('/auth/v1/token')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          access_token: 'fake-token',
          refresh_token: 'fake-refresh',
          expires_in: 3600,
          token_type: 'bearer',
          user: FAKE_USER,
        }),
      });
    }
    // /auth/v1/user (and similar) return the user directly.
    if (url.includes('/auth/v1/user')) {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(FAKE_USER),
      });
    }
    // /auth/v1/logout etc. — return a generic OK.
    return route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({}),
    });
  });
}

export const FAKE_USER_ID = FAKE_USER.id;
