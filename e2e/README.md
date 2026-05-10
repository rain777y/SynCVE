# SynCVE E2E Test Suite

Playwright CLI end-to-end tests covering the React frontend (`localhost:3000`) wired to the Flask backend (`127.0.0.1:5005`).

The suite validates the UX overhaul delivered in Wave 1 and Wave 2 of `docs/methodology_realtime_clinical_ui.md`. It is the user-facing counterpart to the backend flood test in `scripts/flood_e2e.py`.

---

## Prerequisites

The frontend dev server and the Flask backend must be running:

```bash
# Frontend (in src/frontend/)
npm start

# Backend (in repo root)
./run.sh -m src.backend.app
```

Both servers are managed by the developer; the suite intentionally does **not** spawn them itself.

---

## Setup

```bash
cd e2e
npm install              # installs @playwright/test
npx playwright install chromium
```

---

## Running

```bash
npm test                 # headless, list reporter, all 12 specs (~10s)
npm run test:ui          # interactive Playwright UI
npm run report           # open the HTML report in playwright-report/
```

To run a single spec:

```bash
npx playwright test slider-debounce
npx playwright test --grep "@auth-bypass"
```

---

## Layout

```
e2e/
├── playwright.config.ts          chromium-only, 4 workers, 1 retry, 30s timeout
├── tests/
│   ├── _helpers/
│   │   └── auth.ts               injectFakeSession() — Supabase localStorage + auth/v1 stub
│   ├── empty-states.spec.ts      unauthenticated empty-state cards on /, /detection, /history
│   ├── console-clean.spec.ts     no console.error or pageerror across all three pages
│   ├── auth-bypass.spec.ts       EmotionDetector mounts when a fake session is injected
│   ├── network-cancel-on-nav.spec.ts  /detection → /history → /detection round-trip is clean
│   ├── slider-debounce.spec.ts   sensitivity slider drag (10 steps) fires ≤ 3 /events requests
│   └── visual.spec.ts            full-page screenshots
├── playwright-report/            HTML report (regenerated each run)
└── test-results/
    └── screenshots/              6 PNGs from visual + empty-state specs
```

---

## What each spec validates

| Spec | UX claim under test | Wave |
|---|---|---|
| `empty-states` | Unauthenticated users see a guidance card, not an empty page | 1D |
| `console-clean` | No silent errors leak to the console | (regression) |
| `auth-bypass` | Auth context hydrates from localStorage without a network round-trip | (test infra) |
| `network-cancel-on-nav` | Unmount aborts in-flight fetches; no orphaned errors | 1B |
| `slider-debounce` | Input-level debounce collapses 10 onChange events into ≤ 3 /events calls | **1C** |
| `visual` | Frozen baseline screenshots for review | (visual diff) |

---

## Auth bypass strategy

`tests/_helpers/auth.ts` does two things before navigation:

1. **`page.addInitScript`** seeds `localStorage` at `sb-gainvlutpnhyommbkadk-auth-token` with a Supabase v2-shaped session whose `expires_at` is in the year 2099. The frontend's `AuthContext` calls `supabase.auth.getSession()` synchronously from local storage, so it hydrates as authenticated without any network call.

2. **`page.route('**/auth/v1/**')`** stubs Supabase auth endpoints (`/auth/v1/token`, `/auth/v1/user`, `/auth/v1/logout`) with valid-looking responses, so the periodic refresh loop does not 401.

The Flask backend is unaffected — it does not validate Supabase tokens server-side; it only takes a `user_id` query parameter.

---

## Slider-debounce — how it actually mounts the panel

`EventSensitivityPanel` only renders inside the `SessionReport` modal, which only renders when the clicked session has a populated `temporal_summary.smoothed_timeline` (length > 2). Driving a real session via `/analyze` would cost ~30 s on CPU per test run.

To stay headless and fast, the spec stubs the relevant backend endpoints with `page.route(/127\.0\.0\.1:5005|localhost:5005/)`:

- `/session/history` → returns one synthetic session whose `temporal_summary` contains a 60-frame `smoothed_timeline`
- `/session/<id>/clinical_metrics` → returns an empty-but-shaped metrics object
- `/session/<id>/events` → returns zero events; the spec counts how many times this URL is requested

The drag itself uses `page.mouse.down/move/up` over the slider's bounding box.

---

## Adding a new spec

1. Add `tests/<topic>.spec.ts`
2. Import `injectFakeSession` if the page needs auth
3. Use `page.route(BACKEND_RE, ...)` to stub anything that requires session data
4. Run `npx playwright test <topic>` to iterate; the HTML report pinpoints failures

---

## Known gaps

- No spec exercises the camera path (headless Chromium has no webcam; would require a virtual MediaStream)
- No load-test for concurrent users — that lives in `scripts/flood_e2e.py`
- The `/analyze` rate limiter (30/min) means rapid-fire tests within the same minute will see 429s; the suite does not exercise this surface
