# Methodology — Real-Time Clinical UI: System Contribution

> Companion to `docs/clinical_metrics.md` and `docs/use_cases.md`. This
> chapter documents the engineering methodology of the system layer
> (Axis 4 of the project roadmap). The research-method chapter — which
> covers the event detector, uncertainty-aware fusion, and the
> human-baseline evaluation protocol — is presented separately.

---

## 1. Introduction and Motivation

A clinically usable interview-analysis system is more than its inference
pipeline. The supervisor's review of the prior milestone summarised the
gap concisely: "a car without a navigation system." A backend that
computes `valence_drift`, `affect_blunting`, and `reaction_latency`
(see `docs/clinical_metrics.md`) provides no clinical value if the
clinician cannot reach those numbers, calibrate them, or export them to
a report. Axis 4 of the roadmap (`dev/reference/roadmap.md`) therefore
treats the clinician-facing surface as a first-class deliverable rather
than a presentation layer.

This chapter is scoped strictly as a **system contribution**: the
concurrency, cancellation, caching, asynchronous job-handling, and
end-to-end test architecture that allow the research methodology of the
Method chapter to be exercised against a live clinical session. No
research novelty is claimed for the work described here; the
contribution is the engineering discipline that makes the methodology
reproducible, observable, and demonstrable in a five-minute defence.
The chapter explicitly excludes the event-detection algorithm, the
fusion scheme, and the evaluation protocol, which belong to the Method
chapter.

## 2. System Requirements

The clinical-interview workload imposes three concurrent timing
regimes that the system must service simultaneously without head-of-line
blocking:

1. **Real-time inference cadence.** The detection loop captures and
   analyses one frame approximately every two seconds and is expected to
   render sub-second feedback in the UI when network and inference
   latency permit. A queue-based design that allows analysis requests to
   pile up under transient slowness is unacceptable, because the user
   perceives stale frames as broken liveness.
2. **Multi-second LLM-backed reporting.** The Gemini-backed clinical
   report (`src/backend/clinical_report.py`, `gemini_client.py`) and the
   image-generation path may take 4–60 seconds per call. These must
   coexist with the per-frame loop and with interactive widgets such as
   the event-sensitivity sliders.
3. **Interactive review of completed sessions.** Use Case 3 in
   `docs/use_cases.md` requires the clinician to scrub timelines, retune
   detection thresholds and re-export reports without re-running the
   pipeline. Each scrub or slider movement must produce a perceptibly
   immediate response.

These requirements derive directly from the three use cases in
`docs/use_cases.md`. Use Case 1 (trigger-word micro-expression capture)
fixes the per-frame latency budget; Use Case 2 (depression screening)
demands stable concurrent metric computation; Use Case 3 (clinician
fatigue assist) demands fluid post-session review.

## 3. Architecture Overview

The system is partitioned into a thin React 18 frontend and a Flask 3
backend wrapping DeepFace and a Supabase persistence layer:

```
+----------------------------------------------------+
|  React 18 (browser)                                |
|  -- pages: Detection, History                      |
|  -- hooks: useApiClient, useDetectionLoop,         |
|            useDetectionSession                     |
|  -- review surface: EventSensitivityPanel,         |
|                     TimelineView, SessionReport    |
+----------------------------------------------------+
                |  HTTP (JSON, base64 frames)
                v
+----------------------------------------------------+
|  Flask 3 + Werkzeug threaded server                |
|  -- routes.py    (HTTP -> service)                 |
|  -- service.py   (DeepFace ensemble inference)     |
|  -- session_manager.py (lifecycle, LRU caches)     |
|  -- job_queue.py (ThreadPoolExecutor, /jobs)       |
|  -- clinical_report.py, clinical_metrics.py        |
|  -- gemini_client.py (LLM + image)                 |
|  -- storage.py (Supabase)                          |
+----------------------------------------------------+
```

Layer separation is enforced by import direction: `routes.py` imports
from `session_manager.py` and `service.py`; the latter never import from
the route layer. The clinical-review surface in the frontend
(`EventSensitivityPanel`, `TimelineView`, `SessionReport`) consumes only
the read-side endpoints `/session/<id>/events`,
`/session/<id>/clinical_metrics`, and `/session/<id>/clinical_report`,
so a clinician reviewing an old session never causes inference work.

## 4. Wave 1 — Concurrency, Cancellation, Caching

Wave 1 addressed the failure modes that emerged once the UI exposed
multiple interactive controls against a single Flask process: requests
serialised on the dev server, slider drags spammed the backend, and
unmounted modals continued fetching against a closed session.

### 4.1 Threaded Request Handling

The Werkzeug entry point in `src/backend/app.py:142` runs with
`threaded=True`, allowing each HTTP request to occupy its own worker
thread. The flood test (`scripts/flood_e2e.py`) issues four concurrent
PDF-report requests and computes a `parallel_score` defined as
`sum(per_request_ms) / wall_ms`; a value approaching the worker count
indicates true parallelism. Across the four flood runs in
`eval/reports/flood_e2e_*.json`, `parallel_score` is 3.29, 3.37, 3.99
and 4.0 against four workers — i.e. the backend is genuinely parallel
under read-heavy load and is not bottlenecked by the GIL because the
expensive sub-steps (Supabase round-trips, PDF rendering) release it.

### 4.2 LRU Caches Keyed on `(session_id, frame_count, params)`

Both `/session/<id>/events` and `/session/<id>/clinical_metrics` are
backed by a 64-entry FIFO cache (`_events_cache`, `_clinical_cache` in
`src/backend/session_manager.py:127–149`). The cache key includes
`analyzer._frame_count`:

```python
cache_key = (
    session_id,
    analyzer._frame_count,
    method_eff, z_eff, mag_eff, consensus_eff,
)   # session_manager.py:341
```

Embedding the frame count in the key gives **automatic invalidation
without explicit cache-busting**: every newly logged frame increments
`_frame_count` (`temporal_analysis.py:217`), so any cached result keyed
on a stale count is simply never consulted again and is evicted by FIFO
pressure. For a *completed* session — where `_frame_count` is stable —
identical slider re-queries hit the cache. The instrumentation in the
flood test reports cold/warm latencies in the 4–30 ms range; the
warm/cold ratio is unstable at this scale (occasionally `< 1`) because
absolute latencies are dominated by Supabase variance rather than by
the cached compute itself, but the absolute path remains in the
single-digit milliseconds.

### 4.3 Centralised React Fetch Client (`useApiClient`)

Every `fetch` in the frontend now flows through the
`useApiClient` hook (`src/frontend/src/hooks/useApiClient.js`). The
hook wraps the native `fetch` with three guarantees:

- **Per-call `AbortController`.** Each call registers its controller in
  a hook-local `Set`; on component unmount the cleanup effect calls
  `abortAll()` and aborts every in-flight request
  (`useApiClient.js:114–127`).
- **Replace-latest by `key`.** When a caller passes `opts.key`, any
  previous in-flight call sharing the same key is aborted before the
  new call dispatches (`useApiClient.js:32–35`). This is the primitive
  that makes slider drags emit a single useful response.
- **Uniform result shape.** `{ ok, status, data, error, aborted? }` —
  the explicit `aborted` flag lets callers distinguish a superseded
  fetch from a real failure and skip state updates without try/catch
  noise.

The detection loop reuses the same client (`useDetectionLoop.js:29`),
so navigation away from the Detection page aborts the in-flight
`/analyze` rather than letting it complete and write into an unmounted
component (verified by `e2e/tests/network-cancel-on-nav.spec.ts`).

### 4.4 Input-Level Debouncing

A common React pattern is to debounce *inside* a `useEffect` that fires
on state change:

```javascript
useEffect(() => {
  const t = setTimeout(refetch, 200);
  return () => clearTimeout(t);
}, [zThreshold]);     // effect-level debounce
```

This is insufficient. Because React batches state updates, every slider
tick still produces a state commit, which triggers the effect; the
timer is reset, but a *prior* call may already be in flight. The
implementation in `EventSensitivityPanel.jsx:53–123` therefore combines
**both** levels of debouncing:

1. The slider `onChange` writes the new value into both component
   state *and* a `pendingParamsRef` mutable ref. The ref tracks the
   latest user intent independently of React's render cycle.
2. The 200 ms `setTimeout` reads from the ref at fire time, so the
   request is issued with the most recent slider position rather than
   the value captured when the timer was scheduled.
3. The fetch itself is gated by an `AbortController`; if a newer fetch
   starts before the previous resolves, the previous is aborted.

The Playwright test `e2e/tests/slider-debounce.spec.ts` drives ten
slider positions over ~800 ms and asserts that the resulting burst
yields at most three `/events` requests, capturing the regression-proof
behaviour quantitatively.

### 4.5 Self-Rescheduling Frame Loop

The original detection loop used `setInterval(captureAndAnalyze, 2000)`.
On a slow network — where a single `/analyze` may take 5–8 seconds — a
fixed interval guarantees that timer firings stack faster than they can
be serviced, even with an `isProcessingRef` guard, which silently drops
work and decouples UI cadence from real backend health. The replacement
in `useDetectionLoop.js:108–119` is a self-rescheduling chain:

```javascript
const scheduleNext = useCallback(async () => {
  const startedAt = Date.now();
  try { await captureAndAnalyze(); }
  finally {
    const elapsed = Date.now() - startedAt;
    const delay = Math.max(0, interval - elapsed);
    timeoutRef.current = setTimeout(scheduleNext, delay);
  }
}, [captureAndAnalyze, interval]);
```

The next analysis is scheduled only after the previous resolves; the
delay is `interval - elapsed`, clipped at zero. This absorbs network
latency without queue accumulation: a 5 s response on a 2 s nominal
interval simply means the next tick fires immediately, and the cadence
self-aligns to actual response time. On unmount the chain is broken by
`loopActiveRef.current = false`, the pending timer is cleared, and any
in-flight `/analyze` is aborted via `abortAll()` (lines 144–152).

## 5. Wave 2 — Asynchronous Long-Running Reports

Wave 1 made the synchronous endpoints behave well, but it could not
mask the fact that `/session/stop` synchronously runs the entire
report-generation pipeline, including a Gemini call. The early
flood-run baseline in `eval/reports/flood_e2e_1778305713.json` recorded
a synchronous stop wall time of **16 075 ms** (`fallback_sync_stop`).
For 16 seconds the request thread was unavailable, the browser could
not display any progress, and any other client request issued during
that window had to wait.

Wave 2 introduced an in-process `ThreadPoolExecutor(max_workers=4)` job
queue (`src/backend/job_queue.py`). The executor is module-scoped and
created lazily; jobs are tracked in a 200-entry FIFO `OrderedDict`
under a single mutex (`_jobs_lock`), with eviction by insertion order
once the cap is reached (`job_queue.py:46–48`). The HTTP surface gains
two endpoints:

- `POST /session/stop_async` (`routes.py:388–403`) submits
  `session_manager.stop_session(...)` to the pool and returns
  `{"job_id": ..., "status": "pending"}` with HTTP **202 Accepted**.
- `GET /jobs/<job_id>` (`routes.py:426–433`) returns the live
  `{ status, result, error, elapsed_ms }` snapshot.

The frontend polls via `useApiClient.pollJob` at a 1500 ms cadence
(`useApiClient.js:141–172`); each poll is itself a registered
`AbortController`, so unmount cancels the polling chain. The user-facing
session-stop UI displays a live elapsed counter while the job runs,
preserving perceived progress despite the request having returned in
under 30 ms.

Two architectural benefits follow. First, the Werkzeug request-thread
budget is preserved — `/analyze` and `/events` continue to flow even
while four reports are generating. Second, because polling is
non-blocking the *user-perceived* progress (a counter plus a status
chip) and the *backend resource cost* (one thread out of four,
amortised over the job's actual duration) are decoupled.

## 6. Validation Strategy

Validation is intentionally split across two harnesses because the
failure modes themselves split that way: the backend must hold under
concurrent load, while the frontend must avoid emitting wasteful
requests in the first place.

**Backend — scripted flood test.** `scripts/flood_e2e.py` is a
self-contained Python script (`requests` + `concurrent.futures`) that
boots a fresh session, submits 20 `/analyze` requests across 8 workers
using a real face image, then exercises `/events` and
`/clinical_metrics` cold/warm with the same parameters, runs four
concurrent `/clinical_report?format=pdf` calls to compute the
`parallel_score`, and finally drives `/session/stop_async` to job
completion while measuring `submit_ms` and the total polled latency.
Each run dumps a timestamped JSON to `eval/reports/flood_e2e_*.json` so
runs are directly comparable.

**Frontend — Playwright CLI suite.** The `e2e/tests` directory contains
six spec files (`auth-bypass`, `console-clean`, `empty-states`,
`network-cancel-on-nav`, `slider-debounce`, `visual`) executed via the
Playwright CLI. Two cases warrant note:

- `network-cancel-on-nav.spec.ts` verifies that the Wave 1
  `abortAll()` contract holds: navigating away from `/detection`
  cancels the in-flight `/analyze`.
- `slider-debounce.spec.ts` demonstrates the input-level debounce
  claim. Because driving real frames through `/analyze` would make the
  test slow and flaky, the spec **stubs** `/session/history` to return
  a session with a populated `temporal_summary` (60 frames of sinusoidal
  happiness), which is sufficient for `SessionReport` to mount the
  `EventSensitivityPanel`. The slider is then dragged through ten
  positions and the count of intercepted `/events` requests is
  asserted.

Each quantitative claim in this chapter is bound to a measurement
artifact. Concurrency claims map to `parallel_score` in the JSON
report; cancellation claims map to a Playwright spec; latency claims
map to median fields in the same JSON.

## 7. Results

Numbers below are taken from `eval/reports/flood_e2e_1778306611.json`
(canonical Wave 2 run), with corroborating values from the three earlier
flood runs in the same directory.

| Metric                                  | Before  | After (Wave 1+2) | Source                               |
|-----------------------------------------|---------|------------------|--------------------------------------|
| `/health` median latency                | n/a     | 6 ms             | `flood_e2e_1778306611.json` `health` |
| Concurrent PDF parallel\_score (4 calls)| ~1.0    | 3.29 – 3.99      | `concurrent_pdf.parallel_score`      |
| Synchronous `/session/stop` wall time   | 16 075 ms | n/a (deprecated) | `flood_e2e_1778305713.json`        |
| `/session/stop_async` submit latency    | n/a     | 20 – 28 ms       | `async_stop.submit_ms`               |
| Slider-drag `/events` requests (10 ticks)| ≥ 10   | ≤ 3              | `e2e/tests/slider-debounce.spec.ts`  |

`/health` returns in 6 ms once the warmup phase has completed
(`app.py:114–135`), which is consistent with a thread-pooled Werkzeug
process under no GIL contention. The async stop submit is three orders
of magnitude faster than the prior synchronous path (28 ms versus
16 075 ms), and the underlying job continues to run on the background
pool in 4–15 s (`elapsed_ms_in_job`). The full Playwright suite (six
spec files, eight tests) passes in CI against a freshly built React
bundle and a backend started by the same harness.

## 8. Limitations and Future Work

Three honest limitations follow from the choices above. First, the
backend is currently served by Werkzeug's development server; while
`threaded=True` is sufficient for a defence demo and the flood-test
workload, production deployment should switch to a Gunicorn + gevent
configuration (or uvicorn + Hypercorn behind a Flask-ASGI shim) to
obtain proper worker isolation and graceful restarts. Second, the job
queue is an in-process `ThreadPoolExecutor` with a 200-entry registry;
a multi-worker deployment would need to externalise this to Redis +
Celery or to Postgres advisory locks so that `/jobs/<id>` resolves
across workers. Third, the slider-debounce Playwright case relies on a
stubbed `temporal_summary` to mount the panel cheaply; a higher-fidelity
integration test that streams 30 real frames through `/analyze` and
then drives the same slider would close the loop on Use Case 3 but
sits outside the current sprint. A minor follow-up is to unify the
five report-generation endpoints (`/clinical_report`, `/report/emotion`,
`/report/visual`, `/stop_async`, `/pause_async`) behind a single
`/report` resource that takes a `kind` discriminator.

## 9. References to Repository Artifacts

- `src/backend/job_queue.py` — `ThreadPoolExecutor` + bounded job registry.
- `src/backend/session_manager.py:120–149` — LRU caches keyed on `frame_count`.
- `src/frontend/src/hooks/useApiClient.js` — abort-aware fetch client.
- `src/frontend/src/hooks/useDetectionLoop.js:108–119` — self-rescheduling loop.
- `src/frontend/src/components/EventSensitivityPanel.jsx:53–123` — input-level debounce.
- `e2e/tests/slider-debounce.spec.ts`, `e2e/tests/network-cancel-on-nav.spec.ts`.
- `scripts/flood_e2e.py`, `eval/reports/flood_e2e_*.json`.
