"""
End-to-end flooding test for SynCVE backend after Wave 1 UX overhaul.

Validates:
  1. Backend stays responsive under concurrent /analyze load (threaded=True).
  2. /events cache speeds up repeated identical params (LRU cache).
  3. /clinical_metrics cache speeds up repeated calls.
  4. /clinical_report PDF + MD work and don't block other requests.
  5. New async stop / pause endpoints + /jobs/<id> polling.
  6. 4 concurrent PDF requests don't serialize.

Run:
    E:/conda/envs/SynCVE/python.exe scripts/flood_e2e.py

Output:
    eval/reports/flood_e2e_<ts>.json
"""
from __future__ import annotations

import base64
import concurrent.futures
import json
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BACKEND = os.environ.get("SYNCVE_BACKEND", "http://127.0.0.1:5005")
FACE_IMG = (
    ROOT
    / "dev"
    / "reference"
    / "libraries"
    / "deepface"
    / "deepface-repo"
    / "tests"
    / "unit"
    / "dataset"
    / "img1.jpg"
)
REPORT_DIR = ROOT / "eval" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _b64_face() -> str:
    raw = FACE_IMG.read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def post(path: str, json_body: dict, timeout: float = 60) -> tuple[int, dict | bytes, float]:
    t0 = time.perf_counter()
    r = requests.post(BACKEND + path, json=json_body, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    try:
        body = r.json()
    except Exception:
        body = r.content
    return r.status_code, body, dt


def get(path: str, timeout: float = 60) -> tuple[int, dict | bytes, float]:
    t0 = time.perf_counter()
    r = requests.get(BACKEND + path, timeout=timeout)
    dt = (time.perf_counter() - t0) * 1000
    try:
        body = r.json()
    except Exception:
        body = r.content
    return r.status_code, body, dt


def stats(arr: list[float]) -> dict:
    if not arr:
        return {"n": 0}
    return {
        "n": len(arr),
        "min": round(min(arr), 1),
        "p50": round(statistics.median(arr), 1),
        "p95": round(sorted(arr)[int(0.95 * (len(arr) - 1))], 1),
        "max": round(max(arr), 1),
        "mean": round(statistics.mean(arr), 1),
    }


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_health() -> dict:
    print("\n=== HEALTH ===")
    s, b, dt = get("/health", timeout=10)
    print(f"  /health  -> {s} ({dt:.0f}ms)")
    return {"status": s, "latency_ms": round(dt), "ok": s == 200}


def start_session() -> str | None:
    s, b, dt = post(
        "/session/start",
        {"user_id": "flood_test", "metadata": {"source": "flood_e2e"}},
    )
    print(f"  start    -> {s} ({dt:.0f}ms) sid={(b or {}).get('session_id', '?') if isinstance(b, dict) else '?'}")
    if s != 200 or not isinstance(b, dict):
        return None
    return b.get("session_id")


def test_analyze_burst(session_id: str, n: int = 20, workers: int = 8) -> dict:
    print(f"\n=== /analyze BURST  (n={n}, workers={workers}) ===")
    img = _b64_face()
    payload = {
        "img": img,
        "session_id": session_id,
        "actions": ["emotion"],
        "detector_backend": "retinaface",
        "align": True,
        "anti_spoofing": False,
        "enforce_detection": True,
    }
    latencies = []
    statuses = []
    errors = []

    def one():
        s, b, dt = post("/analyze", payload, timeout=60)
        return s, b, dt

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one) for _ in range(n)]
        wall_t0 = time.perf_counter()
        for f in concurrent.futures.as_completed(futs):
            s, b, dt = f.result()
            latencies.append(dt)
            statuses.append(s)
            if s != 200:
                errors.append({"status": s, "body": b if isinstance(b, dict) else str(b)[:80]})
        wall = (time.perf_counter() - wall_t0) * 1000

    success = sum(1 for s in statuses if s == 200)
    print(f"  success: {success}/{n}  wall={wall:.0f}ms  per-req {stats(latencies)}")
    if errors:
        print(f"  first errors: {errors[:2]}")
    return {
        "n": n,
        "workers": workers,
        "success": success,
        "wall_ms": round(wall),
        "per_request": stats(latencies),
        "first_errors": errors[:3],
        # Threaded check: total wall < n * mean / workers * 1.5 means parallelism ok
    }


def test_events_cache(session_id: str) -> dict:
    print("\n=== /events  (slider drag simulation) ===")
    # Phase 1: 10 unique params (cold path)
    cold = []
    for i in range(10):
        z = round(2.0 + 0.05 * i, 2)
        s, b, dt = get(f"/session/{session_id}/events?method=ensemble&z_threshold={z}&min_magnitude=0.10&consensus_min_methods=2")
        cold.append(dt)
    # Phase 2: repeat the SAME 10 (warm cache)
    warm = []
    for i in range(10):
        z = round(2.0 + 0.05 * i, 2)
        s, b, dt = get(f"/session/{session_id}/events?method=ensemble&z_threshold={z}&min_magnitude=0.10&consensus_min_methods=2")
        warm.append(dt)
    print(f"  cold: {stats(cold)}")
    print(f"  warm: {stats(warm)}")
    speedup = (statistics.median(cold) / statistics.median(warm)) if warm and statistics.median(warm) > 0 else None
    print(f"  median speedup (cold/warm): {speedup}")
    return {"cold": stats(cold), "warm": stats(warm), "speedup_x": round(speedup, 2) if speedup else None}


def test_clinical_metrics(session_id: str) -> dict:
    print("\n=== /clinical_metrics  (cache check) ===")
    # First call computes; subsequent identical calls should hit cache.
    # Use 10 samples each phase so noise is averaged out.
    s, b, dt0 = post(f"/session/{session_id}/clinical_metrics", {})
    print(f"  first  -> {s} ({dt0:.0f}ms)  [primes the cache]")
    cold = []
    for _ in range(10):
        s, b, dt = post(f"/session/{session_id}/clinical_metrics", {"triggers": [{"word": str(_), "frame_idx": _, "t_sec": float(_)}]})
        cold.append(dt)
    warm = []
    for _ in range(10):
        s, b, dt = post(f"/session/{session_id}/clinical_metrics", {"triggers": [{"word": str(_), "frame_idx": _, "t_sec": float(_)}]})
        warm.append(dt)
    print(f"  cold: {stats(cold)}")
    print(f"  warm: {stats(warm)}")
    speedup = (statistics.median(cold) / statistics.median(warm)) if warm and statistics.median(warm) > 0 else None
    return {"cold": stats(cold), "warm": stats(warm), "speedup_x": round(speedup, 2) if speedup else None}


def test_clinical_report(session_id: str) -> dict:
    print("\n=== /clinical_report ===")
    s_md, _, dt_md = get(f"/session/{session_id}/clinical_report?format=md")
    s_pdf, body_pdf, dt_pdf = get(f"/session/{session_id}/clinical_report?format=pdf")
    pdf_bytes = len(body_pdf) if isinstance(body_pdf, (bytes, bytearray)) else 0
    print(f"  md  -> {s_md} ({dt_md:.0f}ms)")
    print(f"  pdf -> {s_pdf} ({dt_pdf:.0f}ms, {pdf_bytes}B)")
    return {
        "md_status": s_md, "md_ms": round(dt_md),
        "pdf_status": s_pdf, "pdf_ms": round(dt_pdf), "pdf_bytes": pdf_bytes,
    }


def test_concurrent_pdf(session_id: str, n: int = 4) -> dict:
    """Verify threaded=True actually parallelizes by running 4 PDF renders simultaneously."""
    print(f"\n=== concurrent PDF (n={n}, proves threaded=True) ===")
    def one():
        try:
            return get(f"/session/{session_id}/clinical_report?format=pdf")
        except Exception as e:
            return (-1, {"exc": str(e)[:120]}, 0.0)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: one(), range(n)))
    wall = (time.perf_counter() - t0) * 1000
    per_req = [r[2] for r in results if r[0] >= 0]
    statuses = [r[0] for r in results]
    if not per_req:
        per_req = [0.0]
    print(f"  wall={wall:.0f}ms  per-req {stats(per_req)}  statuses={statuses}")
    # If serial: wall ≈ n * median(per_req). If parallel: wall ≈ max(per_req).
    serial_estimate = n * statistics.median(per_req)
    parallel_score = round(serial_estimate / wall, 2) if wall > 0 else None
    print(f"  parallel_score (>1.5 means threading works): {parallel_score}")
    return {
        "n": n, "wall_ms": round(wall),
        "per_request": stats(per_req),
        "parallel_score": parallel_score,
        "all_success": all(s == 200 for s in statuses),
    }


def test_async_stop(session_id: str) -> dict:
    print("\n=== /session/stop_async + /jobs polling ===")
    s, b, dt = post("/session/stop_async", {"session_id": session_id})
    if s == 404:
        print("  endpoint not implemented (Wave 1A may not be done) — skipping")
        return {"skipped": True, "reason": "endpoint not found"}
    if s not in (200, 202) or not isinstance(b, dict):
        print(f"  unexpected: {s} {b}")
        return {"skipped": True, "reason": f"submit failed {s}"}
    job_id = b.get("job_id")
    print(f"  job_id={job_id}  submit={dt:.0f}ms")
    polls = 0
    poll_ms = []
    final = None
    t0 = time.perf_counter()
    while polls < 60:
        sj, bj, dtj = get(f"/jobs/{job_id}")
        poll_ms.append(dtj)
        polls += 1
        if isinstance(bj, dict) and bj.get("status") in ("done", "error"):
            final = bj
            break
        time.sleep(0.5)
    total = (time.perf_counter() - t0) * 1000
    print(f"  polls={polls}  total={total:.0f}ms  final.status={(final or {}).get('status')}")
    return {
        "submit_ms": round(dt),
        "polls": polls,
        "total_ms": round(total),
        "final_status": (final or {}).get("status"),
        "elapsed_ms_in_job": (final or {}).get("elapsed_ms"),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    out = {"backend": BACKEND, "ts": datetime.now().isoformat()}
    out["health"] = test_health()
    if not out["health"].get("ok"):
        print("Backend not healthy — aborting");
        (REPORT_DIR / f"flood_e2e_{int(time.time())}.json").write_text(json.dumps(out, indent=2))
        return out

    print("\n=== START SESSION ===")
    sid = start_session()
    if not sid:
        print("Could not start session — aborting")
        out["error"] = "start_session_failed"
        (REPORT_DIR / f"flood_e2e_{int(time.time())}.json").write_text(json.dumps(out, indent=2))
        return out
    out["session_id"] = sid

    out["analyze_burst"] = test_analyze_burst(sid, n=20, workers=8)
    out["events_cache"] = test_events_cache(sid)
    out["clinical_metrics_cache"] = test_clinical_metrics(sid)
    out["clinical_report"] = test_clinical_report(sid)
    out["concurrent_pdf"] = test_concurrent_pdf(sid, n=4)
    out["async_stop"] = test_async_stop(sid)

    # If async_stop was skipped, fall back to sync stop so we don't leak the session
    if out["async_stop"].get("skipped"):
        s, b, dt = post("/session/stop", {"session_id": sid}, timeout=120)
        out["fallback_sync_stop"] = {"status": s, "ms": round(dt)}

    fp = REPORT_DIR / f"flood_e2e_{int(time.time())}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\n=== REPORT WRITTEN ===\n  {fp}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
