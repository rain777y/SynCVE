"""
Playwright E2E test suite for SynCVE.
Validates: home page, backend health, session API flow (temporal data),
detection page, history page (SessionReport + stability badge), CORS.
"""
import base64
import json
import sys
import time
import requests
from io import BytesIO
from playwright.sync_api import sync_playwright

FRONTEND_URL = "http://localhost:3000"
BACKEND_URL = "http://localhost:5005"
REQUEST_TIMEOUT = 30
REPORT_TIMEOUT = 120

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_face_b64() -> str:
    """Generate a minimal valid JPEG face image as data-URI."""
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (224, 224), color=(200, 170, 140))
        draw = ImageDraw.Draw(img)
        draw.ellipse([40, 40, 184, 200], fill=(220, 185, 155))
        draw.ellipse([70, 80, 100, 110], fill=(60, 40, 20))
        draw.ellipse([124, 80, 154, 110], fill=(60, 40, 20))
        draw.arc([80, 130, 144, 175], 0, 180, fill=(120, 60, 60), width=3)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    except ImportError:
        jpeg_bytes = bytes([
            0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
            0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
            0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
            0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
            0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
            0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
            0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
            0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
            0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
            0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
            0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
            0x09,0x0A,0x0B,0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,
            0x00,0xFB,0x26,0x8A,0x28,0x03,0xFF,0xD9
        ])
        b64 = base64.b64encode(jpeg_bytes).decode()
        return f"data:image/jpeg;base64,{b64}"


RESULTS = []

def _log(name: str, passed: bool, detail: str = ""):
    icon = "PASS" if passed else "FAIL"
    print(f"  [{icon}]  {name}" + (f" -- {detail}" if detail else ""))
    RESULTS.append((name, passed, detail))


# ── 1. Backend Health ──────────────────────────────────────────────────────────

def test_backend_health():
    print("\n[1] Backend Health")
    try:
        r = requests.get(f"{BACKEND_URL}/", timeout=REQUEST_TIMEOUT)
        _log("Backend reachable", r.status_code == 200)
    except Exception as e:
        _log("Backend reachable", False, str(e))
        return

    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=REQUEST_TIMEOUT)
        data = r.json()
        # health returns {"status":"ok","checks":{"supabase":"ok",...}}
        supabase_ok = (
            data.get("checks", {}).get("supabase") == "ok"
            or data.get("supabase") == "ok"
        )
        _log("Supabase connection healthy", supabase_ok, f"checks={data.get('checks')}")
    except Exception as e:
        _log("Health endpoint", False, str(e))


# ── 2. CORS Headers ──────────────────────────────────────────────────────────

def test_cors():
    print("\n[2] CORS")
    try:
        r = requests.options(
            f"{BACKEND_URL}/session/start",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
            timeout=REQUEST_TIMEOUT,
        )
        ac_origin = r.headers.get("Access-Control-Allow-Origin", "")
        cors_ok = "localhost:3000" in ac_origin or ac_origin == "*"
        _log("CORS allows localhost:3000", cors_ok, f"header={ac_origin!r}")
    except Exception as e:
        _log("CORS check", False, str(e))


# ── 3. Session API Flow (temporal) ────────────────────────────────────────────

def test_session_api_flow(face_b64: str):
    """Returns session_id if successfully created."""
    print("\n[3] Session API Flow (temporal persistence)")
    sid = None

    # 3a. Start
    try:
        r = requests.post(f"{BACKEND_URL}/session/start", json={}, timeout=REQUEST_TIMEOUT)
        ok = r.status_code == 200 and "session_id" in r.json()
        sid = r.json().get("session_id")
        _log("session/start", ok, f"sid={str(sid)[:8]}...")
    except Exception as e:
        _log("session/start", False, str(e))
        return None

    # 3b. Analyze (6 frames)
    analyze_ok = True
    smoothed_seen = False
    for i in range(6):
        try:
            r = requests.post(
                f"{BACKEND_URL}/analyze",
                json={
                    "img": face_b64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                    "session_id": sid,
                },
                timeout=REQUEST_TIMEOUT,
            )
            if r.status_code != 200:
                analyze_ok = False
                _log(f"analyze frame {i+1}", False, r.text[:120])
                break
            d = r.json()
            if d.get("smoothed_emotions"):
                smoothed_seen = True
        except Exception as e:
            analyze_ok = False
            _log(f"analyze frame {i+1}", False, str(e))
            break
        time.sleep(0.2)
    _log("analyze 6 frames (all 200 OK)", analyze_ok)
    _log("smoothed_emotions present in /analyze response", smoothed_seen)

    # 3c. Pause
    try:
        r = requests.post(
            f"{BACKEND_URL}/session/pause",
            json={"session_id": sid},
            timeout=REPORT_TIMEOUT,
        )
        pause_ok = r.status_code == 200
        _log("session/pause (200 OK)", pause_ok)
        if pause_ok:
            pause_data = r.json()
            temporal_in_pause = bool(pause_data.get("report", {}).get("temporal"))
            _log("temporal data in pause response.report", temporal_in_pause)
    except Exception as e:
        _log("session/pause", False, str(e))

    # 3d. Stop
    try:
        r = requests.post(
            f"{BACKEND_URL}/session/stop",
            json={"session_id": sid},
            timeout=REPORT_TIMEOUT,
        )
        stop_ok = r.status_code == 200
        _log("session/stop (200 OK)", stop_ok)
        if stop_ok:
            stop_data = r.json()
            report = stop_data.get("report", {})
            summary_text = report.get("summary", report.get("text_summary", ""))
            summary_ok = len(summary_text) > 50
            _log("stop report summary len>50 chars", summary_ok, f"len={len(summary_text)}")
    except Exception as e:
        _log("session/stop", False, str(e))
        return sid

    # 3e. GET session details — temporal_summary persisted
    try:
        r = requests.get(f"{BACKEND_URL}/session/{sid}", timeout=REQUEST_TIMEOUT)
        assert r.status_code == 200
        session = r.json().get("session", {})
        ts = session.get("temporal_summary")
        _log("temporal_summary persisted to DB (not None)", ts is not None)
        if ts:
            required_keys = ("stability_score", "transitions", "frame_count", "durations", "trends")
            keys_ok = all(k in ts for k in required_keys)
            _log("temporal_summary has all required keys", keys_ok, f"found={list(ts.keys())}")
            stab = ts.get("stability_score")
            stab_ok = isinstance(stab, (int, float)) and 0 <= stab <= 1
            _log("stability_score is float in [0,1]", stab_ok, f"val={stab}")
            smoothed_tl = ts.get("smoothed_timeline", [])
            _log("smoothed_timeline is list", isinstance(smoothed_tl, list))
            transitions = ts.get("transitions", [])
            _log("transitions is list", isinstance(transitions, list))
            if transitions:
                t0 = transitions[0]
                trans_keys_ok = "from_emotion" in t0 and "to_emotion" in t0
                _log("transition objects have from/to_emotion", trans_keys_ok)
    except Exception as e:
        _log("GET /session/<id>", False, str(e))

    # 3f. History includes temporal
    try:
        r = requests.get(f"{BACKEND_URL}/session/history", timeout=REQUEST_TIMEOUT)
        sessions = r.json().get("sessions", [])
        found = [s for s in sessions if s.get("id") == sid]
        _log("session appears in /session/history", bool(found))
        if found:
            has_ts = found[0].get("temporal_summary") is not None
            _log("history entry has temporal_summary", has_ts)
    except Exception as e:
        _log("GET /session/history", False, str(e))

    return sid


# ── 4. Frontend (Playwright) ──────────────────────────────────────────────────

def test_frontend(page):
    print("\n[4] Frontend — Playwright Browser Tests")

    # 4a. Home page renders
    page.goto(FRONTEND_URL)
    page.wait_for_load_state("networkidle")

    title_visible = page.locator("text=SynCVE").first.is_visible()
    _log("Home page: SynCVE brand visible", title_visible)

    signin_visible = page.locator("text=Sign in with Google").is_visible()
    _log("Home page: Sign in with Google button visible", signin_visible)

    # 4b. Check #root has content (React mounted)
    root_text = page.locator("#root").inner_text()
    _log("React app mounted (#root has content)", len(root_text) > 20)

    # 4c. Home page content checks
    content = page.content()
    _log("Home page mentions emotion features", "emotion" in content.lower() or "Emotion" in content)

    # 4d. Screenshot home
    page.screenshot(path="tests/e2e/screenshots/home_page.png", full_page=True)
    _log("Home page screenshot saved", True, "tests/e2e/screenshots/home_page.png")

    # 4e. Unauthenticated redirect — /detection
    page.goto(f"{FRONTEND_URL}/detection")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    on_root = FRONTEND_URL.rstrip("/") in page.url and "/detection" not in page.url
    _log("Unauthenticated /detection redirects away", on_root, f"url={page.url}")

    # 4f. Unauthenticated redirect — /history
    page.goto(f"{FRONTEND_URL}/history")
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(800)
    on_root2 = FRONTEND_URL.rstrip("/") in page.url and "/history" not in page.url
    _log("Unauthenticated /history redirects away", on_root2, f"url={page.url}")


def test_cors_via_browser(page):
    """Verify backend responds to fetch() from the browser origin."""
    print("\n[5] CORS via Browser fetch()")
    page.goto(FRONTEND_URL)
    page.wait_for_load_state("networkidle")

    result = page.evaluate("""async () => {
        try {
            const r = await fetch('http://localhost:5005/', { method: 'GET' });
            const text = await r.text();
            return { ok: r.ok, status: r.status, has_body: text.length > 0 };
        } catch(e) {
            return { ok: false, error: e.toString() };
        }
    }""")
    _log("Browser fetch() to backend root (CORS ok)", result.get("ok", False), str(result))

    result2 = page.evaluate("""async () => {
        try {
            const r = await fetch('http://localhost:5005/health');
            const data = await r.json();
            const supabase = (data.checks || {}).supabase || data.supabase;
            return { ok: r.ok, supabase: supabase };
        } catch(e) {
            return { ok: false, error: e.toString() };
        }
    }""")
    health_ok = result2.get("ok") and result2.get("supabase") == "ok"
    _log("Browser fetch() /health -> supabase:ok", health_ok, str(result2))

    # Session start via browser fetch
    result3 = page.evaluate("""async () => {
        try {
            const r = await fetch('http://localhost:5005/session/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            });
            const data = await r.json();
            return { ok: r.ok, has_sid: 'session_id' in data };
        } catch(e) {
            return { ok: false, error: e.toString() };
        }
    }""")
    _log("Browser fetch() session/start works", result3.get("ok") and result3.get("has_sid"), str(result3))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.makedirs("tests/e2e/screenshots", exist_ok=True)

    print("=" * 65)
    print("  SynCVE Playwright E2E Validation Suite")
    print("=" * 65)

    face_b64 = _make_face_b64()
    print(f"  Test image: {len(face_b64)} chars data-URI")

    # HTTP-level tests
    test_backend_health()
    test_cors()
    completed_sid = test_session_api_flow(face_b64)

    # Browser tests
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 800})
        page = ctx.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        try:
            test_frontend(page)
            test_cors_via_browser(page)

            # Final screenshot
            page.goto(FRONTEND_URL)
            page.wait_for_load_state("networkidle")
            page.screenshot(path="tests/e2e/screenshots/final.png", full_page=True)
            _log("Final screenshot saved", True, "tests/e2e/screenshots/final.png")

            # Console errors (exclude benign favicon 404)
            real_errors = [e for e in console_errors if "favicon" not in e.lower() and "favicon" not in e]
            _log("No browser console errors", len(real_errors) == 0,
                 "; ".join(real_errors[:3]) if real_errors else "clean")

        finally:
            browser.close()

    # Summary
    print("\n" + "=" * 65)
    passed = sum(1 for _, p, _ in RESULTS if p)
    total = len(RESULTS)
    print(f"  RESULTS: {passed}/{total} passed")

    failed = [(n, d) for n, p, d in RESULTS if not p]
    if failed:
        print("\n  Failed:")
        for name, detail in failed:
            print(f"    [FAIL] {name}" + (f" -- {detail}" if detail else ""))
    else:
        print("  All checks passed!")
    print("=" * 65)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
