"""
Real GPU performance and stress tests.
"""
import pytest
import requests
import time
import statistics


class TestGPUPerformanceReal:
    """Performance benchmarks with real GPU inference."""

    def test_warmup_latency(self, backend_url, test_face_image_base64):
        """First request (cold start) latency measurement."""
        start = time.time()
        resp = requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            },
            timeout=60,
        )
        elapsed = time.time() - start
        assert resp.status_code == 200
        print(f"\nWarmup latency: {elapsed:.2f}s")
        # First request can be slow (model loading), but should complete
        assert elapsed < 30.0, f"Warmup took {elapsed:.2f}s, too slow"

    def test_sustained_inference_benchmark(
        self, backend_url, test_face_image_base64
    ):
        """Benchmark 10 consecutive inferences."""
        # Warmup first
        requests.post(
            f"{backend_url}/analyze",
            json={
                "img": test_face_image_base64,
                "actions": ["emotion"],
                "anti_spoofing": False,
            },
            timeout=30,
        )

        latencies = []
        for i in range(10):
            start = time.time()
            resp = requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": test_face_image_base64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                },
                timeout=30,
            )
            elapsed = time.time() - start
            assert resp.status_code == 200
            latencies.append(elapsed)

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        print(f"\n--- Inference Benchmark (10 requests) ---")
        print(f"  Average: {avg:.3f}s")
        print(f"  Min:     {min(latencies):.3f}s")
        print(f"  Max:     {max(latencies):.3f}s")
        print(f"  P95:     {p95:.3f}s")

        assert avg < 5.0, f"Average latency {avg:.2f}s is too high"

    def test_memory_stability(self, backend_url, test_face_image_base64):
        """Run 20 requests and verify no OOM or degradation."""
        for i in range(20):
            resp = requests.post(
                f"{backend_url}/analyze",
                json={
                    "img": test_face_image_base64,
                    "actions": ["emotion"],
                    "anti_spoofing": False,
                },
                timeout=30,
            )
            assert (
                resp.status_code == 200
            ), f"Request {i + 1}/20 failed: {resp.text[:200]}"
