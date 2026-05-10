"""
Background job queue for slow endpoints.

Provides a process-wide ThreadPoolExecutor and a small in-memory job
registry so HTTP handlers can return immediately with a job_id while
the actual work runs in the background.

Public API:
    submit_job(fn, *args, **kwargs) -> job_id (str)
    get_job(job_id) -> dict | None
"""

import threading
import time
import uuid as _uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

# Module-level singleton executor (created lazily so import is cheap).
_EXECUTOR: Optional[ThreadPoolExecutor] = None
_EXECUTOR_LOCK = threading.Lock()

# Bounded job registry. OrderedDict preserves insertion order so we can
# evict the oldest entries (FIFO) once we hit the cap.
_MAX_JOBS = 200
_jobs: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_jobs_lock = threading.Lock()


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="syncve-job"
                )
    return _EXECUTOR


def _new_job_id() -> str:
    return _uuid.uuid4().hex[:12]


def _evict_if_needed_locked() -> None:
    """Drop oldest entries while we're over the cap. Must hold _jobs_lock."""
    while len(_jobs) > _MAX_JOBS:
        _jobs.popitem(last=False)


def _run_job(job_id: str, fn: Callable, args: tuple, kwargs: dict) -> None:
    """Worker thread entry point: execute fn and stash the result."""
    with _jobs_lock:
        entry = _jobs.get(job_id)
        if entry is not None:
            entry["status"] = "running"
            entry["started_at"] = time.time()

    started = time.time()
    try:
        result = fn(*args, **kwargs)
        with _jobs_lock:
            entry = _jobs.get(job_id)
            if entry is not None:
                entry["status"] = "done"
                entry["result"] = result
                entry["error"] = None
                entry["elapsed_ms"] = int((time.time() - started) * 1000)
    except Exception as exc:  # noqa: BLE001
        with _jobs_lock:
            entry = _jobs.get(job_id)
            if entry is not None:
                entry["status"] = "error"
                entry["result"] = None
                entry["error"] = f"{type(exc).__name__}: {exc}"
                entry["elapsed_ms"] = int((time.time() - started) * 1000)


def submit_job(fn: Callable, *args, **kwargs) -> str:
    """Schedule fn(*args, **kwargs) on the worker pool. Returns the job_id."""
    job_id = _new_job_id()
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "elapsed_ms": None,
            "submitted_at": time.time(),
        }
        _evict_if_needed_locked()
    _get_executor().submit(_run_job, job_id, fn, args, kwargs)
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return a snapshot of the job's state, or None if unknown."""
    with _jobs_lock:
        entry = _jobs.get(job_id)
        if entry is None:
            return None
        return {
            "status": entry["status"],
            "result": entry["result"],
            "error": entry["error"],
            "elapsed_ms": entry["elapsed_ms"],
        }
