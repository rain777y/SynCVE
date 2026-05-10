"""
Experiment logging — records run metadata for reproducibility.
Appends to eval/results/experiment_log.jsonl (one JSON object per line).
"""
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "results" / "experiment_log.jsonl"


def log_experiment(experiment_name: str, config: dict, results_summary: dict, duration_sec: float):
    """Append experiment metadata to the log file."""
    entry = {
        "experiment": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "duration_sec": round(duration_sec, 1),
        "config": config,
        "results_summary": results_summary,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def load_experiment_log() -> list:
    """Load all experiment log entries."""
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
