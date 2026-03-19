# **SynCVE Pro Max** — Synchronized Computer Vision Emotion Suite

SynCVE is a research-ready, GPU-first emotion-recognition platform that merges DeepFace vision intelligence, anti-spoofing safeguards, and a modern React experience. This “Pro Max” reference summarizes architecture, day-to-day operations, diagnostics, and tuning advice so you can run SynCVE with confidence.

## Table of Contents

1. [Executive Overview](#executive-overview)
2. [Capabilities at a Glance](#capabilities-at-a-glance)
3. [Architecture & Core Components](#architecture--core-components)
4. [Prerequisites & Environment](#prerequisites--environment)
5. [Installation and First Run](#installation-and-first-run)
6. [Runtime Operations](#runtime-operations)
7. [Configuration Reference](#configuration-reference)
8. [GPU & Performance Tuning](#gpu--performance-tuning)
9. [Development Tooling & Scripts](#development-tooling--scripts)
10. [Diagnostics & Troubleshooting](#diagnostics--troubleshooting)
11. [Documentation Index](#documentation-index)
12. [Roadmap and Maintenance Notes](#roadmap-and-maintenance-notes)
13. [Licensing & Attribution](#licensing--attribution)

## Executive Overview

SynCVE fuses DeepFace’s recognition stack with Docker-friendly Flask APIs and a polished React UI. It is optimized for continuous webcam feeds, anti-spoofing with FasNet, and GPU acceleration (CUDA/TensorFlow/PyTorch) across Windows research PCs. The stack emphasizes:

- **Real-time inference**: Flask API (port 5005) keeps latency under 2 seconds after warm-up.
- **Security posture**: Anti-spoofing guard rails keep user-facing demos resilient.
- **Observability**: Structured logging, GPU monitoring toggles, and developer docs in `dev/docs/`.
- **Graceful lifecycle**: Start/stop scripts coordinate environment activation, model caching, and port cleanup.

## Capabilities at a Glance

- **Emotion detection** with DeepFace models (Facenet, VGG-Face, ArcFace) and configurable detector backends (RetinaFace, OpenCV, MTCNN).
- **Anti-spoofing integration** via FasNet (TensorFlow/PyTorch hybrid).
- **GPU-first performance**: layered TensorFlow/PyTorch handling with memory-growth and cache limiting utilities.
- **Dark-mode React UI** with responsive design (React 18 + Supabase SDK for future expansion).
- **Multi-service orchestration**: independent backend and frontend terminals managed through `.bat` wrappers.
- **Diagnostics**: GPU monitoring, netstat-based port checks, and script-driven troubleshooting (see `dev/tests/`).

## Architecture & Core Components

### Backend (`src/backend/`)
1. **`app.py`**: Bootstraps Flask app, loads `.env`, configures GPU before importing TensorFlow, and exposes the API on `0.0.0.0:5005`.
2. **`routes.py`**: Defines `/`, `/represent`, `/verify`, `/analyze` endpoints. Handles multipart/form-data, JSON payloads, and per-request validation with helpful error responses.
3. **`service.py`**: Wraps DeepFace operations with GPU cleanup helpers (`gpu_utils.clear_gpu_memory()`, `limit_model_cache()`).
4. **`gpu_utils.py`**: Manages TensorFlow GPU sessions, manual garbage collection, memory stats, and cache pruning for repeated loads.

#### Emotion Reporting Pipeline (Two-stage)
- **Purpose**: Turn stored keyframes + raw emotion scores into a Markdown report using a fast prompt stage (Gemini Flash Lite) and a vision stage (Gemini Pro Image Preview).
- **Endpoint**: `POST /session/report/emotion` with `{ "session_id": "...", "raw_vision_data": [...], "max_keyframes": 4 }`. If `raw_vision_data` is omitted, it auto-pulls recent rows from `emotion_logs` and frames from the `syn_cve_assets` bucket.
- **Config**: Tune `EMOTION_NOISE_FLOOR` (default `0.10`) to filter weak emotions and `EMOTION_REPORT_KEYFRAME_LIMIT` (default `4`) to cap downloads from storage.
- **Flow**: Aggregate scores -> Flash Lite builds a context prompt (no images) -> Pro Vision consumes prompt + keyframes -> returns `report_markdown` plus metrics/context for UI display or persistence.

#### Data-to-Visual Pipeline (Image v3.0)
- **Purpose**: Produce a single futuristic “Emotion Analysis Dashboard” image.
- **Endpoint**: `POST /session/report/visual` with `{ "session_id": "...", "raw_vision_data": [...], "aspect_ratio": "16:9", "style_preset": "futuristic" }`. If `raw_vision_data` is omitted, it pulls `emotion_logs`.
- **Config**: `EMOTION_VISUAL_ASPECT_RATIO` (default `16:9`), `EMOTION_VISUAL_STYLE_PRESET` (default `futuristic`), shares `EMOTION_NOISE_FLOOR`.
- **Flow**: Aggregate stats -> Flash-Lite (Art Director) writes the image prompt -> Pro Image Preview renders -> image uploaded to `syn_cve_assets/{session}/reports/...` and returns `public_url` + prompt + stats summary.
- **/session/pause**: now invokes this v3.0 pipeline and persists `visual_report_v3` into `sessions.metadata` plus `summary` as a JSON string for UI consumption.

### Frontend (`src/frontend/`)

- Built with React 18 + React Router DOM 7.
- Designed for modern aesthetics (glassmorphism, dark theme).
- Talkies to backend via `REACT_APP_SERVICE_ENDPOINT`, supports detection interval tuning through env variables.

### Supporting Utilities

- **Configuration storage**: Use `src/backend/backend.env` (falling back to `config/backend/`, `backend/.env`, or root `.env`).
- **Logs**: `dev/log/` contains runtime evidence and diagnostics for troubleshooting GPU, CUDA, or dependency issues.
- **Scripts**: Batch/PowerShell helpers in `dev/tests/` and root `.bat` wrappers orchestrate conda activation, dependency checks, and service startup/shutdown.

## Prerequisites & Environment

### Hardware

- NVIDIA GPU (recommended 6GB VRAM or more). Linux builds require CUDA 11.8 and cuDNN 8.6; Windows builds have corresponding driver expectations (>=522.06).
- CPU: 12-thread Intel i7-12700H or similar for balanced throughput.
- RAM: Minimum 16GB; more recommended for caching multiple DeepFace models simultaneously.

### Software

- **Windows 10/11** (PowerShell available) or Linux WSL for CLI parity.
- **Conda** (Anaconda or Miniconda) to isolate Python 3.10 dependencies.
- **Node.js 16+** for the React frontend.
- **CUDA 11.8** (for GPU mode). Validate via `nvidia-smi`.
- **TensorFlow 2.15.0 and PyTorch 2.1.0+cu118** are locked in `requirements.txt`.
- Optional: `python-dotenv`, `torchvision`, `gdown`, `requests`, etc. (auto-installed via `pip install -r requirements.txt`).

## Installation and First Run

```bash
# Clone repo
git clone https://example.com/SynCVE.git
cd SynCVE

# Create and activate conda environment
conda create -n SynCVE python=3.10 -y
conda activate SynCVE

# Install Python backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd src/frontend
npm install
cd ../..
```

### Environment Variables

- Backend config file: `src/backend/backend.env` (primary) with fallback paths described in `app.py`.
- Frontend expects `src/frontend/.env` or shared `.env` entries for:

  ```
  REACT_APP_SERVICE_ENDPOINT=http://localhost:5005
  REACT_APP_DETECTOR_BACKEND=retinaface
  REACT_APP_ANTI_SPOOFING=1
  REACT_APP_DETECTION_INTERVAL=2000
  ```

### CUDA Health Check

```bash
conda activate SynCVE
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

## Runtime Operations

- **First time setup**: `scripts\setup.bat` — one-click install (Conda env, pip, npm, GPU check, env files).
- **Start backend**: `scripts\scripts\start_backend.bat` — activates Conda, checks GPU/port, starts Flask on port 5005.
- **Start frontend**: `scripts\scripts\start_frontend.bat` — checks Node.js, auto-installs deps, starts React on port 3000.
- **Stop services**: `scripts\stop_service.bat` — graceful shutdown with port cleanup.
- **Run tests**: `scripts\run_tests.bat` — unit + integration + regression test suite.
- **Health check**: `python scripts\health_check.py` — full dependency and environment verification.

### Port Coordination

- Backend listens on port `5005` (configured via `BACKEND_PORT`). `scripts\stop_service.bat` inspects `netstat` to confirm listeners and releases the port after termination.
- Use `netstat -ano | findstr ":5005"` to inspect port ownership manually.

## Configuration Reference

| Variable | Purpose | Default |
| --- | --- | --- |
| `BACKEND_PORT` | Flask listen port | `5005` |
| `DETECTOR_BACKEND` | Face detector choice (`opencv`, `mtcnn`, `retinaface`) | `opencv` |
| `ANTI_SPOOFING` | Enable FasNet anti-spoofing | `1` |
| `TF_FORCE_GPU_ALLOW_GROWTH` | Avoid GPU OOM by growing memory | `true` |
| `TF_GPU_MEMORY_FRACTION` | Limit GPU usage (0.0-1.0) | `0.8` |
| `GPU_MONITORING` | Log GPU stats for each request | `true` |
| `LOG_GPU_MEMORY` | Dump GPU memory usage | `true` |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow log level | `2` |

Front-end environment sits in `src/frontend/.env`. Keep `REACT_APP_SERVICE_ENDPOINT` aligned with backend host/port.

## GPU & Performance Tuning

- Ensure CUDA toolkit, cuDNN, and NVIDIA drivers are matched (11.8 / 8.6 / 522.06+).
- Set `CUDA_VISIBLE_DEVICES=0` to target GPU 0; fallback to CPU by setting `-1`.
- The backend uses `tf.config.experimental.set_memory_growth` and `gpu_utils.limit_model_cache` to prevent repeated OOM errors.
- **Warmup**: First API call may take 5–15 seconds while models load. Subsequent calls settle to 1–2 seconds.
- **Anti-spoofing**: Adds ~500ms per frame. Disable via `ANTI_SPOOFING=0` if latency is critical.
- **Parallel CPU threads**: `OMP_NUM_THREADS` defaults to `12`.

## Development Tooling & Scripts

- All scripts live under `scripts/`:
  - `setup.bat`: one-click first-time setup.
  - `start_backend.bat` / `start_frontend.bat`: service launchers.
  - `stop_service.bat`: graceful shutdown with port cleanup.
  - `run_tests.bat`: unit + integration + regression tests.
  - `health_check.py`: full environment verification.

## Diagnostics & Troubleshooting

| Symptom | Diagnostic Step | Resolution |
| --- | --- | --- |
| Backend fails to start | `python src/backend/app.py` manually | Check `requirements.txt`, reinstall via `pip install -r requirements.txt`, confirm `conda activate SynCVE`. |
| GPU not detected | `nvidia-smi`, `python -c "import torch;print(torch.cuda.is_available())"` | Install/update CUDA/cuDNN, ensure driver >=522.06, set `CUDA_VISIBLE_DEVICES`. |
| Port 5005 busy | `netstat -ano | findstr ":5005"` | Run `dev/tests/scripts\stop_service.bat` or `taskkill /PID <pid> /F`. |
| Frontend cannot reach backend | Confirm `REACT_APP_SERVICE_ENDPOINT` matches backend scheme/port. Use browser DevTools network panel. |
| TensorFlow OOM | Reduce `TF_GPU_MEMORY_FRACTION` (e.g., 0.7). Restart app to reset GPU memory. |
| Anti-spoofing errors | Ensure FasNet dependencies in conda env; disable temporarily with `ANTI_SPOOFING=0`. |

### Logs and Monitoring

- `dev/log/session-summary-2025-11-01.md` documents a sample run. Add more session summaries when you run experiments.
- GPU monitoring toggles (`GPU_MONITORING`, `LOG_GPU_MEMORY`) echo stats per request for audits.

## Documentation Index

- `dev/docs/QUICK_START_GUIDE.md`: Step-by-step bootstrap and startup.
- `dev/docs/frontend-redesign-summary.md`: UI story and decisions.
- `dev/docs/loading-optimization-implementation-summary.md`: Performance tuning insights.
- `dev/docs/backend_gpu_troubleshooting.md`: CUDA/PyTorch-specific deep dive.
- Additional docs under `dev/docs/` cover Google Auth, anti-spoofing fixes, and migration summaries.

## Roadmap and Maintenance Notes

- **High priority**: warmup reduction, adaptive detection intervals, user authentication, emotion analytics dashboards.
- **Medium priority**: multi-face support, export functionality, admin monitoring UI.
- **Low priority**: custom emotion models, multi-language support, responsive refinements.
- **Docs**: expand API references, add architecture diagrams, finalize contribution workflow.

## Changelog

### 2025-12-08

**Bug Fixes:**
- Fixed Supabase RLS policies for `vision_samples`, `session_events`, `session_aggregates`, and `reports` tables
- Fixed `style_preset` validation error in Gemini ImageConfig (moved to prompt text instead)
- Fixed React state closure issue in frontend detection loop (`sessionIdRef` pattern)
- Fixed response key mismatch: frontend now checks both `image_url` and `report_url`

**Improvements:**
- Added `test_full_pipeline.py` for end-to-end testing of emotion analysis and visual report generation
- Improved error handling and logging throughout the session lifecycle

## Licensing & Attribution

This educational project uses open-source frameworks: DeepFace, TensorFlow, PyTorch, React. Follow each dependency’s license when deploying commercially.

---

Need to stop the service manually? Run `dev/tests/scripts\stop_service.bat` afterward to reclaim port 5005 and check status.
