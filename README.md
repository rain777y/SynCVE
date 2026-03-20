<p align="center">
  <img src="assets/banner.svg" alt="SynCVE — Synchronized Computer Vision Emotion" width="100%" />
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-blue?style=for-the-badge&logo=rocket&logoColor=white" alt="Quick Start"/></a>&nbsp;
  <a href="#api-reference"><img src="https://img.shields.io/badge/API_Docs-purple?style=for-the-badge&logo=fastapi&logoColor=white" alt="API Docs"/></a>&nbsp;
  <a href="#docker-deployment"><img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/></a>&nbsp;
  <a href="#evaluation--benchmarking"><img src="https://img.shields.io/badge/Benchmarks-green?style=for-the-badge&logo=speedtest&logoColor=white" alt="Benchmarks"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-3572A5?style=flat-square&logo=python&logoColor=white" alt="Python 3.10"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.10.1-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/PyTorch-2.1+cu118-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/CUDA-11.8-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react&logoColor=black" alt="React"/>
  <img src="https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/Docker-24+-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/Supabase-BaaS-3ecf8e?style=flat-square&logo=supabase&logoColor=white" alt="Supabase"/>
  <img src="https://img.shields.io/badge/License-Academic-6366f1?style=flat-square" alt="License"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Processing Pipeline](#processing-pipeline)
- [Tech Stack](#tech-stack)
- [System Architecture](#system-architecture)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [GPU & Performance Tuning](#gpu--performance-tuning)
- [Testing](#testing)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**SynCVE** (*Synchronized Computer Vision Emotion*) is a research-grade platform for **real-time facial emotion recognition** built with a GPU-first architecture. It fuses state-of-the-art computer vision models with modern web technologies to deliver continuous emotion tracking from live webcam feeds.

The system implements a complete emotion analysis pipeline: from frame capture and face detection through ensemble classification, temporal smoothing, and AI-powered report generation — all within a production-ready containerized deployment.

### Research Contributions

| Contribution | Description |
|:---|:---|
| **Ensemble Detector Fusion** | Weighted combination of OpenCV + SSD face detectors with configurable fallback chain (RetinaFace, MTCNN) for robust face detection across lighting conditions |
| **Anti-Spoofing Integration** | FasNet-based liveness detection using TensorFlow/PyTorch hybrid inference to reject photos, screens, and printouts in real-time |
| **Temporal Emotion Smoothing** | EMA-based smoothing (alpha=0.2) with transition detection and volatility tracking, parameters optimized through systematic ablation studies |
| **Two-Stage AI Reporting** | Gemini Flash for structured text analysis + Gemini Pro for visual dashboard generation with automatic keyframe selection |

<img src="assets/separator.svg" width="100%" />

## Key Features

<p align="center">
  <img src="assets/features.svg" alt="SynCVE Key Features" width="100%" />
</p>

<img src="assets/separator.svg" width="100%" />

## Processing Pipeline

<p align="center">
  <img src="assets/pipeline.svg" alt="Real-Time Emotion Recognition Pipeline" width="100%" />
</p>

The pipeline processes webcam frames at configurable intervals (default 2s), running each frame through face detection, anti-spoofing verification, 7-class emotion classification, temporal smoothing, and optional AI report generation. End-to-end latency is **~1–2 seconds** after model warmup.

<img src="assets/separator.svg" width="100%" />

## Tech Stack

<p align="center">
  <img src="assets/tech-stack.svg" alt="Technology Stack" width="100%" />
</p>

<details>
<summary><b>Full Dependency Breakdown</b></summary>

| Layer | Technology | Version | Purpose |
|:---|:---|:---|:---|
| **ML Framework** | TensorFlow | 2.10.1 | DeepFace emotion models, FasNet anti-spoofing |
| **ML Framework** | PyTorch | 2.1+cu118 | RetinaFace detector, auxiliary models |
| **Face Analysis** | DeepFace | 0.0.99+ | Emotion classification, face embedding, verification |
| **Face Detectors** | OpenCV, SSD, RetinaFace, MTCNN | Various | Ensemble face detection with weighted fusion |
| **GPU Runtime** | NVIDIA CUDA + cuDNN | 11.8 / 8.6 | GPU acceleration for all inference operations |
| **Web Backend** | Flask + Gunicorn | 3.x / 21.2 | REST API server with rate limiting and CORS |
| **Web Frontend** | React + Recharts | 18.3 / 3.8 | SPA with real-time emotion visualizations |
| **Reverse Proxy** | nginx | Alpine | SPA routing, API proxy, static asset serving |
| **AI Reports** | Google Gemini | 2.5 Flash | Text analysis and visual dashboard generation |
| **Cloud Storage** | Supabase | 2.5+ | PostgreSQL, object storage, auth SDK |
| **Validation** | Pydantic | 2.x | Request/response schema validation |
| **Containerization** | Docker Compose | 24+ | Multi-service orchestration with GPU passthrough |

</details>

<img src="assets/separator.svg" width="100%" />

## System Architecture

<p align="center">
  <img src="assets/architecture.svg" alt="System Architecture" width="100%" />
</p>

### Backend — `src/backend/` &middot; Flask + DeepFace &middot; Port 5005

| Module | Responsibility |
|:---|:---|
| `app.py` | Flask bootstrap, GPU config, model warmup, rate limiting, CORS |
| `routes.py` | REST endpoints — `/analyze`, `/session/*`, `/health`, `/represent`, `/verify` |
| `service.py` | DeepFace wrapper with GPU memory cleanup and model cache limiting |
| `emotion_analytics.py` | Score aggregation, noise filtering, emotion statistics |
| `temporal_analysis.py` | EMA smoothing, transition detection, volatility tracking |
| `gemini_client.py` | Two-stage AI report generation (text + visual) via Gemini API |
| `session_manager.py` | Session lifecycle — start, pause, stop, history, reports |
| `storage.py` | Supabase integration — PostgreSQL queries, bucket uploads |
| `gpu_utils.py` | TensorFlow/PyTorch memory management, garbage collection |

### Frontend — `src/frontend/` &middot; React 18 + nginx &middot; Port 3000

| Component | Description |
|:---|:---|
| Webcam Capture | Configurable interval frame capture with client-side crop |
| Emotion Dashboard | Real-time probability bars and dominant emotion display |
| Session Timeline | Recharts-based temporal emotion visualizations |
| Report Viewer | AI-generated text and visual report rendering |
| Session Manager | Start / pause / stop with history navigation |

<img src="assets/separator.svg" width="100%" />

## Quick Start

### Prerequisites

| Requirement | Minimum | Notes |
|:---|:---|:---|
| NVIDIA GPU | 6 GB VRAM | Driver ≥ 522.06 |
| CUDA Toolkit | 11.8 | With cuDNN 8.6 |
| Conda | Miniconda 3 | Or Anaconda |
| Node.js | 16+ | For frontend dev server |
| Docker Engine | 24+ | Only for containerized deployment |

### Option A — Local Development

```bash
# 1. Clone and enter project
git clone <repo-url> && cd SynCVE

# 2. Create conda environment
conda create -n SynCVE python=3.10 -y
conda activate SynCVE

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install frontend dependencies
cd src/frontend && npm install && cd ../..

# 5. Configure environment
cp .env.example .env
# Edit .env → add your GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY

# 6. Verify GPU availability
python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Start services:**

```bash
# Terminal 1 — Backend (GPU-accelerated Flask API)
conda activate SynCVE && python src/backend/app.py

# Terminal 2 — Frontend (React dev server)
cd src/frontend && npm start
```

> **Windows users** — one-click batch scripts are available:
>
> | Script | Purpose |
> |:---|:---|
> | `scripts\setup.bat` | First-time environment setup |
> | `scripts\start_backend.bat` | Launch Flask backend |
> | `scripts\start_frontend.bat` | Launch React dev server |
> | `scripts\stop_service.bat` | Graceful shutdown + port cleanup |

### Option B — Docker Compose

```bash
# 1. Configure secrets
cp .env.example .env   # fill in API keys

# 2. Launch all services
docker compose up -d

# 3. Verify health
curl http://localhost:5005/health

# 4. Open UI → http://localhost:3000
```

<img src="assets/separator.svg" width="100%" />

## Docker Deployment

<p align="center">
  <img src="assets/docker-layout.svg" alt="Docker Compose Orchestration" width="100%" />
</p>

### Multi-Stage Dockerfile

```
Dockerfile
├── backend-base    NVIDIA CUDA 11.8 + Python 3.10 + TF/PyTorch/DeepFace
├── frontend-build  Node 18 Alpine → React production bundle
├── frontend        nginx Alpine → serves SPA + reverse proxy to backend
└── eval            Extends backend-base → benchmark and ablation scripts
```

### File Layout Convention

| File | Location | Reason |
|:---|:---|:---|
| `Dockerfile` | Root | Docker standard — build context must include all source |
| `docker-compose.yml` | Root | Docker standard — orchestration entry point |
| `.dockerignore` | Root | Docker standard — controls build context |
| `nginx.conf` | `docker/` | Supporting config — copied into container at build time |

> Root-level Docker files follow the Docker convention where the build context (`.`) encompasses the entire project. The `docker/` subfolder holds auxiliary configs that are `COPY`'d into containers, keeping the root clean.

### Common Commands

```bash
docker compose up -d                       # Start backend + frontend
docker compose logs -f backend             # Tail backend logs
docker compose --profile eval run eval     # Run evaluation suite
docker compose down                        # Stop all services
docker compose build --no-cache backend    # Rebuild backend image
```

<img src="assets/separator.svg" width="100%" />

## API Reference

### Core Endpoints

| Endpoint | Method | Rate Limit | Description |
|:---|:---|:---|:---|
| `/health` | `GET` | 60/min | System health — DeepFace, Supabase, GPU status |
| `/config` | `GET` | 60/min | Non-secret runtime configuration |
| `/analyze` | `POST` | 30/min | Emotion detection with ensemble + anti-spoofing |
| `/represent` | `POST` | 60/min | Extract face embeddings (Facenet model) |
| `/verify` | `POST` | 60/min | Face verification — 1:1 cosine comparison |

### Session Endpoints

| Endpoint | Method | Rate Limit | Description |
|:---|:---|:---|:---|
| `/session/start` | `POST` | 60/min | Start emotion tracking session |
| `/session/pause` | `POST` | 60/min | Pause session + trigger visual report |
| `/session/stop` | `POST` | 60/min | Stop session + generate final report |
| `/session/history` | `GET` | 60/min | Recent sessions (filter by `user_id`, `limit`) |
| `/session/<id>` | `GET` | 60/min | Specific session details |
| `/session/report/emotion` | `POST` | 10/min | Two-stage Gemini text report |
| `/session/report/visual` | `POST` | 10/min | AI-generated visual dashboard image |

**Supported image input formats:** Base64 data-URI, multipart file upload (max 16 MB)

<img src="assets/separator.svg" width="100%" />

## Configuration

### `settings.yml` — Application Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 5005
  cors_origins: ["http://localhost:3000"]

deepface:
  detector_backend: "opencv"           # Primary face detector
  model_name: "Facenet"                # Face embedding model
  anti_spoofing: true                  # FasNet liveness detection
  confidence_threshold: 0.1
  ensemble:
    enabled: true
    detectors: ["opencv", "ssd"]       # Weighted fusion pair
    weights: { opencv: 0.60, ssd: 0.40 }

gpu:
  cuda_visible_devices: "0"            # "-1" for CPU-only mode
  tf_memory_fraction: 0.8
  tf_allow_growth: true

temporal:
  ema_alpha: 0.2                       # Smoothing factor (ablation-optimized)
  transition_threshold: 0.15           # Min delta for emotion changes
  volatility_window: 10                # Std dev window size

gemini:
  text_model: "gemini-2.5-flash"       # Structured text reports
  image_model: "gemini-2.5-flash-image"  # Visual dashboard generation
  request_timeout: 120
  max_retries: 3

report:
  mode: "fast"                         # "fast" = JSON only, "full" = JSON + AI image
  noise_floor: 0.0
  keyframe_limit: 4
  visual_style_preset: "futuristic"
```

### `.env` — Secrets (gitignored)

```ini
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
REACT_APP_SERVICE_ENDPOINT=http://localhost:5005
REACT_APP_SUPABASE_URL=https://your-project.supabase.co
REACT_APP_SUPABASE_ANON_KEY=your_supabase_anon_key
```

<img src="assets/separator.svg" width="100%" />

## GPU & Performance Tuning

| Parameter | Default | Impact | Recommendation |
|:---|:---|:---|:---|
| `cuda_visible_devices` | `"0"` | GPU selection | `"-1"` for CPU fallback |
| `tf_memory_fraction` | `0.8` | Max VRAM allocation | Lower if sharing GPU |
| `tf_allow_growth` | `true` | Dynamic VRAM allocation | Always `true` in dev |
| `OMP_NUM_THREADS` | Auto | CPU parallelism | Set to core count for non-GPU ops |
| `anti_spoofing` | `true` | +~500 ms/frame | Disable for latency-critical demos |
| `ensemble.enabled` | `true` | +~200 ms/frame | Single detector if VRAM limited |
| `ema_alpha` | `0.2` | Smoothing aggressiveness | Higher = more responsive, noisier |

### Performance Characteristics

| Phase | Latency | Notes |
|:---|:---|:---|
| Cold start (model load) | 5–15 s | One-time warmup at server boot |
| Inference (single frame) | ~1–2 s | With ensemble + anti-spoofing |
| Inference (single detector) | ~0.5–1 s | OpenCV only, no anti-spoofing |
| Gemini text report | ~3–8 s | Depends on session length |
| Gemini visual report | ~10–30 s | Image generation pipeline |

<img src="assets/separator.svg" width="100%" />

## Testing

```bash
# Full test suite (with coverage)
conda activate SynCVE
pytest tests/ -v --tb=short --cov=src/backend

# Specific test categories
pytest tests/unit/ -v              # Core logic
pytest tests/integration/ -v       # API endpoints
pytest tests/e2e/ -v               # Full pipeline
pytest tests/regression/ -v        # Bug regressions

# Health check (dependency verification)
python scripts/health_check.py
```

```
tests/
├── unit/           Core logic — analytics, temporal, service
├── integration/    API endpoint validation
├── e2e/            Full pipeline (capture → analyze → report)
├── regression/     Bug regression guards
└── artifacts/      Generated test images and videos
```

<img src="assets/separator.svg" width="100%" />

## Evaluation & Benchmarking

```bash
# Docker (recommended — isolated environment)
docker compose --profile eval run eval

# Local
conda activate SynCVE
python eval/benchmark.py --config eval/configs/default.yml
```

Evaluation outputs are written to `eval/results/` and include:

- **Accuracy metrics** — Per-emotion precision, recall, F1 scores
- **Latency benchmarks** — Inference time distributions across configurations
- **Ablation studies** — EMA alpha, detector weights, ensemble vs. single detector
- **Stratified sampling** — Balanced evaluation across emotion categories

<img src="assets/separator.svg" width="100%" />

## Project Structure

```
SynCVE/
├── src/
│   ├── backend/                 Flask API + DeepFace emotion detection
│   │   ├── app.py               Bootstrap, GPU config, model warmup
│   │   ├── routes.py            REST endpoint definitions
│   │   ├── service.py           DeepFace wrapper + GPU management
│   │   ├── emotion_analytics.py Score aggregation + noise filtering
│   │   ├── temporal_analysis.py EMA smoothing + transition detection
│   │   ├── gemini_client.py     Two-stage AI report generation
│   │   ├── session_manager.py   Session lifecycle management
│   │   ├── storage.py           Supabase PostgreSQL + bucket ops
│   │   └── gpu_utils.py         TF/PyTorch memory management
│   └── frontend/                React 18 SPA
│       ├── src/                 Components, hooks, styles
│       ├── public/              Static assets + favicon
│       └── package.json         Dependencies + scripts
│
├── eval/                        Evaluation & benchmarking
│   ├── configs/                 Benchmark configurations
│   ├── datasets/                Test datasets (gitignored)
│   ├── results/                 Evaluation outputs
│   └── reports/                 Analysis reports
│
├── tests/                       Test suites
│   ├── unit/                    Core logic tests
│   ├── integration/             API endpoint tests
│   ├── e2e/                     Full pipeline tests
│   └── regression/              Bug regression tests
│
├── scripts/                     Setup + utility scripts
│   ├── setup.bat                One-click environment setup
│   ├── health_check.py          Dependency verification
│   └── generate_test_*.py       Synthetic test data generators
│
├── docker/                      Container support configs
│   └── nginx.conf               SPA routing + API reverse proxy
│
├── assets/                      SVG logos, badges, diagrams
│
├── Dockerfile                   Multi-stage build (GPU + React + eval)
├── docker-compose.yml           Service orchestration
├── .dockerignore                Build context exclusions
├── settings.yml                 Application configuration
├── requirements.txt             Python dependencies
├── environment.yml              Conda environment spec
└── .env.example                 Environment variable template
```

<img src="assets/separator.svg" width="100%" />

## Troubleshooting

| Symptom | Diagnosis | Resolution |
|:---|:---|:---|
| Backend won't start | Missing dependencies | `pip install -r requirements.txt` |
| GPU not detected | Driver or CUDA mismatch | Install CUDA 11.8, driver ≥ 522.06; run `nvidia-smi` |
| TensorFlow OOM | VRAM pre-allocated | Lower `tf_memory_fraction` in `settings.yml`, restart |
| Port 5005/3000 busy | Zombie process | `scripts\stop_service.bat` or `netstat -ano \| findstr :5005` |
| Frontend can't reach backend | Endpoint mismatch | Verify `REACT_APP_SERVICE_ENDPOINT` in `.env` |
| Anti-spoofing errors | FasNet model missing | Let DeepFace auto-download, or disable in `settings.yml` |
| Docker build fails | Context too large | Verify `.dockerignore` excludes datasets and cache |
| Slow first request | Model warmup | Normal — 5–15s cold start, subsequent calls are ~1–2s |

<img src="assets/separator.svg" width="100%" />

## Documentation

Extended documentation is available in [`dev/docs/`](dev/docs/):

| Document | Topic |
|:---|:---|
| Quick Start Guide | Step-by-step environment bootstrap |
| Frontend Redesign | UI architecture decisions and component design |
| Loading Optimization | Performance tuning insights and benchmarks |
| GPU Troubleshooting | CUDA, cuDNN, and PyTorch deep dive |
| Google Auth Integration | Supabase authentication setup |
| Anti-Spoofing Fixes | FasNet integration and model compatibility |

<img src="assets/separator.svg" width="100%" />

## License

This is an academic research project developed as a Final Year Project (FYP). It uses open-source frameworks under their respective licenses:

- [DeepFace](https://github.com/serengil/deepface) — MIT License
- [TensorFlow](https://www.tensorflow.org/) — Apache 2.0
- [PyTorch](https://pytorch.org/) — BSD-3-Clause
- [React](https://react.dev/) — MIT License
- [Flask](https://flask.palletsprojects.com/) — BSD-3-Clause
- [Supabase](https://supabase.com/) — Apache 2.0

---

<p align="center">
  <img src="assets/logo.svg" alt="SynCVE" width="56" />
  <br/><br/>
  <sub><b>SynCVE</b> — Synchronized Computer Vision Emotion</sub>
  <br/>
  <sub>Built with research rigor and engineering craft.</sub>
</p>
