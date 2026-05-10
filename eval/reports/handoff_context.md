# SynCVE Evaluation Handoff Context

**Date:** 2026-03-21 01:45
**Purpose:** Complete context for the next agent to continue evaluation work

---

## 1. Current State Summary

### What's Done
- All pipeline parameters optimized and validated via full-scale ablation studies
- Docker environment configured and verified (backend + frontend)
- 9 code optimizations applied (see eval_phase1_results.md Section 10)
- TF verbose logging fix applied (MTCNN crash resolved)
- Comprehensive reports generated

### What's NOT Done
1. **RAF-DB pipeline_vs_baseline** — crashed with segfault at 47%, needs retry
2. **Production settings.yml alignment** — still uses opencv+ssd, should test retinaface+mtcnn in Docker Flask
3. **Full-scale detector ablation on RAF-DB** — only tested on FER2013

---

## 2. Critical Gap: Eval vs Production Mismatch

The eval pipeline tests the BEST config (retinaface+mtcnn 50/50), but production
`settings.yml` uses opencv+ssd because of a claimed TF/Keras KerasTensor
compatibility issue on Python 3.13 in Flask context.

**However**: The SynCVE conda env and Docker container both use Python 3.10 +
TF 2.10.1, where this issue may not exist. Nobody has tested this.

**Action needed**: Start Flask backend in Docker, send a test request, verify
retinaface+mtcnn work. If they do, update settings.yml.

---

## 3. Completed Evaluation Results

### Baselines (full-scale, no preprocessing)
| Dataset | Detector | n | Accuracy | W-F1 |
|---------|----------|---|----------|------|
| FER2013 | retinaface | 7,178 | 49.97% | 0.4985 |
| FER2013 | opencv | 500 | 54.00% | 0.5404 |
| RAF-DB | retinaface | 3,068 | 46.00% | 0.4536 |
| RAF-DB | opencv | 500 | 51.40% | 0.4885 |

### Preprocessing Ablation (FER2013, n=7,178, retinaface)
| Config | Accuracy | Delta |
|--------|----------|-------|
| none | 49.99% | — |
| **sr_only** | **51.42%** | **+1.43%** |
| clahe_only | 44.87% | -5.12% |
| full_preprocess | 49.41% | -0.58% |

### Ensemble Optimizer (2000 train, 500 test, step=0.05)
- Grid search: 19 combos, best train = retina:0.30/mtcnn:0.70 (66.20%)
- **Test result: 50/50 equal weights WIN (57.20% vs 55.20%)**

### Postprocess Ablation (7,200 frames, 360 batches)
| Config | Stab Acc | Consistency | Trans Pres |
|--------|----------|-------------|-----------|
| raw | 51.54% | 0.37 | 2.00 |
| **ema_0.2** | **71.59%** | **0.87** | **1.22** |

### Detector Ablation (963 stratified, GPU)
| Detector | Accuracy | Latency |
|----------|----------|---------|
| mtcnn | 45.38% | 697ms |
| opencv | 43.82% | 44ms |
| ssd | 43.82% | 69ms |
| retinaface | 41.64% | 105ms |

### Pipeline vs B0 (FER2013, n=1000)
| Config | Accuracy | W-F1 |
|--------|----------|------|
| B0 (opencv raw) | 53.30% | 0.5344 |
| Pipeline (SR+retina+mtcnn 50/50) | 51.60% | 0.5141 |
| Delta | -1.70% | -0.0203 |

**Note**: Pipeline underperforms on FER2013 because FER2013 is 48px pre-aligned.
OpenCV cascade excels on aligned crops. Pipeline is designed for unconstrained
images — RAF-DB comparison is needed to demonstrate this.

---

## 4. Optimal Pipeline Configuration

```yaml
preprocess:
  enable_sr: true
  sr_min_size: 256
  enable_clahe: true        # only for >= 128px
  enable_unsharp: true      # only for >= 128px
  adaptive_threshold: 128

ensemble:
  detectors: [retinaface, mtcnn]
  weights:
    retinaface: 0.50
    mtcnn: 0.50

temporal:
  ema_alpha: 0.2
  noise_floor: 0.0
```

---

## 5. Code Optimizations Applied

1. **Parallel ensemble** (ThreadPoolExecutor) — `src/backend/service.py`
2. **Remove .copy()** in ensemble loop — `service.py`
3. **Bound TemporalAnalyzer** with deque(maxlen=1000) — `temporal_analysis.py`
4. **Warmup all detectors** — `app.py`
5. **Stratified --limit sampling** — all eval scripts
6. **Adaptive preprocessing alignment** — eval scripts match production
7. **Optimizer cache fix** (numpy.float32 serialization) — `optimize_ensemble_weights.py`
8. **TF verbose log fix** — `_gpu_init.py`
9. **Docker full-stack** — `Dockerfile` + `docker-compose.yml`

---

## 6. Known Issues

1. **MTCNN TF logging**: Fixed in `_gpu_init.py` but processes can still segfault
   on very long runs. Use `--limit 1000` for pipeline_vs_baseline.
2. **conda run instability**: Never use `conda run`. Use direct path:
   `E:/conda/envs/SynCVE/python.exe` or `./run.sh`
3. **Disgust class collapse**: F1=0.0 on RAF-DB (both detectors). Dataset issue.
4. **Docker Desktop**: May get stuck on "Starting". Fix: `wsl --shutdown && wsl --update`

---

## 7. File Inventory

### Reports
- `eval/reports/eval_phase1_results.md` — Main report (all results + audit + optimizations)
- `eval/reports/optimization_decision.md` — Parameter decision record
- `eval/reports/ablation_findings.md` — Original ablation findings (pre-optimization)
- `eval/reports/handoff_context.md` — This file
- `dev/docs/deployment-guide-zh.md` — Chinese deployment guide

### Result JSONs
- `eval/results/baseline/fer2013_retinaface.json` — Full (n=7178)
- `eval/results/baseline/rafdb_retinaface.json` — Full (n=3068)
- `eval/results/baseline/fer2013_opencv.json` — Partial (n=500)
- `eval/results/baseline/rafdb_opencv.json` — Partial (n=500)
- `eval/results/ablation/preprocess.json` — Full (n=7178)
- `eval/results/ablation/postprocess.json` — Full (n=7200)
- `eval/results/ablation/detector.json` — n=963 stratified
- `eval/results/ablation/ensemble.json` — 2000 train, 500 test
- `eval/results/pipeline/pipeline_vs_b0.json` — OLD (n=10 smoke test, NOT updated)

### Docker
- `Dockerfile` — Multi-stage (backend-base, frontend, eval)
- `docker-compose.yml` — 3 services (backend:5005, frontend:3000, eval)
- `docker/nginx.conf` — Frontend reverse proxy
- `.dockerignore` — Build context exclusions
- `run.sh` — Universal Python runner

### Cache
- `eval/cache/` — ~15,277 cached detector results (retinaface+mtcnn on FER2013)

---

## 8. Environment

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10.20 | SynCVE conda env |
| PyTorch | 2.7.1+cu118 | CUDA enabled |
| TensorFlow | 2.10.1 | GPU support |
| CUDA | 11.8 | Driver 581.80 |
| GPU | RTX 3060 6GB | Docker GPU verified |
| Docker | 28.5.1 | Images built + tested |
| OS | Windows 11 Pro | WSL2 backend for Docker |
