# =============================================================================
# SynCVE Backend — GPU-accelerated Emotion Recognition Service
# Base: NVIDIA CUDA 11.8 + cuDNN 8 on Ubuntu 22.04
# Stack: Python 3.10, TensorFlow 2.10.1, PyTorch (cu118), DeepFace, Flask
# =============================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS backend-base

# Prevent interactive prompts during apt
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for OpenCV, DeepFace, and general Python builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    pip install --no-cache-dir --upgrade pip setuptools && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Layer 1: PyTorch CUDA (largest, changes least often) ---
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

# --- Layer 2: Pip dependencies (changes on requirements.txt update) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Layer 3: Application code (changes most often) ---
COPY settings.yml .
COPY src/backend/ src/backend/

# GPU environment
ENV CUDA_VISIBLE_DEVICES=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    PYTHONUNBUFFERED=1

EXPOSE 5005

# Health check: hit the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:5005/health || exit 1

CMD ["python", "-m", "src.backend.app"]


# =============================================================================
# Frontend — React production build served by nginx
# =============================================================================
FROM node:18-alpine AS frontend-build

WORKDIR /app/src/frontend
COPY src/frontend/package.json src/frontend/package-lock.json* ./
RUN npm install --legacy-peer-deps

COPY src/frontend/ .

# Build-time env vars (baked into static JS bundle)
ARG REACT_APP_SERVICE_ENDPOINT=http://localhost:5005
ARG REACT_APP_SUPABASE_URL=
ARG REACT_APP_SUPABASE_ANON_KEY=
ENV REACT_APP_SERVICE_ENDPOINT=$REACT_APP_SERVICE_ENDPOINT \
    REACT_APP_SUPABASE_URL=$REACT_APP_SUPABASE_URL \
    REACT_APP_SUPABASE_ANON_KEY=$REACT_APP_SUPABASE_ANON_KEY

RUN npx react-scripts build

FROM nginx:alpine AS frontend
COPY --from=frontend-build /app/src/frontend/build /usr/share/nginx/html
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 3000


# =============================================================================
# Eval runner — same Python stack as backend, for running benchmarks
# =============================================================================
FROM backend-base AS eval

COPY eval/ eval/
COPY settings.yml .

# Eval scripts need sklearn, matplotlib, tqdm (already in requirements.txt)
CMD ["python", "-m", "eval.run_all"]
