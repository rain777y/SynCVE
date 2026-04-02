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

# --- Layer 2: Pip dependencies (split to avoid protobuf conflict) ---
# streamlit requires protobuf<5 but google-genai requires protobuf>=5
# Solution: install core deps inline, skip streamlit (only needed for eval dashboard)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    "numpy>=1.22.0,<2.0" \
    "pandas>=2.0,<3.0" \
    "requests>=2.28,<3.0" \
    "python-dotenv>=1.0,<2.0" \
    "pydantic>=2.0,<3.0" \
    "PyYAML>=6.0,<7.0" \
    "setuptools<81" \
    "tensorflow==2.10.1" \
    "opencv-python>=4.8,<5.0" \
    "Pillow>=10.0,<12.0" \
    "deepface>=0.0.99,<0.1.0" \
    "mtcnn==0.1.1" \
    "retina-face>=0.0.14,<0.0.18" \
    "Flask>=3.0,<4.0" \
    "flask-cors>=4.0,<5.0" \
    "flask-limiter>=3.0,<4.0" \
    "gunicorn>=21.2,<23.0" \
    "google-genai>=1.65,<2.0" \
    "supabase>=2.5,<3.0" \
    "scikit-learn>=1.3,<2.0" \
    "matplotlib>=3.7,<4.0" \
    "tqdm>=4.65,<5.0"

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
RUN npm install --legacy-peer-deps && \
    npm install ajv@8 --legacy-peer-deps

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
