# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── OS-level libraries required by your imports ───────────────────────────────
# mediapipe needs libGL, libGlib
# OpenCV headless needs libSM, libXext
# facenet-pytorch needs cmake for some builds
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libgomp1 \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python packages ───────────────────────────────────────────────────
# Copy requirements first so Docker can CACHE this layer.
# If only your .py files change, Docker skips re-installing packages (saves 10+ min).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy your application code ────────────────────────────────────────────────
COPY main.py .
COPY proctoring_system.py .
COPY engine.py .

# ── Copy data files ───────────────────────────────────────────────────────────
COPY face_registry.json .

# ── Copy YOLO model weights ───────────────────────────────────────────────────
# Only include these lines if you removed *.pth from .gitignore
COPY yolov8n.pt .
COPY yolov8n-face.pt .
COPY face_model_full.pth .

# ── Copy static frontend ───────────────────────────────────────────────────────
COPY static/ ./static/

# ── Pre-create directories your app expects ───────────────────────────────────
# proctoring_system.py does os.makedirs for these, but explicit is safer
RUN mkdir -p uploads proctoring_logs

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start command ─────────────────────────────────────────────────────────────
# --workers 1 is CRITICAL: your sessions dict is in-memory.
# Multiple workers = each has a separate sessions dict = sessions not found = 404 errors.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
