# RF-DETR Football Detection - Production Dockerfile
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    ffmpeg \
    curl \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY run_mjpeg_stream.py .
COPY configs/ ./configs/

# Create required directories
RUN mkdir -p models data/inputs data/outputs clips logs

# Expose MJPEG port
EXPOSE 8554

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8554/stream.mjpg || exit 1

# Run the pipeline
CMD ["python3.10", "run_mjpeg_stream.py"]
