# RF-DETR Football Detection - Production Dockerfile
# Optimized for AWS L40s (Ada Lovelace) / G6e instances

# ==========================================
# Stage 1: Builder
# ==========================================
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime system dependencies
# ffmpeg is required for video processing
# libsm6, libxext6 are often needed for opencv even in headless mode (minimal X11 stubs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    ffmpeg \
    curl \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Create necessary directories with correct permissions
RUN mkdir -p models data/inputs data/outputs clips logs && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser run_mjpeg_stream.py .
COPY --chown=appuser:appuser configs/ ./configs/

# Switch to non-root user
USER appuser

# Expose MJPEG port
EXPOSE 8554

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8554/stream.mjpg || exit 1

# Run the pipeline
CMD ["python", "run_mjpeg_stream.py"]
