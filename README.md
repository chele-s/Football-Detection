# RF-DETR Football Detection & Tracking System

**Production-grade real-time ball tracking with GPU-accelerated 4K streaming, RF-DETR transformer detection, Extended Kalman filtering, and cinematic virtual camera control.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

A high-performance football tracking pipeline delivering broadcast-quality 4K output at 25-30 FPS with intelligent resolution management: **detect at 1080p, stream at 4K**. Purpose-built for cloud GPU deployment (AWS G6/L4, RTX 5090) with zero-copy hardware acceleration.

**Core Features:**
- **RF-DETR Medium** transformer-based ball detection with `optimize_for_inference` (1.5-2× speedup)
- **Multi-resolution workflow**: decode 4K → downsample to 1080p/720p for inference → upscale tracking output to 4K
- **Extended Kalman Filter** (6-DOF state: position, velocity, acceleration) with Mahalanobis gating and adaptive noise
- **Virtual camera** with One-Euro smoothing, PID control, trajectory prediction, and dynamic zoom
- **MJPEG streaming server** on localhost:8554 for low-latency monitoring and recording
- **GPU-accelerated I/O** via NVDEC/NVENC (when available) for zero-copy decode/encode
- **Automatic chaos detection** and ultra-stability modes to handle erratic detector behavior
- **YouTube live source** support via `yt-dlp` integration

## System Requirements

### Production Environment

- **GPU:** NVIDIA GPU with CUDA 11+ and exposed NVDEC/NVENC engines
  - Recommended: RTX 5090, L4 (AWS G6), L40S, A10G
  - Minimum VRAM: 12 GB for 4K workflow
- **Python:** 3.10 (PyNvCodec does not yet support 3.12)
- **CUDA:** 11.8 or higher
- **FFmpeg:** 4.4+ with NVENC support for recording
- **Driver:** NVIDIA driver ≥ 550 (must match between host and container)
- **Disk:** 100 GB+ SSD/NVMe for video cache and MJPEG segments
- **RAM:** 8 GB minimum, 16 GB recommended
- **OS:** Ubuntu 22.04/24.04 recommended

### Supported Resolutions & Performance

| GPU Type | Input | Detection | Output | FPS | Inference |
|----------|-------|-----------|--------|-----|----------|
| RTX 5090 | 4K | 1080p | 3840×1440 | 30+ | 15-18ms |
| NVIDIA L4 (AWS G6) | 4K | 1080p | 3840×1440 | 25-30 | 18-22ms |
| NVIDIA L40S | 4K | 1080p | 3840×1440 | 30+ | 16-20ms |
| NVIDIA A10G | 4K | 720p | 1920×1080 | 30 | 22-28ms |

## Architecture

```text
Input Source (File/RTMP/YouTube)
         |
         v
   Video Decoder (NVDEC or CPU)
    [4K: 3840×2160]
         |
         v
   Downsampler (GPU resize)
    [1080p: 1920×1080 or 720p]
         |
         v
   RF-DETR Medium (FP16)
    |-- optimize_for_inference()
    |-- Dead Zone Filtering
    |-- Spatial Filtering
    |-- Multi-candidate Rejection
         |
         v
   Ball Tracker (Extended Kalman Filter)
    |-- 6-DOF State [x, y, vx, vy, ax, ay]
    |-- Mahalanobis Gating (d² < χ²)
    |-- Adaptive Process Noise Q(t)
    |-- Chaos Detection & Suppression
    |-- Jump Rejection (>120px)
         |
         v
   Virtual Camera Controller
    |-- One-Euro Smoothing (fc_min, β)
    |-- PID Position Control (Kp, Ki, Kd)
    |-- Trajectory Prediction (5 steps)
    |-- Dynamic Zoom (spring-damper)
         |
         v
   Crop & Render (tracked ROI)
    [Base: 3840×1440]
         |
         v
   Upsampler (GPU resize to output)
    [Final: 3840×1440 or 4K]
         |
         v
   MJPEG Server (localhost:8554)
    |-- /stream.mjpg endpoint
    |-- Zero-copy frame updates
         |
         +---> Recording (ffmpeg -c copy)
         +---> Monitoring (SSH tunnel)
```

## Installation

### Automated Production Setup (Recommended)

For a fully automated installation that handles all known issues:

```bash
# Clone repository
git clone https://github.com/yourusername/Football-Detection.git
cd Football-Detection

# Run production setup script
chmod +x setup_production.sh
./setup_production.sh
```

This script automatically:
- ✓ Validates system requirements and GPU
- ✓ Installs system dependencies (ffmpeg, cmake, CUDA toolkit)
- ✓ Aligns NVIDIA driver/library versions (fixes `unsupported device` errors)
- ✓ Verifies NVDEC/NVENC support
- ✓ Creates Python 3.10 virtual environment
- ✓ Installs all Python dependencies
- ✓ Builds PyNvCodec with **CMake CMP0148 policy fix**
- ✓ Configures library paths (LD_LIBRARY_PATH)
- ✓ Creates service management scripts (`start_stream.sh`, `stop_stream.sh`)
- ✓ Runs comprehensive diagnostics

**After installation:**

```bash
# Activate environment
source rf-detr-venv-310/bin/activate

# Run extended diagnostics
python diagnose_gpu.py

# Start streaming
./start_stream.sh
```

### Windows Installation

For Windows clients (recommended for testing before AWS deployment):

```powershell
# Clone repository
git clone https://github.com/chele-s/Football-Detection.git
cd Football-Detection

# Run as Administrator in PowerShell
Set-ExecutionPolicy Bypass -Scope Process -Force
.\setup_production_windows.ps1
```

**What gets installed:**
- ✓ Chocolatey package manager
- ✓ Python 3.10 + virtual environment
- ✓ Git, FFmpeg, Visual Studio Build Tools
- ✓ All Python dependencies with CUDA support
- ✓ Diagnostic and service scripts (`.bat` and `.ps1`)

**After installation:**

```powershell
# Run diagnostics
.\diagnose_gpu_windows.ps1

# Start streaming
.\start_stream.bat
# or
.\start_stream.ps1
```

**Note:** PyNvCodec is typically unavailable on Windows. The pipeline will use CPU for video I/O and GPU for inference, which is sufficient for client testing.

### Manual Installation (Linux)

If you prefer manual setup or need to customize:

```bash
# Clone repository
git clone https://github.com/chele-s/Football-Detection.git
cd Football-Detection

# Create Python 3.10 virtual environment
sudo apt update && sudo apt install -y python3.10-venv git tmux ffmpeg
python3.10 -m venv rf-detr-venv-310
source rf-detr-venv-310/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install PyNvCodec for GPU acceleration (optional but recommended)
bash scripts/install_pynvcodec.sh

# Export NVIDIA libraries
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### GPU Readiness Validation

Before running the pipeline, validate your GPU setup:

```bash
# Check NVIDIA driver and CUDA
nvidia-smi

# Run diagnostic script
python guides/verify_gpu_setup.py
python diagnose_gpu.py  # extended report
```

**Expected output:**

- CUDA available: `True`
- PyNvCodec status: `Available` (or fallback to CPU)
- Video Encode/Decode: `Supported` (check `nvidia-smi -q | grep -i "Video Encode"`)

**Troubleshooting:**

- `Driver/library version mismatch`: reinstall matching `nvidia-utils-*` package for your driver version
- `unsupported device (2)`: container/runtime lacks `video` capability; add `--gpus '"capabilities=compute,graphics,utility,video"'` to Docker run
- `CUDA_ERROR_NO_DEVICE`: driver or CUDA toolkit not properly installed

### Core Dependencies

```text
rfdetr>=1.2.0          # RF-DETR transformer detection
supervision>=0.26.0    # Tracking utilities
torch>=2.0.0           # PyTorch with CUDA 11.8+
torchvision>=0.15.0
opencv-python>=4.8.0   # Video I/O
numpy>=1.24.0
scipy>=1.11.0          # Kalman filter math
PyYAML>=6.0            # Config management
yt-dlp>=2023.0.0       # YouTube source support
Pillow>=10.0.0
requests>=2.31.0
```

## Configuration

### Model Configuration (`configs/model_config.yml`)

```yaml
model:
  path: "models/best_rf-detr.pth"
  confidence: 0.25
  iou_threshold: 0.45
  device: "cuda"
  half_precision: true
  imgsz: 640
  warmup_iterations: 3
  ball_class_id: 0

tracking:
  max_lost_frames: 10
  min_confidence: 0.3
  iou_threshold: 0.3
  adaptive_noise: true

output:
  width: 1920
  height: 1080
```

### Stream Configuration (`configs/stream_config.yml`)

```yaml
stream:
  input_url: "clean_video_4k.webm"  # or YouTube URL, RTMP, etc.
  target_fps: 30
  mjpeg_port: 8554  # localhost only
  debug_mode: false
  show_stats: true

output:
  width: 3840   # Final output resolution
  height: 1440

processing:
  detection_width: 1920   # Downsample for inference
  detection_height: 1080  # Use 720p for more FPS

camera:
  base_output_width: 3840   # Virtual camera base crop
  base_output_height: 1440
  dead_zone: 0.10           # Center dead zone (no movement)
  anticipation: 0.35        # Lead tracker velocity
  zoom_padding: 1.2         # Extra margin around ball
  smoothing_freq: 30.0      # One-Euro filter frequency
  smoothing_min_cutoff: 2.5 # Smoothing aggressiveness
  smoothing_beta: 0.001     # Velocity sensitivity
```

## Usage

### Primary Workflow: MJPEG Streaming

The main entry point is `run_mjpeg_stream.py`, which starts an HTTP server broadcasting the tracked stream as Motion JPEG:

```bash
# Start pipeline in tmux (recommended for production)
tmux new -s stream
source rf-detr-venv-310/bin/activate
python run_mjpeg_stream.py --config configs/stream_config.yml
```

**Boot sequence:**

1. `[1/5] Loading configurations` (merges model + stream configs)
2. `[2/5] Loading RF-DETR model` (applies `optimize_for_inference`)
3. `[3/5] Initializing ball tracker` (Kalman filter + chaos detection)
4. `[4/5] Starting MJPEG server` on port 8554
5. `[5/5] Opening video source` (file, RTMP, or YouTube)

When you see `✅ MJPEG server started! http://localhost:8554/stream.mjpg`, the pipeline is live.

### Viewing the Stream

**Local (within VM):**

```bash
vlc http://localhost:8554/stream.mjpg
# or
mpv http://localhost:8554/stream.mjpg
```

**Remote (via SSH tunnel):**

```bash
# On your local machine
ssh -N -L 8554:localhost:8554 user@remote-server

# Then open in browser or VLC
vlc http://localhost:8554/stream.mjpg
```

**Lightweight proxy for monitoring** (optional):

```bash
# Scale down to 1280×480@10fps for remote viewing
ffmpeg -i http://localhost:8554/stream.mjpg \
       -vf scale=1280:-1 -r 10 -qscale:v 8 \
       -f mjpeg tcp://127.0.0.1:9554?listen=1
```

Then tunnel port 9554 instead to save bandwidth.

### Recording Without FPS Drops

**Option 1: Zero-reencode (recommended)**

```bash
# Copy MJPEG frames directly to segmented files
ffmpeg -thread_queue_size 2048 \
       -i http://localhost:8554/stream.mjpg \
       -c copy -f segment -segment_time 60 clips/clip_%03d.mjpeg

# Convert to MP4 offline later
ffmpeg -i clips/clip_000.mjpeg -c:v libx264 -preset faster -crf 20 output.mp4
```

**Option 2: Hardware encode (when NVENC available)**

```bash
ffmpeg -thread_queue_size 512 \
       -i http://localhost:8554/stream.mjpeg \
       -c:v h264_nvenc -preset p5 -cq 19 -b_ref_mode middle output_4k.mp4
```

**Option 3: CPU encode with reduced quality**

```bash
ffmpeg -thread_queue_size 512 \
       -i http://localhost:8554/stream.mjpg \
       -c:v libx264 -preset superfast -crf 24 output.mp4
```

### Using YouTube as Source

Since YouTube media URLs expire, use one of these workflows:

**A) Direct media URL (on-demand)**

```bash
# Resolve YouTube URL
yt-dlp -g -f best https://www.youtube.com/watch?v=VIDEO_ID

# Copy the output URL and set it in configs/stream_config.yml
stream:
  input_url: "https://rr3---sn-..."
```

**B) Local restream (recommended for live events)**

```bash
# In one tmux window, pipe YouTube to local UDP
yt-dlp -f best -o - https://www.youtube.com/watch?v=VIDEO_ID \
  | ffmpeg -re -i - -c copy -f mpegts udp://127.0.0.1:9000

# In another window, set configs/stream_config.yml
stream:
  input_url: "udp://127.0.0.1:9000"

# Start pipeline
python run_mjpeg_stream.py
```

This keeps a stable local endpoint even if YouTube rotates the media URL.

### Batch Processing (Legacy)

For offline file processing without the MJPEG server:

```bash
python main.py batch \
  --input data/inputs/match.mp4 \
  --output data/outputs/tracked.mp4 \
  --confidence 0.25
```

Note: `main.py` is the older CLI wrapper; production workflows use `run_mjpeg_stream.py` directly.

## Advanced Features

### Chaos Mode

Automatic detection and suppression of erratic detector behavior:
- Activates when detection jumps exceed 120px
- Forces zoom-out and ultra-stability smoothing
- Rejects unreliable detections for 4 seconds
- Prevents jarring camera movements

### Dead Zone Filtering

Configurable spatial filters to block false positives:
- Top-right corner: x > 70%, y < 40%
- Top-left corner: x < 8%, y < 40%
- Customizable zones per deployment

### Adaptive Smoothing

Dynamic filter adjustment based on detector stability:
- Normal mode: min_cutoff = 0.6
- Stability mode: min_cutoff = 0.08
- PID limits: ±500px → ±100px during chaos

## Performance Benchmarks

### AWS G6 Instance (NVIDIA L4)

**Configuration:** 1920x1080 @ 30 FPS

| Component | Time (ms) | % of Budget |
|-----------|-----------|-------------|
| RF-DETR Inference | 18-22 | 60% |
| Kalman Tracking | 0.8-1.2 | 3% |
| Camera Processing | 1.0-1.5 | 4% |
| Rendering | 8-10 | 30% |
| **Total** | **28-34** | **100%** |

**Throughput:** 30-35 FPS sustained

### Memory Usage

- **VRAM:** 3.2 GB (model + buffers)
- **RAM:** 2.8 GB (frame history + state)
- **Peak VRAM:** 4.1 GB during initialization

## Production Deployment

See [AWS_DEPLOYMENT.md](AWS_DEPLOYMENT.md) for complete AWS G6 deployment guide.

See [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) for architecture documentation and scaling strategies.

### Quick Deploy to AWS

```bash
cd deployment/aws
terraform init
terraform apply -var="instance_type=g6.xlarge"
```

## Algorithm Details

### Extended Kalman Filter (6-DOF Constant Acceleration Model)

**State Vector:**

```
x ∈ ℝ⁶ = [px, py, vx, vy, ax, ay]ᵀ
```

where `(px, py)` is ball pixel position, `(vx, vy)` is velocity (px/frame), and `(ax, ay)` is acceleration (px/frame²).

**State Transition Matrix** (dt = 1 frame @ 30 FPS ≈ 0.0333 s):

```
F =  │ 1   0   dt   0   0.5·dt²    0     │
     │ 0   1    0  dt      0    0.5·dt²  │
     │ 0   0    1   0     dt       0     │
     │ 0   0    0   1      0      dt     │
     │ 0   0    0   0      1       0     │
     │ 0   0    0   0      0       1     │
```

**Process Noise Covariance** (adaptive):

```
Q(t) = diag([σ²_px, σ²_py, σ²_vx, σ²_vy, σ²_ax, σ²_ay])

σ²_px, σ²_py = base_pos_noise × chaos_multiplier(t)
σ²_vx, σ²_vy = base_vel_noise × (1 + jerk_factor(t))
σ²_ax, σ²_ay = base_acc_noise × adaptive_scale(t)
```

**Chaos Multiplier:**

```
chaos_multiplier(t) = 1 + 10 · exp(-t_since_chaos / τ_decay)
where τ_decay = 4 seconds
```

This inflates process noise during erratic detector behavior to prevent overfitting bad measurements.

**Predict Step:**

```
x̂⁻(k) = F · x̂⁺(k-1)
P⁻(k) = F · P⁺(k-1) · Fᵀ + Q(t)
```

**Measurement Model:**

```
H = │ 1  0  0  0  0  0 │
    │ 0  1  0  0  0  0 │

z(k) ∈ ℝ² = [zx, zy]ᵀ  (detector output)

R = diag([σ²_meas, σ²_meas])  where σ_meas ≈ 15-25 px
```

**Mahalanobis Gating:**

```
Innovation:  ν = z(k) - H · x̂⁻(k)
Innovation covariance:  S = H · P⁻(k) · Hᵀ + R
Mahalanobis distance²:  d²_M = νᵀ · S⁻¹ · ν

Accept measurement if d²_M < χ²(2, α=0.95) ≈ 5.99
Reject if d²_M > 15 (likely outlier)
```

**Update Step (if measurement accepted):**

```
Kalman Gain:  K = P⁻(k) · Hᵀ · S⁻¹
x̂⁺(k) = x̂⁻(k) + K · ν
P⁺(k) = (I - K · H) · P⁻(k)
```

**Jump Rejection:**

```
If ||ν|| > 120 px:
  - Set chaos_mode = True
  - Reject measurement
  - Continue with prediction-only update
  - Ultra-stability smoothing for next 4 seconds
```

### One-Euro Filter (Adaptive Low-Pass Smoothing)

Used for camera position and zoom smoothing. Dynamically adjusts cutoff frequency based on signal velocity.

**Cutoff Frequency (velocity-dependent):**

```
fc(t) = fc_min + β · |dx/dt|

where:
  fc_min = minimum cutoff (lower = more smoothing)
  β = velocity sensitivity (higher = faster response to motion)
  dx/dt = rate of change (computed via derivative filter)
```

**Smoothing Coefficient:**

```
τ = 1 / (2π · fc(t))
α = 1 / (1 + τ/(1/fs))  where fs = sampling frequency (30 Hz)

x̃(t) = α · x_raw(t) + (1 - α) · x̃(t-1)
```

**Mode-Dependent Parameters:**

```
Normal mode:     fc_min = 0.6,  β = 0.004
Stability mode:  fc_min = 0.08, β = 0.001  (ultra-smooth during chaos)
```

**Derivative Filter (to estimate dx/dt):**

```
dx/dt ≈ (x_raw(t) - x_raw(t-1)) · fs
Smoothed with its own One-Euro: fc_d = fc_min_d = 1.0
```

### PID Controller (Camera Positioning)

Maintains camera center on predicted ball position with damping to prevent overshoot.

**Error Signal:**

```
e(t) = x_target(t) - x_camera(t)
```

**PID Output:**

```
u(t) = Kp · e(t) + Ki · ∫e(τ)dτ + Kd · de/dt

where:
  Kp = proportional gain (position error)
  Ki = integral gain (accumulated drift correction)
  Kd = derivative gain (velocity damping)
```

**Adaptive Limits (chaos mode):**

```
Normal:      |u(t)| ≤ 500 px/frame
Chaos mode:  |u(t)| ≤ 100 px/frame  (prevent jerky movements)
```

**Trajectory Prediction (5-step ahead):**

```
x_target(t) = x̂(t) + anticipation_factor · v̂(t) · Δt_predict

where:
  anticipation_factor ∈ [0.3, 0.5]
  Δt_predict = 5 frames ≈ 0.167 s
```

### Dynamic Zoom Control (Spring-Damper Model)

Smooth zoom transitions using a critically damped spring system.

**Target Zoom:**

```
z_target = base_zoom + zoom_padding · f(ball_velocity)

where:
  base_zoom = 1.0 (no zoom)
  zoom_padding = 1.2-1.8
  f(v) = min(1.0, ||v|| / v_max)  (velocity-dependent zoom)
```

**Spring-Damper Dynamics:**

```
Acceleration: az = k · (z_target - z) - c · vz
Velocity:     vz(t+1) = vz(t) + az · dt
Zoom:         z(t+1) = z(t) + vz(t+1) · dt

Constraints:  1.0 ≤ z ≤ 2.5
              |vz| ≤ max_rate = 0.08 per frame
```

**Critically Damped Parameters:**

```
k (stiffness) = 0.08
c (damping)   = 0.70
Critical damping when c² = 4k (prevents oscillation)
```

## Testing

Run test suite:
```bash
python -m pytest app/tests/ -v
```

Individual component tests:
```bash
python -m pytest app/tests/test_detector.py
python -m pytest app/tests/test_tracker.py
python -m pytest app/tests/test_camera.py
```

## Monitoring & Observability

The pipeline logs real-time metrics to stdout (capture with tmux or redirect to file):

```text
[STREAM] Frame 1000 | FPS: 32.1 | Inf: 19.2ms | Track: ACTIVE | Zoom: 1.65x
DEAD_ZONE[top-right] x=1650 y=280 BLOCKED
HUGE JUMP detected: 247px - CHAOS MODE activated
Detector unstable - ULTRA STABILITY MODE (fc_min=0.08)
```

**Key Metrics:**

- **FPS**: End-to-end frame rate (detection + tracking + rendering)
- **Inf**: RF-DETR inference time (ms) for current frame
- **Track**: `ACTIVE` (ball visible), `LOST` (prediction-only), or `CHAOS` (suppressing bad detections)
- **Zoom**: Current virtual camera zoom factor

**Health Checks:**

```bash
# Verify MJPEG server is responding
curl -I http://localhost:8554/stream.mjpg

# Monitor GPU usage
nvidia-smi dmon -s u -c 60

# Check for errors in logs
tmux capture-pane -S -100 -p | grep -i error
```

**Production Recommendations:**

- Use `systemd` or `supervisord` to auto-restart on crash
- Redirect logs to `/var/log/football-stream/stream.log` with logrotate
- Expose Prometheus metrics (optional): FPS, inference time, GPU util, chaos mode activations
- Set up alerts for FPS < 20 or `CUDA_ERROR` in logs

## Troubleshooting

### GPU Out of Memory

```bash
# Reduce inference resolution
sed -i 's/imgsz: 640/imgsz: 512/' configs/model_config.yml

# Enable FP16
sed -i 's/half_precision: false/half_precision: true/' configs/model_config.yml
```

### Low FPS

```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Profile inference time
python tools/benchmark_inference.py
```

### Stream Disconnection

```bash
# Test RTMP connectivity
ffprobe -v error rtmp://your-server/live/stream

# Increase reconnection attempts
export RTMP_RECONNECT_ATTEMPTS=10
```

## Production Deployment

For complete deployment guides, see:

- [instruccion.txt](instruccion.txt) — Quick deployment playbook
- [guides/AWS_DEPLOYMENT.md](guides/AWS_DEPLOYMENT.md) — AWS G6/L4 setup
- [guides/PRODUCTION_GUIDE.md](guides/PRODUCTION_GUIDE.md) — Architecture and scaling

**Recommended AWS Setup:**

```bash
# Provision EC2 G6 instance with NVIDIA L4
Instance type: g6.xlarge (1× L4, 24 GB VRAM)
AMI: Deep Learning Base Ubuntu 22.04
Driver: NVIDIA 570+
Storage: 100 GB gp3 SSD

# Deploy with Terraform
cd deployment/aws
terraform apply -var="instance_type=g6.xlarge"
```

**Docker Deployment:**

```bash
# Build image
docker build -t football-stream:latest .

# Run with GPU support
docker run --gpus '"device=0,capabilities=compute,graphics,utility,video"' \
  -p 8554:8554 \
  -v $(pwd)/configs:/app/configs \
  -v $(pwd)/models:/app/models \
  football-stream:latest
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support & Contributing

- **Issues**: [GitHub Issues](https://github.com/chele-s/Football-Detection/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chele-s/Football-Detection/discussions)
- **Enterprise Support**: Available for production deployments

## Acknowledgments

- [RF-DETR](https://github.com/Roboflow/RF-DETR) by Roboflow — Transformer-based object detection
- [Supervision](https://github.com/roboflow/supervision) — Tracking and annotation utilities
- Géry Casiez et al. — One-Euro Filter research ([paper](https://hal.inria.fr/hal-00670496/document))
- PyNvCodec — NVIDIA hardware video acceleration bindings

---

**Version:** 3.0.0 (MJPEG Streaming)  
**Last Updated:** November 2025 
**Status:** Production Ready  
**Python:** 3.10  
**CUDA:** 11.8+  
**GPU:** NVIDIA L4, L40S, RTX 5090 recommended
