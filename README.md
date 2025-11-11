# Football Detection & Tracking System

**Production-grade real-time ball tracking system combining RF-DETR object detection with advanced Kalman filtering and intelligent virtual camera control.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This system provides broadcast-quality football tracking with sub-33ms latency, capable of processing 1080p streams at 30+ FPS on modern GPU infrastructure. Built for production deployment on AWS GPU instances.

**Key Capabilities:**
- Real-time ball detection using RF-DETR transformer architecture
- Extended Kalman Filter with adaptive noise and outlier rejection
- Cinematic camera control with One-Euro smoothing and PID positioning
- RTMP streaming with hardware-accelerated encoding
- Automatic chaos detection and stability modes for robust operation

## System Requirements

### Production Environment
- **GPU:** NVIDIA L4, A100, or T4 with 16GB+ VRAM
- **Python:** 3.8 or higher
- **CUDA:** 11.8 or higher
- **FFmpeg:** 4.4+ with NVENC support
- **RAM:** 8GB minimum, 16GB recommended
- **OS:** Linux (Ubuntu 20.04+ recommended)

### Supported Resolutions

| GPU Type | Max Resolution | FPS | Inference Time |
|----------|---------------|-----|----------------|
| NVIDIA A100 | 1920x1080 | 60+ | 12-15ms |
| NVIDIA L4 (AWS G6) | 1920x1080 | 30+ | 18-22ms |
| NVIDIA T4 | 1280x720 | 30 | 28-32ms |

## Architecture

```
Input Stream (RTMP/File)
    |
    v
Video Reader (OpenCV/FFmpeg)
    |
    v
RF-DETR Medium Detector (FP16)
    |--- Dead Zone Filtering
    |--- Spatial Filtering
    |--- Multi-candidate Rejection
    v
Ball Tracker (Extended Kalman Filter)
    |--- 6-DOF State Estimation
    |--- Mahalanobis Gating
    |--- Chaos Mode Detection
    |--- Jump Rejection
    v
Virtual Camera System
    |--- One-Euro Filter (Smoothing)
    |--- PID Controller (Positioning)
    |--- Trajectory Prediction
    |--- Dynamic Zoom Control
    v
Output Renderer (NVENC)
    |
    v
Output Stream (RTMP/File)
```

## Installation

### Quick Start

```bash
git clone https://github.com/yourusername/Football-Detection.git
cd Football-Detection

pip install -r requirements.txt

python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Dependencies

Core libraries:
```
rfdetr>=1.2.0
supervision>=0.26.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
PyYAML>=6.0
yt-dlp>=2023.3.4
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
  target_fps: 30
  bitrate: "4000k"
  preset: "ultrafast"
  debug_mode: false
  show_stats: true

camera:
  dead_zone: 0.10
  anticipation: 0.3
  zoom_padding: 1.2
  smoothing_min_cutoff: 0.6
  smoothing_beta: 0.004
  use_pid: true
  prediction_steps: 5
```

## Usage

### Batch Processing

Process video files offline:

```bash
python main.py batch \
  --input data/inputs/match.mp4 \
  --output data/outputs/tracked.mp4 \
  --confidence 0.25
```

### Stream Processing

Real-time RTMP streaming:

```bash
python main.py stream \
  --input "rtmp://source.example.com/live/input" \
  --output "rtmp://destination.example.com/live/output" \
  --device cuda
```

Debug mode with preview:

```bash
python main.py stream \
  --debug \
  --input "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Command-Line Options

```
usage: main.py [-h] [--model-config PATH] [--input PATH/URL] 
               [--output PATH/URL] [--debug] [--confidence FLOAT]
               [--device {cuda,cpu}] {batch,stream}

positional arguments:
  {batch,stream}        Processing mode

optional arguments:
  --model-config PATH   Model configuration file
  --input PATH/URL      Input video/stream
  --output PATH/URL     Output video/stream
  --debug               Enable debug mode with preview
  --confidence FLOAT    Detection confidence threshold (0.0-1.0)
  --device {cuda,cpu}   Processing device
```

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

### Extended Kalman Filter

State vector: `[x, y, vx, vy, ax, ay]`

Process model with constant acceleration:
```
x(k+1) = F * x(k) + w
where F is the state transition matrix
w ~ N(0, Q) is process noise
```

Measurement update with Mahalanobis gating:
```
d² = (z - Hx)ᵀ S⁻¹ (z - Hx)
Accept measurement if d² < threshold
```

### One-Euro Filter

Adaptive low-pass filter with velocity-dependent cutoff:
```
fc = fc_min + β * |dx/dt|
α = 1 / (1 + τ/Te)
x_filtered = α * x_raw + (1-α) * x_prev
```

Parameters auto-tune based on jerk and stability metrics.

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

## Monitoring

The system exposes real-time metrics via logging:

```
[STREAM] Frame 1000 | FPS: 32.1 | Inf: 19.2ms | Track: ACTIVE | Zoom: 1.65x
DEAD_ZONE[top-right] x=1650 y=280 BLOCKED
HUGE JUMP detected: 247px - CHAOS MODE
Detector unstable - ULTRA STABILITY MODE (smoothing=0.08)
```

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

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Support

For production support and enterprise licensing:
- GitHub Issues: [Report Bug](https://github.com/yourusername/Football-Detection/issues)
- Enterprise Support: contact@example.com

## Acknowledgments

- RF-DETR by Roboflow
- Supervision library for tracking utilities
- One-Euro Filter research by Géry Casiez et al.

---

**Version:** 2.0.0  
**Last Updated:** 2024  
**Status:** Production Ready
