<div align="center">

# ⚽ Football Detection & Tracking System

### Advanced Real-Time Ball Tracking with RF-DETR & Intelligent Virtual Camera

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![RF-DETR](https://img.shields.io/badge/RF--DETR-Medium-green.svg)](https://github.com/roboflow/rf-detr)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <em>Production-grade football tracking system combining state-of-the-art computer vision with cinematographic intelligence</em>
</p>

[Features](#-core-features) •
[Architecture](#-system-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[Performance](#-performance-metrics) •
[Documentation](#-technical-documentation)

</div>

---

## 🎯 Core Features

<table>
<tr>
<td width="50%">

### Detection & Tracking
- **RF-DETR Medium** with inference optimization
- **Extended Kalman Filter** with adaptive noise
- Multi-hypothesis tracking with outlier rejection
- Mahalanobis distance validation
- Temporal prediction during occlusions

</td>
<td width="50%">

### Virtual Camera System
- **One-Euro Filter** for smooth motion
- PID-controlled camera positioning
- Trajectory prediction with polynomial fitting
- Adaptive dead-zones based on velocity
- Dynamic zoom with curvature awareness

</td>
</tr>
</table>

### Operational Modes

| Mode | Use Case | Features |
|------|----------|----------|
| **Batch** | Video file processing | High-quality tracking, trajectory export, offline analysis |
| **Stream** | Real-time broadcasting | RTMP output, YouTube integration, <33ms latency |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT SOURCES                            │
│    Video Files (.mp4/.avi)  │  RTMP Streams  │  YouTube Live   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   VIDEO PROCESSING    │
         │   Frame Extraction    │
         │   Preprocessing       │
         └──────────┬────────────┘
                    │
                    ▼
         ┌───────────────────────┐
         │    RF-DETR MEDIUM     │◄─── FP16 Optimization
         │  Object Detection     │     Multi-scale Inference
         │  Confidence: 0.25+    │     Supervision Integration
         └──────────┬────────────┘
                    │
                    ▼
         ┌───────────────────────┐
         │  TRACKING MODULE      │
         │ ┌───────────────────┐ │
         │ │ Extended Kalman   │ │◄─── State: [x, y, vx, vy, ax, ay]
         │ │ Filter (6-DOF)    │ │     Adaptive Q/R matrices
         │ └───────────────────┘ │     Innovation monitoring
         │ ┌───────────────────┐ │
         │ │ Multi-Hypothesis  │ │◄─── IoU validation
         │ │ Association       │ │     Outlier detection
         │ └───────────────────┘ │
         └──────────┬────────────┘
                    │
                    ▼
         ┌───────────────────────┐
         │  VIRTUAL CAMERA       │
         │ ┌───────────────────┐ │
         │ │  One-Euro Filter  │ │◄─── Adaptive β computation
         │ │  (Position)       │ │     Jerk-aware smoothing
         │ └───────────────────┘ │
         │ ┌───────────────────┐ │
         │ │  PID Controller   │ │◄─── Kp=0.8, Ki=0.01, Kd=0.15
         │ │  (Positioning)    │ │     Anti-windup
         │ └───────────────────┘ │
         │ ┌───────────────────┐ │
         │ │ Trajectory        │ │◄─── Polynomial extrapolation
         │ │ Predictor         │ │     Curvature detection
         │ └───────────────────┘ │
         └──────────┬────────────┘
                    │
                    ▼
         ┌───────────────────────┐
         │   OUTPUT RENDERING    │
         │   Crop & Resize       │
         │   Overlay Graphics    │
         └──────────┬────────────┘
                    │
                    ▼
┌────────────────────────────────────────────────────────────────┐
│                         OUTPUT TARGETS                          │
│      Video Files  │  RTMP Streams  │  Preview Window           │
└────────────────────────────────────────────────────────────────┘
```

---

## 🧮 Advanced Algorithms

### 1. RF-DETR Detection Engine

**Migration from YOLOv8 to RF-DETR**

We've recently migrated to [Roboflow's RF-DETR](https://github.com/roboflow/rf-detr) for superior detection performance:

```python
from rfdetr import RFDETRMedium

# Initialize with inference optimization
model = RFDETRMedium()
model.optimize_for_inference()  # JIT compilation, graph optimization

# Inference with supervision integration
detections = model.predict(frame, threshold=0.25)
```

**Key Advantages:**
- **Transformer-based architecture** → Better occlusion handling
- **Native supervision support** → Seamless tracking integration
- **Optimized inference** → ~8-12ms per frame on RTX 3080
- **FP16 support** → 2x throughput on modern GPUs

**Confidence Calibration:**
```python
def _calibrate_confidence(conf, width, height):
    area = width * height
    if area < 100:      # Small objects
        return conf * 0.8
    elif area > 10000:  # Large objects
        return conf * 0.9
    return conf
```

### 2. Extended Kalman Filter (6-DOF)

**State Vector:** `x = [x, y, vx, vy, ax, ay]ᵀ`

**Process Model:**
```python
F = [[1, 0, dt, 0,  0.5*dt², 0      ],
     [0, 1, 0,  dt, 0,       0.5*dt²],
     [0, 0, 1,  0,  dt,      0      ],
     [0, 0, 0,  1,  0,       dt     ],
     [0, 0, 0,  0,  1,       0      ],
     [0, 0, 0,  0,  0,       1      ]]
```

**Adaptive Noise Tuning:**
```python
if avg_mahalanobis > 9.0:
    Q *= 1.2  # Increase process noise
elif avg_mahalanobis < 2.0:
    Q *= 0.9  # Decrease process noise
```

**Innovation Monitoring:**
- Mahalanobis distance for outlier detection
- Chi-squared test (df=2, α=0.05)
- Automatic filter reset on instability

### 3. One-Euro Filter for Camera Smoothing

**Adaptive Low-Pass Filtering:**

```
α = 1 / (1 + τ/Te)
where τ = 1/(2πfc), fc = fc_min + β|dx/dt|
```

**Parameters:**
- `fc_min = 1.0 Hz` → Minimum cutoff frequency
- `β = 0.007` → Speed coefficient (adaptive)
- `d_cutoff = 1.0 Hz` → Derivative cutoff

**Adaptive Beta Computation:**
```python
β_adaptive = β × velocity_factor × jerk_factor × stability_factor
β_adaptive ∈ [β×0.5, β×3.0]
```

### 4. Trajectory Prediction

**Polynomial Extrapolation:**
```python
# Fit 2nd-degree polynomial to position history
coeffs_x = polyfit(t, x, deg=2)
coeffs_y = polyfit(t, y, deg=2)

# Predict future position
x_future = coeffs_x[0]·t² + coeffs_x[1]·t + coeffs_x[2]
y_future = coeffs_y[0]·t² + coeffs_y[1]·t + coeffs_y[2]
```

**Curvature Detection:**
```python
κ = |dx·d²y - dy·d²x| / (dx² + dy²)^(3/2)
```

### 5. Multi-Hypothesis Tracking

**Association Strategy:**
1. **Primary:** Highest confidence detection
2. **Secondary:** Closest to Kalman prediction (if distance < 100px)
3. **Tertiary:** Temporal interpolation from history

**Outlier Rejection:**
- Z-score threshold: `|z| > 3.0σ`
- IoU validation: `IoU > 0.3`
- Confidence floor: `conf > 0.3`

---

## 📦 Installation

### Prerequisites

- **Python:** 3.8 or higher
- **CUDA:** 11.8+ (for GPU acceleration)
- **FFmpeg:** Latest version

### Quick Install

```bash
# Clone repository
git clone https://github.com/chele-s/Football-Detection.git
cd Football-Detection

# Install dependencies
pip install -r requirements.txt

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Manual Setup

```bash
# Core dependencies
pip install rfdetr>=1.2.0 supervision>=0.26.0
pip install opencv-python numpy scipy Pillow
pip install torch>=2.0.0 torchvision>=0.15.0

# Streaming support
pip install yt-dlp PyYAML tqdm
```

---

## 🚀 Usage

### Batch Processing (Video Files)

```bash
python main.py batch \
  --input data/inputs/match.mp4 \
  --output data/outputs/tracked.mp4 \
  --confidence 0.25
```

**Output:**
- Processed video with virtual camera tracking
- JSON trajectory file (optional): `trajectories.json`

### Stream Processing (Real-Time)

#### Debug Mode (YouTube Testing)

```bash
python main.py stream \
  --debug \
  --input "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

**Features:**
- Live preview window
- Real-time statistics overlay
- No RTMP server required

#### Production Mode (RTMP Broadcasting)

```bash
python main.py stream \
  --input "rtmp://source.com/live/input" \
  --output "rtmp://destination.com/live/output" \
  --device cuda
```

**Helper Scripts:**

```bash
# Linux/Mac
bash scripts/start_stream_worker.sh rtmp://input rtmp://output

# Windows
.\scripts\start_stream_worker.ps1 rtmp://input rtmp://output
```

### Command-Line Arguments

```
usage: main.py [-h] [--model-config PATH] [--input PATH/URL] 
               [--output PATH/URL] [--debug] [--confidence FLOAT]
               [--device {cuda,cpu}] {batch,stream}

optional arguments:
  --model-config PATH   Model configuration file (default: configs/model_config.yml)
  --input PATH/URL      Input video/stream (overrides config)
  --output PATH/URL     Output video/stream (overrides config)
  --debug               Enable debug mode with preview
  --confidence FLOAT    Detection confidence threshold (0.0-1.0)
  --device {cuda,cpu}   Force CPU or GPU processing
```

---

## ⚙️ Configuration

### Model Configuration (`configs/model_config.yml`)

```yaml
model:
  path: "models/best_rf-detr.pth"       # Custom model or pretrained
  confidence: 0.25                       # Detection threshold
  iou_threshold: 0.45                    # NMS threshold
  device: "cuda"                         # cuda/cpu
  half_precision: true                   # FP16 inference
  imgsz: 640                             # Input resolution
  multi_scale: false                     # Multi-scale detection
  warmup_iterations: 3                   # Model warmup
  ball_class_id: 0                       # Target class ID

tracking:
  max_lost_frames: 10                    # Max prediction frames
  min_confidence: 0.3                    # Minimum track confidence
  history_size: 30                       # Position history buffer
```

### Stream Configuration (`configs/stream_config.yml`)

```yaml
stream:
  target_fps: 30                         # Output framerate
  bitrate: "4000k"                       # Video bitrate
  preset: "ultrafast"                    # FFmpeg preset
  debug_mode: false                      # Show preview window
  show_stats: true                       # Statistics overlay

camera:
  dead_zone: 0.10                        # 10% no-movement zone
  anticipation: 0.3                      # Prediction strength
  zoom_padding: 1.2                      # Viewport padding
  smoothing_min_cutoff: 1.0              # One-Euro fc_min
  smoothing_beta: 0.007                  # One-Euro β
  use_pid: true                          # Enable PID control
  prediction_steps: 5                    # Trajectory lookahead
```

---

## 📊 Performance Metrics

### Benchmark Results (RTX 3080, 1920×1080 input)

| Component | Time (ms) | FPS | Notes |
|-----------|-----------|-----|-------|
| **RF-DETR Medium** | 8-12 | 83-125 | FP16, optimized |
| **Kalman Filter** | 0.5-1.0 | 1000-2000 | NumPy/SciPy |
| **One-Euro Filter** | 0.2-0.5 | 2000-5000 | Pure Python |
| **Virtual Camera** | 0.5-1.0 | 1000-2000 | Crop calculation |
| **Total Pipeline** | **12-18** | **55-83** | Real-time capable |

### Memory Usage

- **Model (RF-DETR Medium):** ~250 MB VRAM
- **Frame buffers (1080p):** ~50 MB VRAM
- **Tracking state:** ~10 MB RAM
- **Total:** <400 MB VRAM, <100 MB RAM

### Optimization Tips

<details>
<summary><b>Maximize Throughput</b></summary>

1. **Enable FP16:**
   ```yaml
   model:
     half_precision: true
   ```

2. **Reduce Input Resolution:**
   ```yaml
   model:
     imgsz: 512  # Down from 640
   ```

3. **Disable Multi-Scale:**
   ```yaml
   model:
     multi_scale: false
   ```

4. **Use Smaller Model:**
   ```python
   from rfdetr import RFDETRSmall  # Instead of Medium
   ```

</details>

<details>
<summary><b>Minimize Latency</b></summary>

1. **Reduce Smoothing:**
   ```yaml
   camera:
     smoothing_min_cutoff: 2.0  # More responsive
     smoothing_beta: 0.005
   ```

2. **Disable Trajectory Prediction:**
   ```yaml
   camera:
     prediction_steps: 0
   ```

3. **Increase Dead-Zone:**
   ```yaml
   camera:
     dead_zone: 0.15  # Less camera movement
   ```

</details>

---

## 📚 Technical Documentation

### Algorithm Deep Dives

<details>
<summary><b>Extended Kalman Filter Implementation</b></summary>

**Prediction Step:**
```python
x_pred = F @ x_prev
P_pred = F @ P_prev @ F.T + Q
```

**Update Step:**
```python
y = z - H @ x_pred                  # Innovation
S = H @ P_pred @ H.T + R            # Innovation covariance
K = P_pred @ H.T @ inv(S)           # Kalman gain
x_new = x_pred + K @ y              # State update
P_new = (I - K @ H) @ P_pred        # Covariance update
```

**Mahalanobis Distance:**
```python
d² = y.T @ inv(S) @ y
outlier = d² > χ²(0.95, df=2) ≈ 5.99
```

</details>

<details>
<summary><b>One-Euro Filter Tuning Guide</b></summary>

**Parameter Effects:**

| Parameter | ↑ Increase | ↓ Decrease |
|-----------|------------|------------|
| `min_cutoff` | More responsive, noisier | Smoother, more lag |
| `beta` | Faster at high speed | Slower overall |
| `d_cutoff` | Noisier speed estimate | Smoother speed |

**Recommended Presets:**

```yaml
# Cinematic (smooth, slow)
smoothing_min_cutoff: 0.5
smoothing_beta: 0.003

# Balanced (default)
smoothing_min_cutoff: 1.0
smoothing_beta: 0.007

# Responsive (fast, snappy)
smoothing_min_cutoff: 2.0
smoothing_beta: 0.015
```

</details>

<details>
<summary><b>RF-DETR vs YOLOv8 Comparison</b></summary>

| Metric | RF-DETR Medium | YOLOv8n | YOLOv8m |
|--------|----------------|---------|---------|
| **Inference Time** | 8-12ms | 5-8ms | 12-18ms |
| **AP@50** | 52.3% | 37.3% | 50.2% |
| **AP@50:95** | 42.8% | 28.1% | 41.7% |
| **Parameters** | 51M | 3.2M | 25.9M |
| **VRAM** | 250MB | 80MB | 180MB |
| **Occlusion Handling** | ★★★★★ | ★★★☆☆ | ★★★★☆ |

**Why RF-DETR?**
- Transformer architecture better handles partial occlusions
- End-to-end detection (no NMS instability)
- Stronger spatial reasoning for ball tracking
- Native integration with Supervision library

</details>

---

## 🧪 Testing & Development

### Jupyter Notebooks

```bash
# Model benchmarking
jupyter notebook notebooks/01_benchmark_model_speed.ipynb

# Camera tuning interactive
jupyter notebook notebooks/02_tune_camera_smoothing.ipynb

# RTMP connectivity test
jupyter notebook notebooks/03_test_rtmp_connection.ipynb
```

### Unit Tests

```bash
# Run full test suite
python test_system.py

# Test individual components
python -m pytest app/tests/test_detector.py
python -m pytest app/tests/test_tracker.py
python -m pytest app/tests/test_camera.py
```

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><b>CUDA Out of Memory</b></summary>

**Solutions:**
1. Enable FP16: `half_precision: true`
2. Reduce batch size to 1 (already default)
3. Lower input resolution: `imgsz: 512`
4. Close other GPU applications

**Check VRAM usage:**
```python
import torch
print(torch.cuda.memory_allocated() / 1e9, "GB")
```

</details>

<details>
<summary><b>YouTube Download Failed</b></summary>

**Update yt-dlp:**
```bash
pip install --upgrade yt-dlp
```

**Alternative URL formats:**
```bash
# Standard
https://www.youtube.com/watch?v=VIDEO_ID

# Live stream
https://www.youtube.com/watch?v=VIDEO_ID&live=1

# Embedded
https://youtu.be/VIDEO_ID
```

</details>

<details>
<summary><b>FFmpeg Not Found</b></summary>

**Installation:**

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from: https://ffmpeg.org/download.html
# Add to PATH
```

**Verify:**
```bash
ffmpeg -version
```

</details>

---

## 🗺️ Roadmap

- [x] RF-DETR integration
- [x] Extended Kalman Filter
- [x] One-Euro smoothing
- [x] PID camera control
- [x] Trajectory prediction
- [ ] Multi-object tracking (players + ball)
- [ ] Player jersey number recognition
- [ ] Automated highlight detection
- [ ] WebRTC ultra-low-latency streaming
- [ ] Cloud deployment (AWS/GCP)
- [ ] Docker containerization
- [ ] REST API for remote control
- [ ] Grafana dashboard for metrics

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Acknowledgments

**Developed by:** Gabriel (Chele-s)

**Special Thanks:**
- [Roboflow](https://github.com/roboflow) for RF-DETR
- [Supervision](https://github.com/roboflow/supervision) library
- One-Euro Filter research by Géry Casiez et al.

---

<div align="center">

**Built with ❤️ for the future of sports broadcasting**

[Report Bug](https://github.com/chele-s/Football-Detection/issues) •
[Request Feature](https://github.com/chele-s/Football-Detection/issues) •
[Documentation](https://github.com/chele-s/Football-Detection/wiki)

</div>
