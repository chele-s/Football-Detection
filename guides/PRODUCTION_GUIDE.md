# Production Deployment Guide

Technical documentation for production deployment of the Football Detection & Tracking System.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Documentation](#component-documentation)
3. [API Reference](#api-reference)
4. [Configuration Management](#configuration-management)
5. [Performance Optimization](#performance-optimization)
6. [Failure Modes & Recovery](#failure-modes--recovery)
7. [Scaling Strategies](#scaling-strategies)
8. [Security Considerations](#security-considerations)

---

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRODUCTION STACK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Input      │  │   Detection  │  │   Tracking   │          │
│  │   Layer      │──▶   Engine     │──▶   Module     │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│        │                  │                  │                   │
│        │                  │                  │                   │
│  [VideoReader]    [RF-DETR Medium]   [Kalman Filter]           │
│  - RTMP Input     - FP16 Inference   - 6-DOF State             │
│  - H.264 Decode   - Dead Zones       - Mahalanobis Gate        │
│  - Buffering      - Spatial Filter   - Chaos Detection         │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Camera     │  │   Renderer   │  │   Output     │          │
│  │   System     │──▶   Pipeline   │──▶   Layer      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│        │                  │                  │                   │
│  [VirtualCamera]    [OpenCV/FFmpeg]    [FFMPEGWriter]          │
│  - One-Euro       - Crop & Resize     - NVENC H.264            │
│  - PID Control    - Overlay Stats     - RTMP Streaming         │
│  - Trajectory     - Color Space       - Bitrate Control        │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Stage**: RTMP stream decoded at 30 FPS
2. **Detection Stage**: RF-DETR processes frame in 18-22ms
3. **Tracking Stage**: Kalman filter updates state in 1-2ms
4. **Camera Stage**: Virtual camera computes crop in 1-2ms
5. **Render Stage**: Frame cropped, resized, overlaid in 8-10ms
6. **Output Stage**: NVENC encodes and streams via RTMP

**Total Latency**: 28-36ms (glass-to-glass: ~100ms including network)

---

## Component Documentation

### 1. BallDetector (`app/inference/detector.py`)

**Purpose**: Ball detection using RF-DETR transformer model

**Key Methods**:

```python
class BallDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: str = "cuda",
        half_precision: bool = True,
        imgsz: int = 640
    )
    
    def predict_ball_only(
        self,
        frame: np.ndarray,
        ball_class_id: int = 0,
        return_candidates: bool = False
    ) -> Optional[Tuple[float, float, float, float, float]]
```

**Performance Characteristics**:
- Inference time: 18-22ms (L4 GPU, FP16)
- VRAM usage: ~2.5GB
- Batch size: 1 (real-time constraint)

**Error Handling**:
- Model load failure: Falls back to CPU (with warning)
- GPU OOM: Automatic retry with FP32
- Invalid input: Returns None, logs error

### 2. BallTracker (`app/tracking/tracker.py`)

**Purpose**: State estimation and outlier rejection

**State Vector**: `[x, y, vx, vy, ax, ay]`

**Key Methods**:

```python
class BallTracker:
    def update(
        self,
        detection: Optional[Tuple],
        detections_list: Optional[List[Tuple]]
    ) -> Optional[Tuple[float, float, bool]]
    
    def get_state(self) -> Dict:
        return {
            'is_tracking': bool,
            'velocity_magnitude': float,
            'chaos_mode': bool,
            'stats': Dict
        }
```

**Operational Modes**:

| Mode | Trigger | Behavior |
|------|---------|----------|
| Normal | Consecutive detections > 15 | Standard Kalman filtering |
| Recovering | Lost frames 1-5 | Relaxed gating thresholds |
| Chaos | Jump > 120px | Reject new detections, zoom out |
| Stability Lock | Movement < 5px for 25 frames | Tight 80px acceptance radius |

**Parameters**:
- Process noise (Q): 0.01 (adaptive)
- Measurement noise (R): 3-10 (dynamic based on stability)
- Max lost frames: 10 (300ms at 30 FPS)

### 3. VirtualCamera (`app/camera/virtual_camera.py`)

**Purpose**: Cinematic camera control with smooth motion

**Key Components**:

```python
class VirtualCamera:
    position_filter: OneEuroFilter  # Position smoothing
    pid_x: PIDController           # X-axis positioning
    pid_y: PIDController           # Y-axis positioning
    trajectory_predictor: TrajectoryPredictor  # Future position
```

**PID Parameters**:
```
Kp = 0.8   (Proportional gain)
Ki = 0.01  (Integral gain - anti-windup enabled)
Kd = 0.15  (Derivative gain)
```

**One-Euro Filter**:
```
fc_min = 0.6 Hz (normal) / 0.08 Hz (stability mode)
beta = 0.004
d_cutoff = 1.0 Hz
```

**Output Limits**:
- Normal: ±500 px/frame
- Stability mode: ±100 px/frame

### 4. OneEuroFilter (`app/camera/one_euro_filter.py`)

**Purpose**: Adaptive low-pass filtering for jitter-free motion

**Algorithm**:
```python
# Compute velocity
dx = (x - x_prev) * freq

# Adaptive cutoff
fc = fc_min + beta * abs(dx)

# Low-pass filter
alpha = 1 / (1 + tau / Te)
x_filtered = alpha * x + (1 - alpha) * x_prev
```

**Adaptive Beta Computation**:
```python
beta_adaptive = beta * velocity_factor * jerk_factor * stability_factor
beta_adaptive = clip(beta_adaptive, beta * 0.5, beta * 3.0)
```

---

## API Reference

### Configuration Files

#### model_config.yml

```yaml
model:
  path: str                    # Path to .pth model file
  confidence: float            # Detection threshold (0.0-1.0)
  iou_threshold: float         # NMS IoU threshold
  device: str                  # "cuda" | "cpu"
  half_precision: bool         # Enable FP16
  imgsz: int                   # Input resolution
  warmup_iterations: int       # Warmup runs
  ball_class_id: int          # Target class (0 for ball)

tracking:
  max_lost_frames: int        # Max prediction frames
  min_confidence: float       # Minimum detection confidence
  iou_threshold: float        # IoU for association
  adaptive_noise: bool        # Enable adaptive Q/R

output:
  width: int                  # Output width
  height: int                 # Output height
```

#### stream_config.yml

```yaml
stream:
  target_fps: int             # Target framerate
  bitrate: str                # Video bitrate (e.g. "4000k")
  preset: str                 # FFmpeg preset
  debug_mode: bool            # Show preview window
  show_stats: bool            # Display overlay

camera:
  dead_zone: float            # Dead zone percentage
  anticipation: float         # Prediction strength
  zoom_padding: float         # Viewport padding multiplier
  smoothing_min_cutoff: float # One-Euro fc_min
  smoothing_beta: float       # One-Euro beta
  use_pid: bool               # Enable PID control
  prediction_steps: int       # Trajectory lookahead frames
```

### Command-Line Interface

```bash
python main.py {batch|stream} [OPTIONS]

Options:
  --model-config PATH       Path to model configuration YAML
  --input PATH/URL          Input video file or RTMP stream
  --output PATH/URL         Output video file or RTMP stream
  --debug                   Enable debug mode (preview window)
  --confidence FLOAT        Override detection confidence threshold
  --device {cuda,cpu}       Force processing device
```

### Exit Codes

```
0   - Success
1   - Configuration error
2   - Model load failure
3   - Input source error
4   - Output destination error
5   - GPU/CUDA error
6   - Runtime exception
```

---

## Configuration Management

### Environment Variables

```bash
# CUDA configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

# Performance tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Logging
export LOG_LEVEL=INFO
export LOG_FILE=/var/log/football-tracker/app.log

# RTMP configuration
export RTMP_TIMEOUT=5000
export RTMP_BUFFER_SIZE=4096000
```

### Configuration Hierarchy

1. Default values (hardcoded)
2. Config files (`configs/*.yml`)
3. Environment variables
4. Command-line arguments (highest priority)

### Dynamic Configuration Updates

For runtime parameter changes without restart:

```python
# Example: Update detection confidence
import yaml

config = yaml.safe_load(open('configs/model_config.yml'))
config['model']['confidence'] = 0.30
detector.set_confidence_threshold(0.30)
```

**Hot-reloadable parameters**:
- Detection confidence
- Smoothing parameters
- Dead zone coordinates
- Output bitrate

**Requires restart**:
- Model path
- Input/output sources
- GPU device selection
- Image size

---

## Performance Optimization

### GPU Utilization

Target: 75-85% GPU utilization for optimal throughput/latency balance

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Profile application
nsys profile --stats=true python main.py stream --input test.mp4
```

### Memory Management

**VRAM Budget** (24GB L4):
- Model weights: 2.5GB
- Input tensors: 0.5GB
- Feature maps: 0.8GB
- Output buffers: 0.2GB
- Overhead: 0.5GB
- **Total**: ~4.5GB
- **Available**: 19.5GB (can run 5 concurrent streams)

**RAM Budget** (16GB instance):
- Python runtime: 0.5GB
- Frame buffers: 1.5GB
- State history: 0.5GB
- FFmpeg: 0.5GB
- **Total**: 3GB
- **Available**: 13GB

### Batch Processing

For multiple concurrent streams:

```python
# Process 4 streams in parallel
streams = [
    ('rtmp://server/live/stream1', 'rtmp://out/stream1'),
    ('rtmp://server/live/stream2', 'rtmp://out/stream2'),
    ('rtmp://server/live/stream3', 'rtmp://out/stream3'),
    ('rtmp://server/live/stream4', 'rtmp://out/stream4'),
]

import multiprocessing as mp

def process_stream(input_url, output_url):
    # Each process gets dedicated GPU memory slice
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # ... run pipeline

with mp.Pool(4) as pool:
    pool.starmap(process_stream, streams)
```

### Network Optimization

```bash
# TCP tuning for RTMP
sudo sysctl -w net.ipv4.tcp_window_scaling=1
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728

# Reduce TCP retransmissions
sudo sysctl -w net.ipv4.tcp_retries2=8
```

---

## Failure Modes & Recovery

### Detection Failures

**Symptom**: No detections for extended period

**Causes**:
1. Occluded ball
2. Camera angle change
3. Lighting conditions
4. Model confidence threshold too high

**Recovery**:
- Kalman prediction continues for `max_lost_frames` (10 frames)
- Camera performs gentle zoom-out
- Search pattern expands to field center

**Monitoring**:
```python
if tracker.lost_frames > 5:
    logger.warning(f"Extended tracking loss: {tracker.lost_frames} frames")
```

### Chaos Mode Activation

**Symptom**: "HUGE JUMP detected" messages, zoom-out

**Causes**:
1. Multiple false positives (lights, spectators)
2. Rapid camera movement in source feed
3. Model confusion during replays

**Recovery**:
- Automatic activation for 120 frames (4 seconds)
- Rejects detections > 80px from last position
- Forces zoom=1.0 and ultra-smoothing
- Self-deactivates after cooldown

**Prevention**:
- Properly configured dead zones
- Spatial filtering of non-field regions
- Model fine-tuning for venue

### GPU Errors

**Symptom**: "CUDA out of memory" or GPU hang

**Recovery**:
1. Automatic model unload
2. Clear CUDA cache: `torch.cuda.empty_cache()`
3. Reload model with FP32 (slower but more stable)
4. If persistent, restart service

**Monitoring**:
```bash
# GPU memory watchdog
while true; do
  MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ $MEM -gt 20000 ]; then
    systemctl restart football-tracker
  fi
  sleep 30
done
```

### Stream Disconnection

**Symptom**: "Connection refused" or "End of stream"

**Recovery**:
1. Exponential backoff reconnection (1s, 2s, 4s, 8s, ...)
2. Buffer replay of last known good position
3. Alert monitoring system after 5 failed attempts

**Configuration**:
```python
RTMP_RECONNECT_ATTEMPTS = 10
RTMP_RECONNECT_DELAY = 2  # seconds
```

---

## Scaling Strategies

### Horizontal Scaling

**Single Stream per Instance**:
- Simplest deployment
- Each instance handles 1 1080p stream
- Easy to scale with auto-scaling group
- Higher cost per stream

**Multiple Streams per Instance**:
- 4-6 streams on g6.2xlarge (2x L4)
- Process isolation via multiprocessing
- Shared GPU memory pool
- Lower cost per stream

### Vertical Scaling

| Instance | Streams | Cost/Stream/Month |
|----------|---------|-------------------|
| g6.xlarge | 1 | $876 |
| g6.2xlarge | 4 | $300 |
| g6.4xlarge | 8 | $175 |

**Recommendation**: Use g6.2xlarge for 3-4 streams (optimal cost/complexity)

### Load Balancing

```
┌──────────────┐
│  ELB/ALB     │ (Health checks on /health endpoint)
└──────┬───────┘
       │
   ┌───┴─────┬──────────┬──────────┐
   │         │          │          │
   v         v          v          v
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ G6.1 │ │ G6.2 │ │ G6.3 │ │ G6.4 │
└──────┘ └──────┘ └──────┘ └──────┘
```

### Geographic Distribution

For global deployment:

```
us-east-1 (Virginia)    → North America
eu-west-1 (Ireland)     → Europe
ap-southeast-1 (Singapore) → Asia-Pacific
```

Latency considerations:
- Intra-region: <5ms
- Cross-region: 50-200ms
- Use CloudFront for output distribution

---

## Security Considerations

### Network Security

1. **VPC Configuration**:
   - Private subnets for processing instances
   - NAT gateway for outbound-only internet
   - Security groups with least-privilege

2. **RTMP Security**:
   - Use RTMPS (TLS) when possible
   - IP whitelist for input sources
   - Token-based authentication

3. **Secrets Management**:
   ```python
   import boto3
   
   def get_secret(secret_name):
       client = boto3.client('secretsmanager')
       response = client.get_secret_value(SecretId=secret_name)
       return response['SecretString']
   ```

### Application Security

1. **Input Validation**:
   ```python
   def validate_rtmp_url(url: str) -> bool:
       if not url.startswith('rtmp://'):
           raise ValueError('Invalid RTMP URL')
       return True
   ```

2. **Resource Limits**:
   ```python
   # Prevent runaway memory usage
   import resource
   resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
   ```

3. **Error Sanitization**:
   - Never expose internal paths in logs
   - Sanitize user input before logging
   - Use structured logging (JSON)

### Compliance

**Data Handling**:
- Video streams processed in-memory only
- No persistent storage of video data
- GDPR-compliant (no PII collected)

**Audit Logging**:
- CloudTrail for API calls
- VPC Flow Logs for network traffic
- Application logs to CloudWatch

---

## Monitoring & Alerting

### Key Metrics

| Metric | Threshold | Action |
|--------|-----------|--------|
| FPS | < 25 | Alert, check GPU utilization |
| GPU Util | > 95% | Scale up or optimize |
| VRAM | > 20GB | Risk of OOM, reduce load |
| Latency | > 50ms | Investigate bottleneck |
| Lost Frames | > 20% | Check detection model |
| Chaos Mode | > 10%  | Tune thresholds |

### CloudWatch Alarms

```bash
# Create FPS alarm
aws cloudwatch put-metric-alarm \
  --alarm-name football-tracker-low-fps \
  --alarm-description "Alert when FPS drops below 25" \
  --metric-name FPS \
  --namespace FootballTracker \
  --statistic Average \
  --period 60 \
  --evaluation-periods 2 \
  --threshold 25 \
  --comparison-operator LessThanThreshold \
  --alarm-actions arn:aws:sns:region:account:topic
```

### Dashboard

Create CloudWatch dashboard with:
- Real-time FPS graph
- GPU utilization heatmap
- Inference time histogram
- Stream health status
- Error rate timeline

---

## Appendix

### Glossary

- **Chaos Mode**: Emergency stability mode triggered by erratic detections
- **Dead Zone**: Spatial region where detections are automatically rejected
- **Gating**: Statistical test for outlier rejection (Mahalanobis distance)
- **One-Euro Filter**: Adaptive low-pass filter with velocity-dependent cutoff
- **ROI**: Region of Interest - cropped detection region for performance

### References

1. RF-DETR Paper: https://arxiv.org/abs/2304.07788
2. Kalman Filter: https://www.kalmanfilter.net/
3. One-Euro Filter: http://cristal.univ-lille.fr/~casiez/1euro/
4. AWS G6 Instances: https://aws.amazon.com/ec2/instance-types/g6/

### Change Log

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2025 | Chaos mode, dead zones, production deployment |
| 1.5.0 | 2025 | One-Euro filter, PID control |
| 1.0.0 | 2025 | Initial RF-DETR implementation |

---

**Document Version:** 2.0  
**Maintained By:** Engineering Team  
**Last Review:** 2025  
**Next Review:** Quarterly

