# Production Quick Start Guide

Complete setup guide for deploying RF-DETR Football Detection in production environments.

## Prerequisites

- Ubuntu 22.04/24.04 (recommended)
- NVIDIA GPU with CUDA 11.8+
- Driver version ≥ 550
- Root/sudo access
- Internet connection

## One-Command Setup

```bash
git clone https://github.com/yourusername/Football-Detection.git
cd Football-Detection
chmod +x setup_production.sh
./setup_production.sh
```

**Duration:** ~10-15 minutes depending on internet speed and GPU compilation.

## What Gets Installed

The setup script handles:

### System Packages
- Python 3.10 + venv + dev headers
- Build tools (gcc, g++, cmake, pkg-config)
- FFmpeg with NVENC support
- NVIDIA CUDA toolkit
- tmux for process management

### NVIDIA Libraries (Version-Aligned)
- `nvidia-utils-{driver_version}`
- `libnvidia-encode-{driver_version}`
- `libnvidia-decode-{driver_version}`

**Fixes:** "Driver/library version mismatch" errors

### Python Environment
- Virtual environment: `rf-detr-venv-310`
- PyTorch 2.0+ with CUDA 11.8
- RF-DETR transformer model
- OpenCV, NumPy, SciPy
- Supervision tracking utilities
- yt-dlp for YouTube sources

### PyNvCodec (GPU Video Acceleration)
- Built from source with **CMake CMP0148 policy fix**
- Enables zero-copy NVDEC/NVENC
- Automatic fallback to CPU if unavailable

**Fixes:** CMake policy warnings during compilation

## Post-Installation Verification

```bash
# Activate environment
source rf-detr-venv-310/bin/activate

# Run comprehensive diagnostics
python diagnose_gpu.py
```

**Expected output:**
```
✓ NVIDIA Driver: 550.x
✓ CUDA Toolkit: 11.8
✓ PyTorch CUDA: Available
✓ NVENC: Supported
✓ NVDEC: Supported
✓ PyNvCodec: Installed
```

## Configuration

### 1. Model Weights

Place your trained RF-DETR model:
```bash
cp /path/to/your/model.pth models/best_rf-detr.pth
```

### 2. Stream Configuration

Edit `configs/stream_config.yml`:

```yaml
stream:
  input_url: "your_video.mp4"  # or RTMP URL, or YouTube
  target_fps: 30
  mjpeg_port: 8554

output:
  width: 3840    # Output resolution (4K)
  height: 1440

processing:
  detection_width: 1920   # Inference resolution (lower = faster)
  detection_height: 1080
```

**Performance tuning:**
- RTX 5090 / L4: `detection: 1920x1080` (30 FPS)
- A10G: `detection: 1280x720` (30 FPS)
- Lower-end GPUs: `detection: 960x540` (25-30 FPS)

### 3. Model Configuration

Edit `configs/model_config.yml`:

```yaml
model:
  path: "models/best_rf-detr.pth"
  confidence: 0.25          # Detection threshold
  half_precision: true      # FP16 for 2× speedup
  imgsz: 640               # Model input size

tracking:
  max_lost_frames: 10      # Prediction-only tolerance
  min_confidence: 0.10     # Minimum track confidence
  adaptive_noise: true     # Kalman filter chaos detection
```

## Starting the Pipeline

### Method 1: Managed Service (Recommended)

```bash
# Start in tmux
./start_stream.sh

# Stream will be available at: http://localhost:8554/stream.mjpg
# Attach to session: tmux attach -t football-stream
# Detach: Ctrl+B then D
```

### Method 2: Direct Execution

```bash
source rf-detr-venv-310/bin/activate
python run_mjpeg_stream.py 2>&1 | tee logs/stream.log
```

## Viewing the Stream

### Local Viewing
```bash
vlc http://localhost:8554/stream.mjpg
# or
mpv http://localhost:8554/stream.mjpg
```

### Remote Viewing (SSH Tunnel)

On your local machine:
```bash
ssh -N -L 8554:localhost:8554 user@remote-server
```

Then open: `http://localhost:8554/stream.mjpg`

### Bandwidth-Optimized Monitoring

```bash
# Create a lower-quality proxy (1280x480 @ 10fps)
ffmpeg -i http://localhost:8554/stream.mjpg \
       -vf scale=1280:-1 -r 10 -qscale:v 8 \
       -f mjpeg tcp://127.0.0.1:9554?listen=1
       
# Tunnel port 9554 instead
ssh -N -L 9554:localhost:9554 user@remote-server
```

## Recording

### Zero-Reencode (No FPS Drop)

```bash
# Record MJPEG directly to segmented files
mkdir -p clips
ffmpeg -thread_queue_size 2048 \
       -i http://localhost:8554/stream.mjpg \
       -c copy -f segment -segment_time 60 clips/clip_%03d.mjpeg

# Convert to MP4 offline later
ffmpeg -i clips/clip_000.mjpeg -c:v libx264 -preset faster -crf 20 output.mp4
```

### Hardware Encode (NVENC)

```bash
ffmpeg -thread_queue_size 512 \
       -i http://localhost:8554/stream.mjpeg \
       -c:v h264_nvenc -preset p5 -cq 19 -b_ref_mode middle output.mp4
```

## YouTube as Input Source

### Option A: Direct URL (expires after ~6 hours)

```bash
# Resolve media URL
yt-dlp -g -f best https://www.youtube.com/watch?v=VIDEO_ID

# Copy URL to configs/stream_config.yml
stream:
  input_url: "https://rr3---sn-..."
```

### Option B: Local Restream (recommended for live)

```bash
# Terminal 1: Restream YouTube to UDP
yt-dlp -f best -o - https://www.youtube.com/watch?v=VIDEO_ID \
  | ffmpeg -re -i - -c copy -f mpegts udp://127.0.0.1:9000

# Terminal 2: Configure and start pipeline
# Edit configs/stream_config.yml:
#   input_url: "udp://127.0.0.1:9000"
./start_stream.sh
```

## Troubleshooting

### GPU Not Detected
```bash
nvidia-smi
# If fails: sudo apt install nvidia-driver-550
```

### "unsupported device (2)" Error
**Cause:** Container lacks video capability

**Fix:**
```bash
docker run --gpus '"capabilities=compute,graphics,utility,video"' ...
```

### "Driver/library version mismatch"
**Cause:** Misaligned NVIDIA packages

**Fix:**
```bash
DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
sudo apt install nvidia-utils-${DRIVER_VER} \
                 libnvidia-encode-${DRIVER_VER} \
                 libnvidia-decode-${DRIVER_VER}
```

### CUDA Out of Memory
```bash
# Reduce inference resolution in configs/stream_config.yml
processing:
  detection_width: 1280
  detection_height: 720
```

### Low FPS (<20)
1. Check GPU utilization: `nvidia-smi dmon -s u -c 60`
2. Lower detection resolution
3. Verify FP16 enabled: `half_precision: true` in model_config.yml
4. Ensure `optimize_for_inference` succeeded (check logs)

### Stream Drops/Buffering
```bash
# Increase FFmpeg thread queue
ffmpeg -thread_queue_size 4096 ...

# Check bandwidth
iftop -i eth0
```

## Monitoring

### Real-Time Logs
```bash
# If using tmux
tmux attach -t football-stream

# If using systemd
journalctl -u football-stream -f
```

### Key Metrics
```
[STREAM] Frame 1000 | FPS: 32.1 | Inf: 19.2ms | Track: ACTIVE | Zoom: 1.65x
```

- **FPS**: End-to-end performance
- **Inf**: RF-DETR inference time (target: <25ms)
- **Track**: Ball tracking state
- **Zoom**: Current camera zoom factor

### Health Check
```bash
curl -I http://localhost:8554/stream.mjpg
# Expect: HTTP/1.0 200 OK
```

## Stopping the Pipeline

```bash
./stop_stream.sh
# or manually kill tmux session
tmux kill-session -t football-stream
```

## Production Best Practices

1. **Use systemd for auto-restart**
   ```bash
   sudo cp deployment/football-stream.service /etc/systemd/system/
   sudo systemctl enable football-stream
   sudo systemctl start football-stream
   ```

2. **Log rotation**
   ```bash
   sudo cp deployment/logrotate.conf /etc/logrotate.d/football-stream
   ```

3. **Firewall configuration** (if exposing publicly)
   ```bash
   # Only allow specific IPs
   sudo ufw allow from 203.0.113.0/24 to any port 8554
   ```

4. **Resource monitoring**
   ```bash
   # Set up Prometheus exporters
   nvidia_gpu_prometheus_exporter --port=9101
   ```

5. **Backup model weights**
   ```bash
   rsync -avz models/ backup-server:/backups/rf-detr/models/
   ```

## AWS Deployment

For G6 instances (NVIDIA L4):

```bash
# Launch instance
aws ec2 run-instances \
  --image-id ami-0c7217cdde317cfec \
  --instance-type g6.xlarge \
  --key-name your-key \
  --security-group-ids sg-xxx \
  --subnet-id subnet-xxx

# SSH in and run setup
ssh ubuntu@<instance-ip>
git clone <repo>
cd Football-Detection
./setup_production.sh
```

See `guides/AWS_DEPLOYMENT.md` for detailed Terraform setup.

## Support

- **Documentation:** [README.md](README.md)
- **Troubleshooting:** [instruccion.txt](instruccion.txt)
- **Issues:** https://github.com/yourusername/Football-Detection/issues

---

**Last Updated:** November 2024  
**Version:** 3.0.0 (MJPEG Streaming)
