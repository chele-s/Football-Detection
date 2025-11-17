# Windows Setup Guide - RF-DETR Football Detection

Complete guide for setting up the RF-DETR pipeline on Windows for client testing before AWS deployment.

## Prerequisites

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with driver ≥ 522.0
- **Administrator privileges**
- **20 GB free disk space**

## Quick Install

### 1. Open PowerShell as Administrator

Right-click Start menu → **Windows PowerShell (Admin)**

### 2. Allow Script Execution

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
```

### 3. Clone Repository

```powershell
git clone https://github.com/yourusername/Football-Detection.git
cd Football-Detection
```

### 4. Run Setup Script

```powershell
.\setup_production_windows.ps1
```

**Duration:** 15-25 minutes (includes downloading and installing dependencies)

### 5. Verify Installation

```powershell
.\diagnose_gpu_windows.ps1
```

## What Gets Installed

### System Components

- **Chocolatey** - Windows package manager
- **Git** - Version control
- **Python 3.10** - Installed at `C:\Python310\`
- **FFmpeg** - Video processing
- **Visual Studio Build Tools** - For compiling Python packages

### Python Environment

- Virtual environment: `rf-detr-venv-310\`
- **PyTorch** with CUDA 11.8 support
- **RF-DETR** transformer model
- **OpenCV** for video I/O
- **Supervision** for tracking
- All dependencies from `requirements.txt`

### Scripts Created

- `start_stream.ps1` - PowerShell start script
- `start_stream.bat` - Batch file for easy start
- `stop_stream.ps1` - Stop script
- `diagnose_gpu_windows.ps1` - System diagnostics

## Configuration

### 1. Model Weights

Copy your trained RF-DETR model:

```powershell
Copy-Item "C:\path\to\your\model.pth" "models\best_rf-detr.pth"
```

### 2. Stream Configuration

Edit `configs\stream_config.yml`:

```yaml
stream:
  input_url: "C:\\Videos\\football_4k.mp4"  # Use double backslashes
  target_fps: 30
  mjpeg_port: 8554

output:
  width: 3840
  height: 1440

processing:
  detection_width: 1280   # Lower for Windows (no PyNvCodec)
  detection_height: 720
```

**Windows-specific settings:**
- Use `1280x720` or `960x540` for detection (CPU video decode)
- Full paths with double backslashes: `C:\\path\\to\\file.mp4`
- RTMP not recommended on Windows (use file sources)

### 3. Performance Tuning

Edit `configs\model_config.yml`:

```yaml
model:
  path: "models\\best_rf-detr.pth"
  confidence: 0.25
  half_precision: true
  imgsz: 640

tracking:
  max_lost_frames: 10
  min_confidence: 0.10
```

## Running the Pipeline

### Method 1: Double-Click (Easiest)

Simply double-click `start_stream.bat`

### Method 2: PowerShell

```powershell
.\start_stream.ps1
```

### Method 3: Manual Activation

```powershell
# Activate virtual environment
.\rf-detr-venv-310\Scripts\Activate.ps1

# Run pipeline
python run_mjpeg_stream.py
```

## Viewing the Stream

### Option 1: VLC Media Player

1. Open VLC
2. Media → Open Network Stream
3. Enter: `http://localhost:8554/stream.mjpg`
4. Click Play

### Option 2: Web Browser

Open Chrome/Edge and navigate to:
```
http://localhost:8554/stream.mjpg
```

**Note:** May have higher latency than VLC.

### Option 3: FFplay

```powershell
ffplay http://localhost:8554/stream.mjpg
```

## Recording

### Zero-Reencode (Best Quality)

```powershell
# In a new PowerShell window
ffmpeg -i http://localhost:8554/stream.mjpg -c copy output.mjpeg

# Convert to MP4 later
ffmpeg -i output.mjpeg -c:v libx264 -preset faster -crf 20 output.mp4
```

### Direct to MP4

```powershell
ffmpeg -i http://localhost:8554/stream.mjpg `
       -c:v libx264 -preset superfast -crf 24 output.mp4
```

### Hardware Encode (if NVENC available)

```powershell
ffmpeg -i http://localhost:8554/stream.mjpg `
       -c:v h264_nvenc -preset p5 -cq 19 output.mp4
```

## Troubleshooting

### GPU Not Detected

**Check NVIDIA driver:**
```powershell
nvidia-smi
```

**If fails:**
1. Download latest driver: https://www.nvidia.com/Download/index.aspx
2. Install and reboot
3. Run `.\setup_production_windows.ps1` again

### CUDA Not Available in PyTorch

```powershell
# Activate environment
.\rf-detr-venv-310\Scripts\Activate.ps1

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Low FPS (<15)

**Reduce inference resolution** in `configs\stream_config.yml`:
```yaml
processing:
  detection_width: 960
  detection_height: 540
```

**Or reduce output resolution:**
```yaml
output:
  width: 1920
  height: 1080
```

### "Access Denied" Errors

Run PowerShell as Administrator

### Firewall Blocks Stream

Add firewall rule:
```powershell
New-NetFirewallRule -DisplayName "RF-DETR Stream" `
                    -Direction Inbound `
                    -LocalPort 8554 `
                    -Protocol TCP `
                    -Action Allow
```

### Visual Studio Build Tools Error

Manually install:
```powershell
choco install visualstudio2022buildtools `
  --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
```

### Python Package Build Failures

Install Microsoft C++ Build Tools:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Stream Disconnects Frequently

**Increase FFmpeg buffer:**
```powershell
ffmpeg -thread_queue_size 4096 -i http://localhost:8554/stream.mjpg ...
```

## Performance Optimization

### Expected FPS on Windows

| GPU | Detection Res | Output Res | FPS |
|-----|--------------|------------|-----|
| RTX 4090 | 1280x720 | 3840x1440 | 25-30 |
| RTX 4080 | 1280x720 | 3840x1440 | 22-28 |
| RTX 3080 | 960x540 | 1920x1080 | 25-30 |
| RTX 3070 | 960x540 | 1920x1080 | 20-25 |

**Note:** Windows performance is ~10-15% lower than Linux due to CPU video decode.

### Optimize for Your GPU

**High-end (RTX 4090, 4080):**
```yaml
processing:
  detection_width: 1280
  detection_height: 720
```

**Mid-range (RTX 3080, 3070):**
```yaml
processing:
  detection_width: 960
  detection_height: 540
```

**Entry-level (RTX 3060, 2060):**
```yaml
processing:
  detection_width: 640
  detection_height: 360
```

## Monitoring

### Check Logs

```powershell
# View latest log
Get-Content -Path "logs\*.log" -Tail 50
```

### Real-Time Metrics

Look for console output:
```
[STREAM] Frame 1000 | FPS: 28.3 | Inf: 21.5ms | Track: ACTIVE | Zoom: 1.45x
```

- **FPS:** Overall performance
- **Inf:** Inference time (target <30ms)
- **Track:** Ball tracking state
- **Zoom:** Current camera zoom

### GPU Utilization

```powershell
# Monitor GPU usage
nvidia-smi dmon -s u -c 60
```

Target: 70-95% GPU utilization for good performance

## Stopping the Pipeline

### Method 1: Keyboard

Press **Ctrl+C** in the PowerShell window

### Method 2: Script

```powershell
.\stop_stream.ps1
```

### Method 3: Task Manager

1. Open Task Manager (Ctrl+Shift+Esc)
2. Find `python.exe` running `run_mjpeg_stream.py`
3. Right-click → End Task

## Differences from Linux/AWS

| Feature | Windows | Linux (AWS) |
|---------|---------|-------------|
| Video Decode | CPU (OpenCV) | GPU (NVDEC) |
| Video Encode | N/A (MJPEG) | GPU (NVENC) optional |
| PyNvCodec | Not available | Available |
| Performance | -10-15% | Optimal |
| Use Case | Client testing | Production |

**Recommendation:** Use Windows for development and testing, deploy to AWS Linux for production.

## Testing Before AWS Deployment

### Checklist

- [ ] Stream runs at target FPS (20-30)
- [ ] Ball detection is accurate
- [ ] Tracking is smooth (no jittering)
- [ ] Recording works without drops
- [ ] Configuration is finalized

### Configuration Transfer to AWS

1. Export your configs:
```powershell
# Create deployment package
Compress-Archive -Path configs\*,models\best_rf-detr.pth `
                 -DestinationPath aws_deployment.zip
```

2. Upload to AWS instance and extract

3. Run Linux setup:
```bash
./setup_production.sh
```

4. Copy configs from zip

## Support

**Diagnostics:**
```powershell
.\diagnose_gpu_windows.ps1
```

**Common Issues:**
- See `PRODUCTION_QUICKSTART.md` for general troubleshooting
- See `README.md` for algorithm details
- Check logs in `logs\` folder

**Need Help?**
- GitHub Issues: https://github.com/chele-s/Football-Detection/issues
- Include output from `diagnose_gpu_windows.ps1`

---

**Last Updated:** November 2025  
**Version:** 3.0.0 (Windows Edition)
