# Changelog

All notable changes to RF-DETR Football Detection will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2024-11-16

### Added - Production Release

#### Installation & Setup
- **Automated setup scripts** for Linux (`setup_production.sh`) and Windows (`setup_production_windows.ps1`)
- Complete GPU diagnostics tools (`diagnose_gpu.py`, `diagnose_gpu_windows.ps1`)
- Service management scripts (`start_stream.sh/bat`, `stop_stream.sh/ps1`)
- Docker support with `Dockerfile` and `docker-compose.yml`
- Systemd service file for Linux auto-start (`deployment/football-stream.service`)

#### Streaming & Performance
- **MJPEG streaming server** on localhost:8554 for low-latency monitoring
- Multi-resolution workflow: decode 4K → detect at 1080p/720p → output 4K
- Zero-copy GPU video acceleration with PyNvCodec (NVDEC/NVENC)
- Automatic CPU fallback when PyNvCodec unavailable (Windows)
- YouTube live stream support via yt-dlp integration
- SSH tunnel support for remote monitoring

#### Detection & Tracking
- RF-DETR Medium with `optimize_for_inference` (1.5-2× speedup)
- Extended Kalman Filter with 6-DOF state (position, velocity, acceleration)
- Mahalanobis gating for outlier rejection
- Adaptive process noise during chaos mode
- Jump rejection (>120px) with ultra-stability smoothing
- Automatic chaos detection and suppression

#### Virtual Camera
- One-Euro filter for smooth position tracking
- PID controller for responsive positioning
- Trajectory prediction (5-step lookahead)
- Dynamic zoom with spring-damper model
- Velocity-dependent smoothing parameters
- Dead zone configuration for stability

#### Documentation
- Comprehensive README with detailed algorithm mathematics
- Windows setup guide (`WINDOWS_SETUP.md`)
- Production quick start guide (`PRODUCTION_QUICKSTART.md`)
- AWS deployment guide in `guides/AWS_DEPLOYMENT.md`
- Complete production playbook (`instruccion.txt`)
- Example configuration files with inline comments

### Fixed

#### Installation Issues
- **CMake CMP0148 policy warning** when building PyNvCodec
- **Driver/library version mismatch** errors on Linux
- **"unsupported device (2)"** error in Docker containers
- NVDEC/NVENC capability detection and validation
- LD_LIBRARY_PATH configuration for NVIDIA libraries
- Python 3.10 compatibility across platforms

#### Runtime Issues
- Endless 'video ended' loop in GPU pipeline (PyNvCodec usage)
- Video decode/encode capability detection in containers
- Proper error handling when CUDA unavailable
- Fallback to CPU video processing on Windows

#### Pipeline Issues
- Frame drop during recording (zero-reencode option)
- FPS instability with high-resolution inputs
- Detection coordinate scaling with multi-resolution workflow
- Virtual camera jitter during erratic detector behavior

### Changed

- **Primary entry point** now `run_mjpeg_stream.py` (instead of `main.py`)
- **Streaming architecture** from RTMP to MJPEG for lower latency
- **Configuration** split into `model_config.yml` and `stream_config.yml`
- **Detection resolution** separated from output resolution
- **Recording workflow** using FFmpeg from MJPEG endpoint

### Performance

#### Benchmarks (AWS L4 GPU)
- **Input:** 4K (3840×2160)
- **Detection:** 1920×1080
- **Output:** 3840×1440
- **FPS:** 25-30
- **Inference:** 18-22ms
- **GPU Utilization:** 75-90%

#### Benchmarks (RTX 5090)
- **Input:** 4K (3840×2160)
- **Detection:** 1920×1080
- **Output:** 3840×1440
- **FPS:** 30+
- **Inference:** 15-18ms
- **GPU Utilization:** 70-85%

#### Benchmarks (Windows - RTX 4080)
- **Input:** 4K (3840×2160)
- **Detection:** 1280×720 (CPU decode)
- **Output:** 3840×1440
- **FPS:** 22-28
- **Inference:** 19-23ms
- **Note:** -10-15% vs Linux due to CPU video decode

### Deployment

#### Supported Platforms
- Ubuntu 22.04/24.04 (recommended for production)
- Windows 10/11 (client testing)
- AWS EC2 G6 instances (L4 GPU)
- Docker with GPU support

#### Requirements
- Python 3.10
- CUDA 11.8+
- NVIDIA Driver ≥ 550
- 12 GB+ VRAM for 4K workflow

### Known Limitations

- **PyNvCodec** not available on Windows (uses CPU video I/O)
- **ONNX/TensorRT backends** not fully tested in v3.0
- **Multi-GPU support** not implemented
- **Batch processing** (`main.py`) considered legacy

### Migration from v2.x

1. Update installation:
   ```bash
   # Linux
   ./setup_production.sh
   
   # Windows
   .\setup_production_windows.ps1
   ```

2. Update configuration files to new format:
   - Split into `model_config.yml` and `stream_config.yml`
   - Add `processing` section for detection resolution
   - Update `camera` settings with new parameters

3. Change entry point:
   ```bash
   # Old
   python main.py stream --input video.mp4
   
   # New
   python run_mjpeg_stream.py
   ```

4. Update monitoring:
   - Stream available at `http://localhost:8554/stream.mjpg`
   - Use VLC or FFmpeg to view/record

### Security

- MJPEG server binds to localhost only (0.0.0.0 requires explicit change)
- Model weights not included in repository (user must provide)
- No hardcoded credentials or API keys
- Docker containers run with minimal privileges

### Credits

- RF-DETR by Roboflow
- Supervision library for tracking utilities
- One-Euro Filter research by Géry Casiez et al.
- PyNvCodec by NVIDIA

---

## [2.x] - Previous Versions

Legacy versions with RTMP streaming and older pipeline architecture.
See git history for details.

---

**Current Version:** 3.0.0 (MJPEG Streaming)
**Status:** Production Ready
**Last Updated:** November 16, 2025
