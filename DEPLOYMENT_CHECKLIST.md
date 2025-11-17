# RF-DETR Football Detection - Deployment Checklist

Complete checklist for deploying to production. Use this before delivering to client or going live.

## Pre-Deployment

### System Requirements
- [ ] GPU verified: NVIDIA with CUDA 11.8+ support
- [ ] Driver version ≥ 550 installed
- [ ] NVDEC/NVENC capability confirmed (run `nvidia-smi -q | grep -i "Video Encode"`)
- [ ] Minimum 12 GB VRAM available
- [ ] Ubuntu 22.04/24.04 or Windows 10/11
- [ ] Python 3.10 installed
- [ ] 100 GB+ disk space available

### Installation
- [ ] Setup script executed successfully
  - Linux: `./setup_production.sh`
  - Windows: `.\setup_production_windows.ps1`
- [ ] Virtual environment created: `rf-detr-venv-310/`
- [ ] All dependencies installed (check `pip list`)
- [ ] GPU diagnostics passed: `python diagnose_gpu.py` (Linux) or `.\diagnose_gpu_windows.ps1` (Windows)
- [ ] PyTorch CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`

### Model & Configuration
- [ ] Model weights placed in: `models/best_rf-detr.pth`
- [ ] Model file size reasonable (typically 50-200 MB)
- [ ] `configs/model_config.yml` customized
  - [ ] Confidence threshold adjusted (default 0.25)
  - [ ] Half precision enabled for modern GPUs
  - [ ] Device set to "cuda"
- [ ] `configs/stream_config.yml` customized
  - [ ] Input source URL/path configured
  - [ ] Detection resolution optimized for GPU
  - [ ] Output resolution matches requirements
  - [ ] MJPEG port configured (default 8554)

### Testing (Development)
- [ ] Test run completed on sample video
- [ ] FPS meets target (≥25 for production)
- [ ] Ball detection accuracy acceptable
- [ ] Tracking smooth without jitter
- [ ] No memory leaks (run for >1 hour)
- [ ] GPU utilization 70-90% (optimal range)
- [ ] No CUDA errors in logs

## Production Deployment

### Environment Setup
- [ ] Production server/VM provisioned
  - AWS: G6.xlarge (L4) or better
  - On-premise: RTX 5090, L40S, or similar
- [ ] Firewall configured
  - [ ] Port 8554 open (if remote access needed)
  - [ ] SSH port 22 open for management
- [ ] SSL/TLS certificates installed (if exposing publicly)
- [ ] Backup storage configured for clips

### Service Configuration
- [ ] Service scripts installed
  - Linux: `start_stream.sh`, `stop_stream.sh`
  - Windows: `start_stream.bat`, `stop_stream.ps1`
- [ ] Systemd service configured (Linux only)
  ```bash
  sudo cp deployment/football-stream.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable football-stream
  ```
- [ ] Auto-start on boot tested
- [ ] Log rotation configured
  ```bash
  sudo cp deployment/logrotate.conf /etc/logrotate.d/football-stream
  ```

### Docker Deployment (Optional)
- [ ] Docker and nvidia-docker2 installed
- [ ] `docker-compose.yml` configured
- [ ] Volumes mapped correctly
  - [ ] `./models` → `/app/models`
  - [ ] `./configs` → `/app/configs`
  - [ ] `./clips` → `/app/clips`
- [ ] GPU passthrough working
  ```bash
  docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
  ```
- [ ] Container starts successfully
  ```bash
  docker-compose up -d
  ```
- [ ] Health check passing
  ```bash
  docker-compose ps
  ```

### Monitoring & Alerting
- [ ] Monitoring solution configured
  - [ ] Prometheus + Grafana (recommended)
  - [ ] CloudWatch (AWS)
  - [ ] Azure Monitor (Azure)
- [ ] Key metrics tracked:
  - [ ] FPS (end-to-end)
  - [ ] Inference time (ms)
  - [ ] GPU utilization (%)
  - [ ] Memory usage
  - [ ] Stream uptime
- [ ] Alerts configured:
  - [ ] FPS < 20 for >1 minute
  - [ ] GPU memory > 90%
  - [ ] Stream down for >30 seconds
  - [ ] CUDA errors
- [ ] Log aggregation configured
  - [ ] Logs sent to centralized system
  - [ ] Retention policy set (30 days recommended)

### Security
- [ ] MJPEG server bound to localhost only (unless public access required)
- [ ] SSH key-based authentication configured
- [ ] Root login disabled
- [ ] Firewall rules reviewed
- [ ] Model weights not committed to git
- [ ] `.gitignore` includes:
  - [ ] `models/*.pth`
  - [ ] `*.log`
  - [ ] `clips/`
  - [ ] `.env`
- [ ] Secrets management configured
  - [ ] API keys in environment variables
  - [ ] Credentials in secure vault

### Performance Validation
- [ ] Benchmark tests run for ≥1 hour
- [ ] FPS stable at target (25-30+)
- [ ] Inference time < 30ms average
- [ ] GPU temperature < 80°C under load
- [ ] No memory leaks detected
- [ ] Recording works without frame drops
- [ ] Stream viewable with <500ms latency

## Go-Live

### Final Checks
- [ ] Production input source configured
- [ ] Recording enabled (if required)
- [ ] Backup/redundancy tested
- [ ] Disaster recovery plan documented
- [ ] Client trained on:
  - [ ] Starting/stopping stream
  - [ ] Viewing stream
  - [ ] Recording stream
  - [ ] Basic troubleshooting
- [ ] Support contact information provided

### Launch
- [ ] Service started:
  ```bash
  # Linux
  sudo systemctl start football-stream
  # or
  ./start_stream.sh
  
  # Windows
  .\start_stream.bat
  ```
- [ ] Stream accessible:
  ```
  http://localhost:8554/stream.mjpg
  ```
- [ ] Metrics dashboard reviewed
- [ ] Initial recording captured and verified
- [ ] No errors in first 15 minutes

### Post-Launch (First 24 Hours)
- [ ] Stream uptime > 99%
- [ ] FPS remains stable
- [ ] No unexpected crashes
- [ ] Logs reviewed for warnings/errors
- [ ] Client feedback collected
- [ ] Performance matches expectations

## Ongoing Maintenance

### Daily
- [ ] Check stream uptime
- [ ] Review error logs
- [ ] Verify recording quality

### Weekly
- [ ] Analyze performance trends
- [ ] Check disk space usage
- [ ] Review GPU health metrics
- [ ] Clean old logs and clips

### Monthly
- [ ] Update dependencies (security patches)
- [ ] Backup model weights
- [ ] Review and optimize configuration
- [ ] Performance audit

### Quarterly
- [ ] Full system update
- [ ] GPU driver update (if stable)
- [ ] Review disaster recovery plan
- [ ] Client satisfaction survey

## Troubleshooting References

### Quick Fixes
- **Stream down:** Restart service (`sudo systemctl restart football-stream`)
- **Low FPS:** Reduce detection resolution in `configs/stream_config.yml`
- **GPU OOM:** Enable half precision, reduce batch size
- **CUDA errors:** Run diagnostics (`python diagnose_gpu.py`)

### Documentation
- README.md - Full documentation
- PRODUCTION_QUICKSTART.md - Quick reference
- WINDOWS_SETUP.md - Windows-specific guide
- instruccion.txt - Deployment playbook
- guides/AWS_DEPLOYMENT.md - AWS setup
- CHANGELOG.md - Version history

### Support Contacts
- **GitHub Issues:** https://github.com/yourusername/Football-Detection/issues
- **Email:** support@example.com
- **Documentation:** https://github.com/yourusername/Football-Detection

## Sign-Off

### Pre-Production
- [ ] Technical lead approved
- [ ] QA testing passed
- [ ] Security review completed
- [ ] Client preview approved

### Production
- [ ] Deployed by: _________________ Date: _________________
- [ ] Verified by: _________________ Date: _________________
- [ ] Client accepted: _________________ Date: _________________

---

**Version:** 3.0.0  
**Last Updated:** November 16, 2024  
**Status:** Production Ready
