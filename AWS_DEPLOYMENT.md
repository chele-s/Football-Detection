# AWS Deployment Guide - G6 Instances

Complete guide for deploying the Football Detection & Tracking System on AWS G6 instances with NVIDIA L4 GPUs.

## Table of Contents

1. [Instance Selection](#instance-selection)
2. [Initial Setup](#initial-setup)
3. [Environment Configuration](#environment-configuration)
4. [Application Deployment](#application-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring & Logging](#monitoring--logging)
7. [Auto-Scaling](#auto-scaling)
8. [Troubleshooting](#troubleshooting)

---

## Instance Selection

### Recommended Instance Types

| Instance Type | vCPUs | GPU | VRAM | RAM | Network | Cost/hr | Use Case |
|--------------|-------|-----|------|-----|---------|---------|----------|
| **g6.xlarge** | 4 | 1x L4 | 24GB | 16GB | 10 Gbps | ~$1.20 | Single stream (1080p) |
| **g6.2xlarge** | 8 | 1x L4 | 24GB | 32GB | 12 Gbps | ~$1.60 | 2-3 concurrent streams |
| **g6.4xlarge** | 16 | 1x L4 | 24GB | 64GB | 25 Gbps | ~$2.40 | 4-6 concurrent streams |

**Recommended for production:** `g6.xlarge` for single 1080p stream at 30 FPS.

### Expected Performance (g6.xlarge)

```
Input: 1920x1080 @ 30 FPS
Output: 1920x1080 @ 30 FPS

Component Breakdown:
- RF-DETR Inference: 18-22ms
- Tracking: 1-2ms
- Camera Processing: 1-2ms
- Rendering: 8-10ms
- Total: 28-36ms (30-35 FPS sustained)

VRAM Usage: 3.2GB / 24GB
RAM Usage: 2.8GB / 16GB
CPU Usage: 15-25%
```

---

## Initial Setup

### 1. Launch Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type g6.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --block-device-mappings DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3} \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=football-tracker-prod}]'
```

### 2. Security Group Configuration

Required inbound rules:

```
SSH         TCP  22     Your IP         (Management)
Custom TCP  TCP  1935   0.0.0.0/0       (RTMP input)
Custom TCP  TCP  1936   0.0.0.0/0       (RTMP output)
HTTPS       TCP  443    0.0.0.0/0       (API/Monitoring)
```

### 3. Connect to Instance

```bash
ssh -i your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com
```

---

## Environment Configuration

### System Updates

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget vim
```

### NVIDIA Driver Installation

```bash
# Verify GPU presence
lspci | grep -i nvidia

# Install NVIDIA driver
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot

# Verify installation (after reboot)
nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA L4           Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   45C    P0    25W / 72W |      0MiB / 23034MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### CUDA Toolkit Installation

```bash
# Install CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

### Python Environment

```bash
# Install Python 3.10
sudo apt install -y python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv /opt/football-tracker/venv
source /opt/football-tracker/venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### FFmpeg with NVENC

```bash
# Install FFmpeg with NVIDIA acceleration
sudo apt install -y ffmpeg

# Verify NVENC support
ffmpeg -codecs | grep nvenc
```

Expected output:
```
DEV.LS h264     H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (decoders: h264 h264_cuvid ) (encoders: h264_nvenc nvenc nvenc_h264 )
```

---

## Application Deployment

### 1. Clone Repository

```bash
cd /opt
sudo git clone https://github.com/chele-s/Football-Detection.git football-tracker
sudo chown -R ubuntu:ubuntu football-tracker
cd football-tracker
```

### 2. Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True, Device: NVIDIA L4
```

### 3. Download Model

```bash
mkdir -p models
cd models

# Option 1: Use pre-trained RF-DETR
# (Model will auto-download on first run)

# Option 2: Use custom trained model
aws s3 cp s3://your-bucket/models/best_rf-detr.pth ./best_rf-detr.pth
```

### 4. Configure for Production

Create production config: `configs/production.yml`

```yaml
model:
  path: "models/best_rf-detr.pth"
  confidence: 0.25
  iou_threshold: 0.45
  device: "cuda"
  half_precision: true
  imgsz: 640
  warmup_iterations: 5

tracking:
  max_lost_frames: 10
  min_confidence: 0.3
  iou_threshold: 0.3
  adaptive_noise: true

output:
  width: 1920
  height: 1080

stream:
  target_fps: 30
  bitrate: "5000k"
  preset: "fast"
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

### 5. Create Systemd Service

Create `/etc/systemd/system/football-tracker.service`:

```ini
[Unit]
Description=Football Detection & Tracking Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/football-tracker
Environment="PATH=/opt/football-tracker/venv/bin:/usr/local/cuda-11.8/bin"
Environment="LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64"
Environment="CUDA_VISIBLE_DEVICES=0"
ExecStart=/opt/football-tracker/venv/bin/python main.py stream \
  --model-config configs/production.yml \
  --input rtmp://input-server/live/stream \
  --output rtmp://output-server/live/stream \
  --device cuda
Restart=always
RestartSec=10
StandardOutput=append:/var/log/football-tracker/output.log
StandardError=append:/var/log/football-tracker/error.log

[Install]
WantedBy=multi-user.target
```

### 6. Start Service

```bash
# Create log directory
sudo mkdir -p /var/log/football-tracker
sudo chown ubuntu:ubuntu /var/log/football-tracker

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable football-tracker
sudo systemctl start football-tracker

# Check status
sudo systemctl status football-tracker
```

---

## Performance Tuning

### GPU Optimization

```bash
# Set GPU persistence mode
sudo nvidia-smi -pm 1

# Set power limit (optional, for consistency)
sudo nvidia-smi -pl 72

# Lock GPU clocks (prevents throttling)
sudo nvidia-smi -lgc 2100
```

### System Optimization

```bash
# Increase file descriptor limits
sudo tee -a /etc/security/limits.conf <<EOF
ubuntu soft nofile 65536
ubuntu hard nofile 65536
EOF

# Optimize network for streaming
sudo tee -a /etc/sysctl.conf <<EOF
net.core.rmem_max=134217728
net.core.wmem_max=134217728
net.ipv4.tcp_rmem=4096 87380 67108864
net.ipv4.tcp_wmem=4096 65536 67108864
net.core.netdev_max_backlog=5000
EOF

sudo sysctl -p
```

### FFmpeg NVENC Configuration

For maximum performance, modify stream config:

```yaml
stream:
  preset: "p4"  # NVENC preset (p1-p7, p4 is balanced)
  bitrate: "5000k"
  codec: "h264_nvenc"
  rc_mode: "vbr"  # Variable bitrate
```

---

## Monitoring & Logging

### CloudWatch Integration

Install CloudWatch agent:

```bash
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Configure agent
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/football-tracker/deployment/cloudwatch-config.json
```

Create `deployment/cloudwatch-config.json`:

```json
{
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/football-tracker/output.log",
            "log_group_name": "/aws/ec2/football-tracker",
            "log_stream_name": "{instance_id}/output"
          },
          {
            "file_path": "/var/log/football-tracker/error.log",
            "log_group_name": "/aws/ec2/football-tracker",
            "log_stream_name": "{instance_id}/error"
          }
        ]
      }
    }
  },
  "metrics": {
    "namespace": "FootballTracker",
    "metrics_collected": {
      "mem": {
        "measurement": [
          {"name": "mem_used_percent", "unit": "Percent"}
        ],
        "metrics_collection_interval": 60
      },
      "disk": {
        "measurement": [
          {"name": "used_percent", "unit": "Percent"}
        ],
        "metrics_collection_interval": 60
      }
    }
  }
}
```

### GPU Monitoring Script

Create `scripts/monitor_gpu.sh`:

```bash
#!/bin/bash
while true; do
  nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv,noheader >> /var/log/football-tracker/gpu_metrics.csv
  sleep 10
done
```

Run in background:

```bash
chmod +x scripts/monitor_gpu.sh
nohup ./scripts/monitor_gpu.sh &
```

### Health Check Endpoint

Add to your application (optional):

```python
# app/health_check.py
from flask import Flask, jsonify
import torch

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'gpu_available': torch.cuda.is_available(),
        'gpu_memory_allocated': torch.cuda.memory_allocated() / 1e9,
        'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## Auto-Scaling

### Launch Template

Create launch template for auto-scaling group:

```bash
aws ec2 create-launch-template \
  --launch-template-name football-tracker-template \
  --version-description "Production v1" \
  --launch-template-data '{
    "ImageId": "ami-xxxxxxxxx",
    "InstanceType": "g6.xlarge",
    "KeyName": "your-key",
    "SecurityGroupIds": ["sg-xxxxxxxxx"],
    "UserData": "'$(base64 -w 0 deployment/user-data.sh)'",
    "TagSpecifications": [{
      "ResourceType": "instance",
      "Tags": [{"Key": "Name", "Value": "football-tracker-asg"}]
    }]
  }'
```

### Auto-Scaling Group

```bash
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name football-tracker-asg \
  --launch-template LaunchTemplateName=football-tracker-template \
  --min-size 1 \
  --max-size 5 \
  --desired-capacity 2 \
  --vpc-zone-identifier "subnet-xxxx,subnet-yyyy" \
  --health-check-type ELB \
  --health-check-grace-period 300
```

### Scaling Policies

```bash
# Scale up when GPU utilization > 80%
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name football-tracker-asg \
  --policy-name scale-up-gpu \
  --scaling-adjustment 1 \
  --adjustment-type ChangeInCapacity \
  --cooldown 300

# Scale down when GPU utilization < 20%
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name football-tracker-asg \
  --policy-name scale-down-gpu \
  --scaling-adjustment -1 \
  --adjustment-type ChangeInCapacity \
  --cooldown 300
```

---

## Troubleshooting

### Issue: GPU Not Detected

```bash
# Check driver installation
nvidia-smi

# Reinstall if needed
sudo apt purge nvidia-* -y
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Issue: Out of Memory

```bash
# Check VRAM usage
nvidia-smi

# Reduce batch size or resolution in config
# Or upgrade to g6.2xlarge
```

### Issue: Low FPS

```bash
# Profile application
python -m cProfile -o profile.stats main.py stream --input test.mp4

# Check GPU utilization
nvidia-smi dmon -s u

# If GPU usage < 80%, bottleneck is elsewhere (CPU/IO)
# If GPU usage > 95%, need better GPU or lower resolution
```

### Issue: Stream Disconnection

```bash
# Check network connectivity
ping rtmp-server.example.com

# Test RTMP endpoint
ffmpeg -re -i test.mp4 -c copy -f flv rtmp://server/live/test

# Check firewall rules
sudo ufw status
```

### Issue: Service Crashes

```bash
# View logs
sudo journalctl -u football-tracker -n 100 --no-pager

# Check for GPU errors
dmesg | grep -i nvidia

# Restart service
sudo systemctl restart football-tracker
```

---

## Cost Optimization

### Reserved Instances

For 24/7 operation, purchase 1-year reserved instances:

```
g6.xlarge on-demand: ~$1.20/hour = $876/month
g6.xlarge reserved (1-yr): ~$0.72/hour = $525/month
Savings: ~40%
```

### Spot Instances

For non-critical workloads:

```bash
aws ec2 run-instances \
  --instance-type g6.xlarge \
  --instance-market-options MarketType=spot,SpotOptions={MaxPrice=0.60} \
  ...
```

Typical savings: 60-70% vs on-demand

### Scheduled Scaling

If streams only during certain hours:

```bash
# Scale down at 2 AM
aws autoscaling put-scheduled-action \
  --auto-scaling-group-name football-tracker-asg \
  --scheduled-action-name scale-down-night \
  --recurrence "0 2 * * *" \
  --desired-capacity 0

# Scale up at 6 AM
aws autoscaling put-scheduled-action \
  --auto-scaling-group-name football-tracker-asg \
  --scheduled-action-name scale-up-morning \
  --recurrence "0 6 * * *" \
  --desired-capacity 2
```

---

## Security Best Practices

1. **Use IAM roles** instead of access keys
2. **Enable VPC Flow Logs** for network monitoring
3. **Use AWS Secrets Manager** for credentials
4. **Enable CloudTrail** for audit logging
5. **Restrict security groups** to known IPs
6. **Keep system updated**: `sudo apt update && sudo apt upgrade -y`
7. **Use HTTPS** for any web interfaces
8. **Rotate SSH keys** regularly

---

## Support

For deployment issues:
- GitHub Issues: https://github.com/chele-s/Football-Detection/issues
- AWS Support: Open a support ticket in AWS Console
- Enterprise Support: alvanezg1@gmail.com

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Instance Type:** AWS G6 (NVIDIA L4)

