# Football Detection - Detailed Deployment Guide

This guide provides step-by-step instructions for deploying the Football Detection system in two environments:
1.  **AWS Cloud**: Using G6e instances with L40S GPUs.
2.  **Local Windows**: Using a high-end workstation with RTX 5070 Ti+ (or equivalent high-performance GPU).

---

## Part 1: AWS Deployment (G6e / L40S)

The **Amazon EC2 G6e** instances feature NVIDIA L40S GPUs, which are excellent for inference and video processing.

### 1. Launching the Instance

1.  **Log in to AWS Console** and navigate to **EC2**.
2.  Click **Launch Instances**.
3.  **Name**: `football-detection-prod`
4.  **OS / AMI**:
    *   Search for "Deep Learning AMI".
    *   Select **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.0.1 (Ubuntu 22.04)** (or similar latest version).
    *   *Why?* This comes with NVIDIA drivers, CUDA, and Docker pre-installed, saving hours of setup.
5.  **Instance Type**:
    *   Search for `g6e`.
    *   Select `g6e.xlarge` (1 GPU, 4 vCPUs, 32GB RAM) for standard usage.
    *   Select `g6e.2xlarge` (1 GPU, 8 vCPUs, 64GB RAM) if you need more CPU power for video decoding.
6.  **Key Pair**: Select an existing key pair or create a new one (save the `.pem` file).
7.  **Network Settings**:
    *   Check **Allow SSH traffic from** (My IP).
    *   Check **Allow HTTP traffic from the internet**.
    *   **IMPORTANT**: You must open port `8554` for the MJPEG stream.
        *   Click "Edit" in Network Settings.
        *   Add Security Group Rule: Custom TCP, Port `8554`, Source `0.0.0.0/0` (or your specific IP for security).
8.  **Storage**: Increase the Root volume to at least **100 GB** (gp3) to accommodate video files and Docker images.
9.  Click **Launch Instance**.

### 2. Connecting to the Instance

Open your terminal (or PowerShell on Windows) and SSH into the server:

```bash
ssh -i "path/to/your-key.pem" ubuntu@<your-instance-public-ip>
```

### 3. System Setup & Installation

Once connected, follow these steps to set up the environment.

#### Step 3.1: Clone the Repository
```bash
git clone https://github.com/chele-s/Football-Detection.git
cd Football-Detection
```

#### Step 3.2: Run the Production Setup Script
We have a dedicated script that handles all dependencies, including the tricky `PyNvCodec` installation for hardware acceleration.

```bash
chmod +x setup_production.sh
./setup_production.sh
```

*Note: This script will take 5-10 minutes to complete as it compiles libraries and installs Python dependencies.*

### 4. Model Setup

You need to upload your trained model weights (`best_rf-detr.pth`).

**Option A: Upload via SCP (from your local machine)**
```bash
scp -i "path/to/key.pem" path/to/local/best_rf-detr.pth ubuntu@<ip>:/home/ubuntu/Football-Detection/models/
```

**Option B: Download from S3 (if stored in AWS)**
```bash
# Configure AWS CLI first if needed
aws s3 cp s3://your-bucket/best_rf-detr.pth ./models/
```

### 5. Configuration

Edit the stream configuration to point to your video source (RTSP url or file path).

```bash
nano configs/stream_config.yml
```

Update the `input` section:
```yaml
input:
  source: "path/to/video.mp4" # or rtsp://...
  # ...
```

### 6. Running the Service

Use the generated helper scripts to run the stream in the background (using tmux).

**Start the Stream:**
```bash
./start_stream.sh
```

**Stop the Stream:**
```bash
./stop_stream.sh
```

**View Logs:**
```bash
tail -f logs/stream_*.log
```

### 7. Accessing the Stream
Open your browser or VLC player and navigate to:
`http://<your-instance-public-ip>:8554/stream.mjpg`

---

## Part 2: Windows Local Deployment (RTX 5070 Ti+)

This guide assumes you have a high-end Windows workstation with an NVIDIA RTX 5070 Ti or better.

### 1. Prerequisites

Before starting, ensure you have the following installed:

1.  **NVIDIA Drivers**: Install the latest "Game Ready" or "Studio" drivers from GeForce Experience or the NVIDIA website.
2.  **Git**: Download and install from [git-scm.com](https://git-scm.com/download/win).
3.  **PowerShell**: Ensure you can run PowerShell as Administrator.

### 2. Installation

We have automated the entire Windows setup process.

1.  **Open PowerShell as Administrator**.
    *   Right-click the Start button -> Terminal (Admin) or PowerShell (Admin).

2.  **Clone the Repository** (if you haven't already):
    ```powershell
    cd C:\Users\YourUser\Documents
    git clone https://github.com/chele-s/Football-Detection.git
    cd Football-Detection
    ```

3.  **Run the Setup Script**:
    This script will automatically:
    *   Install Python 3.10 (if missing).
    *   Install CUDA Toolkit (if missing).
    *   Install FFmpeg.
    *   Create a virtual environment.
    *   Install all Python dependencies (PyTorch, OpenCV, etc.).

    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force
    .\setup_production_windows.ps1
    ```

    *Follow the on-screen prompts. If asked to restart, please do so and re-run the script.*

### 3. Verification

After installation, run the diagnostics script to ensure your RTX 5070 Ti is correctly detected and CUDA is working:

```powershell
.\diagnose_gpu_windows.ps1
```

You should see green `[OK]` messages for GPU, PyTorch, and CUDA.

### 4. Running the Application

You can start the application using the generated batch file or PowerShell script.

**Option A: Double-Click**
*   Open the folder in File Explorer.
*   Double-click `start_stream.bat`.

**Option B: PowerShell**
```powershell
.\start_stream.ps1
```

The stream will be available at: `http://localhost:8554/stream.mjpg`

### 5. Performance Tuning for RTX 5070 Ti+

For a high-end card like the 5070 Ti, you can maximize quality and performance in `configs/stream_config.yml`:

```yaml
processing:
  device: "cuda:0"
  resolution: [3840, 2160] # 4K Processing
  batch_size: 1 # Keep at 1 for lowest latency in real-time

visualization:
  resize_output: 1.0 # 100% scale (4K output)
```

### Troubleshooting Windows

*   **"CUDA not available"**: Re-run `setup_production_windows.ps1`. It often fixes path issues.
*   **Permission Errors**: Always run PowerShell as Administrator.
*   **Video Load Errors**: Ensure FFmpeg is in your system PATH (the setup script handles this, but a reboot might be required).
