#!/bin/bash
set -e

# ============================================================================
# RF-DETR Football Detection - Production Installation Script
# ============================================================================
# This script handles all known issues from deployment including:
# - CMake policy warnings for PyNvCodec
# - NVIDIA driver/library version mismatches
# - NVDEC/NVENC capability validation
# - Python 3.10 environment setup
# - GPU diagnostics
# ============================================================================

COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m' # No Color

log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_NC} $1"
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_NC} $1"
}

log_warning() {
    echo -e "${COLOR_YELLOW}[WARNING]${COLOR_NC} $1"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_NC} $1"
}

# ============================================================================
# 1. System Requirements Check
# ============================================================================
log_info "Checking system requirements..."

# Check Ubuntu version
if ! grep -q "Ubuntu" /etc/os-release; then
    log_warning "Not running Ubuntu. This script is optimized for Ubuntu 22.04/24.04"
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

log_success "NVIDIA driver detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"

# Check Python 3.10
if ! command -v python3.10 &> /dev/null; then
    log_info "Python 3.10 not found. Installing..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev
else
    log_success "Python 3.10 found: $(python3.10 --version)"
fi

# ============================================================================
# 2. Install System Dependencies
# ============================================================================
log_info "Installing system dependencies..."

sudo apt update
sudo apt install -y \
    git \
    tmux \
    ffmpeg \
    build-essential \
    cmake \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    nvidia-cuda-toolkit \
    curl \
    wget

log_success "System dependencies installed"

# ============================================================================
# 3. NVIDIA Driver and Library Alignment
# ============================================================================
log_info "Checking NVIDIA driver/library alignment..."

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
log_info "Detected driver version: $DRIVER_VERSION"

# Ensure matching nvidia-utils and encode/decode libraries
if [ -f /etc/apt/sources.list.d/cuda.list ] || [ -f /etc/apt/sources.list.d/nvidia-container-runtime.list ]; then
    log_info "Installing/updating NVIDIA libraries for driver $DRIVER_VERSION..."
    
    # Remove any conflicting packages
    sudo apt remove --purge -y libnvidia-encode-* libnvidia-decode-* 2>/dev/null || true
    
    # Install matching versions
    sudo apt update
    sudo apt install -y \
        nvidia-utils-${DRIVER_VERSION} \
        libnvidia-encode-${DRIVER_VERSION} \
        libnvidia-decode-${DRIVER_VERSION} \
        libnvidia-fbc1-${DRIVER_VERSION} || {
        log_warning "Could not install versioned libraries. Trying generic packages..."
        sudo apt install -y libnvidia-encode-1 libnvidia-decode-1
    }
    
    log_success "NVIDIA libraries installed"
else
    log_warning "CUDA repository not configured. Skipping library alignment."
fi

# ============================================================================
# 4. Verify NVDEC/NVENC Support
# ============================================================================
log_info "Verifying NVDEC/NVENC hardware support..."

NVENC_COUNT=$(nvidia-smi --query-gpu=encoder.max_sessions --format=csv,noheader 2>/dev/null || echo "0")
NVDEC_COUNT=$(nvidia-smi --query-gpu=decoder.max_sessions --format=csv,noheader 2>/dev/null || echo "0")

if [ "$NVENC_COUNT" = "0" ] || [ "$NVDEC_COUNT" = "0" ]; then
    log_warning "NVENC/NVDEC may not be available on this GPU"
    log_warning "Pipeline will fall back to CPU video processing"
else
    log_success "NVENC sessions available: $NVENC_COUNT"
    log_success "NVDEC sessions available: $NVDEC_COUNT"
fi

# Check for video capability in Docker (if applicable)
if [ -f /.dockerenv ]; then
    log_info "Running inside Docker container"
    if ! nvidia-smi -q | grep -q "Video Encoder"; then
        log_error "Video encode/decode not available in container!"
        log_error "Restart container with: --gpus '\"capabilities=compute,graphics,utility,video\"'"
        exit 1
    fi
    log_success "Docker GPU video capabilities confirmed"
fi

# ============================================================================
# 5. Export Library Paths
# ============================================================================
log_info "Configuring library paths..."

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

# Make persistent
if ! grep -q "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    log_success "Library paths added to ~/.bashrc"
fi

# ============================================================================
# 6. Create Python Virtual Environment
# ============================================================================
log_info "Creating Python 3.10 virtual environment..."

VENV_PATH="rf-detr-venv-310"

if [ -d "$VENV_PATH" ]; then
    log_warning "Virtual environment already exists at $VENV_PATH"
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        log_info "Removed old environment"
    else
        log_info "Using existing environment"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    python3.10 -m venv "$VENV_PATH"
    log_success "Virtual environment created"
fi

source "$VENV_PATH/bin/activate"
log_success "Virtual environment activated"

# ============================================================================
# 7. Install Python Dependencies
# ============================================================================
log_info "Installing Python dependencies..."

pip install --upgrade pip setuptools wheel

# Install core dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    log_success "Core dependencies installed"
else
    log_error "requirements.txt not found!"
    exit 1
fi

# ============================================================================
# 8. Install PyNvCodec (with CMake policy fix)
# ============================================================================
log_info "Building PyNvCodec from source (with CMake policy fixes)..."

PYNVCODEC_DIR="PyNvCodec"

if [ -d "$PYNVCODEC_DIR" ]; then
    log_warning "PyNvCodec directory exists. Cleaning..."
    rm -rf "$PYNVCODEC_DIR"
fi

git clone https://github.com/NVIDIA/PyNvCodec.git
cd "$PYNVCODEC_DIR"

# Apply CMake policy fix for CMP0148 warning
log_info "Applying CMake policy fix..."
cat > cmake_policy_fix.patch << 'EOF'
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1,5 +1,8 @@
 cmake_minimum_required(VERSION 3.10)
 
+# Fix CMake policy warnings
+cmake_policy(SET CMP0148 OLD)
+
 project(PyNvCodec)
 
 set(CMAKE_CXX_STANDARD 17)
EOF

# Try to apply patch (ignore if already applied)
patch -p1 < cmake_policy_fix.patch 2>/dev/null || log_warning "Patch may already be applied or not needed"

# Build and install
mkdir -p build
cd build

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DGENERATE_PYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
    ..

make -j$(nproc)
make install

cd ../..
log_success "PyNvCodec built and installed"

# ============================================================================
# 9. Verify Installation
# ============================================================================
log_info "Running installation verification..."

python << 'EOF'
import sys
import torch
import cv2

print("\n" + "="*60)
print("Installation Verification")
print("="*60)

# Check Python version
print(f"Python: {sys.version}")

# Check PyTorch and CUDA
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Check OpenCV
print(f"OpenCV: {cv2.__version__}")

# Check PyNvCodec
try:
    import PyNvCodec as nvc
    print(f"PyNvCodec: Available")
    
    # Try to query GPU
    try:
        gpu_id = 0
        enc_info = nvc.EncodeDeviceInfo()
        if enc_info.IsSupported(gpu_id):
            print(f"NVENC: Supported on GPU {gpu_id}")
        else:
            print(f"NVENC: NOT supported on GPU {gpu_id}")
    except Exception as e:
        print(f"NVENC: Could not query ({str(e)})")
        
except ImportError as e:
    print(f"PyNvCodec: NOT available ({str(e)})")
    print("Pipeline will use CPU video processing")

# Check RF-DETR
try:
    import rfdetr
    print(f"RF-DETR: {rfdetr.__version__}")
except ImportError:
    print("RF-DETR: NOT installed")

print("="*60 + "\n")
EOF

if [ $? -eq 0 ]; then
    log_success "Installation verification passed"
else
    log_error "Installation verification failed"
    exit 1
fi

# ============================================================================
# 10. Create Required Directories
# ============================================================================
log_info "Creating project directories..."

mkdir -p models
mkdir -p data/inputs
mkdir -p data/outputs
mkdir -p clips
mkdir -p logs

log_success "Directories created"

# ============================================================================
# 11. Download Model Weights (optional)
# ============================================================================
if [ ! -f "models/best_rf-detr.pth" ]; then
    log_info "Model weights not found in models/best_rf-detr.pth"
    log_info "Please download your trained RF-DETR model and place it there"
else
    log_success "Model weights found"
fi

# ============================================================================
# 12. Create Service Management Scripts
# ============================================================================
log_info "Creating service management scripts..."

# Start script
cat > start_stream.sh << 'EOF'
#!/bin/bash
# Start RF-DETR streaming pipeline in tmux

SESSION_NAME="football-stream"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

cd "$(dirname "$0")"
source rf-detr-venv-310/bin/activate

tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "source rf-detr-venv-310/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python run_mjpeg_stream.py 2>&1 | tee logs/stream_$(date +%Y%m%d_%H%M%S).log" C-m

echo "Stream started in tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "Detach with: Ctrl+B then D"
echo "Stream URL: http://localhost:8554/stream.mjpg"
EOF

chmod +x start_stream.sh

# Stop script
cat > stop_stream.sh << 'EOF'
#!/bin/bash
# Stop RF-DETR streaming pipeline

SESSION_NAME="football-stream"

if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    tmux kill-session -t $SESSION_NAME
    echo "Stream session terminated"
else
    echo "No active stream session found"
fi
EOF

chmod +x stop_stream.sh

log_success "Service scripts created: start_stream.sh, stop_stream.sh"

# ============================================================================
# 13. Final Instructions
# ============================================================================
echo ""
echo "============================================================================"
log_success "Installation Complete!"
echo "============================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Verify GPU setup:"
echo "   python guides/verify_gpu_setup.py"
echo ""
echo "2. Configure your pipeline:"
echo "   - Edit configs/model_config.yml"
echo "   - Edit configs/stream_config.yml"
echo "   - Place model weights in models/best_rf-detr.pth"
echo ""
echo "3. Start the stream:"
echo "   ./start_stream.sh"
echo ""
echo "4. View the stream:"
echo "   vlc http://localhost:8554/stream.mjpg"
echo "   # Or tunnel from remote: ssh -N -L 8554:localhost:8554 user@server"
echo ""
echo "5. Stop the stream:"
echo "   ./stop_stream.sh"
echo ""
echo "For production deployment, see:"
echo "   - instruccion.txt (quick playbook)"
echo "   - README.md (full documentation)"
echo ""
echo "============================================================================"
echo ""

log_info "Reactivate environment with: source rf-detr-venv-310/bin/activate"
