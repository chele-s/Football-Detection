#!/bin/bash
set -e

# ============================================================================
# RF-DETR Football Detection - Robust Production Setup
# ============================================================================
# Optimized for AWS G6e (Ubuntu 22.04) and Standard Ubuntu 22.04/24.04
# ============================================================================

COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m'

log_info() { echo -e "${COLOR_BLUE}[INFO]${COLOR_NC} $1"; }
log_success() { echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_NC} $1"; }
log_warning() { echo -e "${COLOR_YELLOW}[WARNING]${COLOR_NC} $1"; }
log_error() { echo -e "${COLOR_RED}[ERROR]${COLOR_NC} $1"; }

# ============================================================================
# 1. Environment Checks
# ============================================================================
log_info "Checking environment..."

# Check for NVIDIA GPU and Drivers
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    log_success "NVIDIA Driver detected: $DRIVER_VERSION"
    
    # On AWS DL AMI, we DO NOT want to mess with drivers
    INSTALL_DRIVERS=false
else
    log_warning "nvidia-smi not found. This machine may not have NVIDIA drivers installed."
    read -p "Do you want to attempt to install NVIDIA drivers? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        INSTALL_DRIVERS=true
    else
        log_error "Cannot proceed without GPU drivers. Exiting."
        exit 1
    fi
fi

# ============================================================================
# 2. System Dependencies
# ============================================================================
log_info "Installing system libraries..."

sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    cmake \
    build-essential \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    curl \
    unzip \
    tmux \
    ffmpeg

# Install NVIDIA Video Codec SDK headers if possible (often needed for PyNvCodec)
# We'll try to install the interface headers via apt if available, otherwise we rely on the git clone later.
sudo apt-get install -y libnvidia-encode-dev libnvidia-decode-dev || log_warning "Could not install nvidia-encode/decode-dev via apt. This is expected on some cloud kernels."

# ============================================================================
# 3. Python Environment
# ============================================================================
log_info "Setting up Python environment..."

VENV_NAME="rf-detr-venv-310"

if [ -d "$VENV_NAME" ]; then
    log_warning "Virtual environment $VENV_NAME already exists."
    # We don't delete it automatically to save time on re-runs, but we ensure it's valid
else
    python3.10 -m venv "$VENV_NAME"
    log_success "Created virtual environment: $VENV_NAME"
fi

source "$VENV_NAME/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Core Requirements
if [ -f "requirements.txt" ]; then
    log_info "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    log_error "requirements.txt not found!"
    exit 1
fi

# ============================================================================
# 4. PyNvCodec Installation (The Tricky Part)
# ============================================================================
log_info "Checking PyNvCodec..."

if python -c "import PyNvCodec" &> /dev/null; then
    log_success "PyNvCodec is already installed and working."
else
    log_info "PyNvCodec not found. Attempting to build from source..."
    
    # Clone Video Processing Framework
    if [ -d "VideoProcessingFramework" ]; then
        rm -rf VideoProcessingFramework
    fi
    
    git clone https://github.com/NVIDIA/VideoProcessingFramework.git
    cd VideoProcessingFramework
    
    # We need to find where the CUDA toolkit is
    # AWS DL AMI usually has it at /usr/local/cuda
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        log_info "Using CUDA at $CUDA_HOME"
    fi

    mkdir -p build
    cd build
    
    # Configure CMake
    # We disable tests and samples to speed up build and reduce errors
    cmake .. \
        -DENABLE_TESTS=OFF \
        -DENABLE_SAMPLES=OFF \
        -DPYTHON_EXECUTABLE=$(which python) \
        -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
        -DCMAKE_BUILD_TYPE=Release

    # Build and Install
    make -j$(nproc)
    make install
    
    cd ../..
    # Cleanup
    rm -rf VideoProcessingFramework
    
    # Verify
    if python -c "import PyNvCodec" &> /dev/null; then
        log_success "PyNvCodec successfully built and installed!"
    else
        log_error "PyNvCodec build failed or is not importable."
        log_warning "The system will fall back to CPU decoding (slower but functional)."
    fi
fi

# ============================================================================
# 5. Directory Setup
# ============================================================================
mkdir -p models data/inputs data/outputs clips logs
log_success "Project directories verified."

# ============================================================================
# 6. Service Scripts
# ============================================================================
# Create a simple start script
cat > start_stream.sh << 'EOF'
#!/bin/bash
source rf-detr-venv-310/bin/activate
export PYTHONPATH=$PYTHONPATH:.
# Log to file with timestamp
LOG_FILE="logs/stream_$(date +%Y%m%d_%H%M%S).log"
echo "Starting stream... Logs at $LOG_FILE"
python run_mjpeg_stream.py 2>&1 | tee "$LOG_FILE"
EOF
chmod +x start_stream.sh

# ============================================================================
# 7. Final Verification
# ============================================================================
echo ""
log_success "Setup Complete!"
echo ""
echo "To start the stream:"
echo "  ./start_stream.sh"
echo ""
echo "Remember to upload your model to: models/best_rf-detr.pth"
echo ""
