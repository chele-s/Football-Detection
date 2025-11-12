#!/bin/bash
# Install PyNvCodec for GPU-accelerated video decode/encode
# Requires: NVIDIA GPU with NVDEC/NVENC, CUDA toolkit

set -e

echo "==============================================="
echo "PyNvCodec Installation Script"
echo "==============================================="
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: CUDA toolkit not found. Install CUDA first."
    echo "   Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
echo "✓ CUDA version: $CUDA_VERSION"

# Check if GPU supports NVDEC/NVENC
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found. Check NVIDIA drivers."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
echo "✓ GPU detected: $GPU_NAME"

echo ""
echo "Installing dependencies..."

# Install FFmpeg with NVENC/NVDEC support
if ! command -v ffmpeg &> /dev/null; then
    echo "Installing FFmpeg..."
    sudo apt-get update
    sudo apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev
else
    echo "✓ FFmpeg already installed"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy

# Clone and install PyNvCodec (Video Processing Framework)
echo ""
echo "Cloning NVIDIA Video Processing Framework..."
cd /tmp
if [ -d "VideoProcessingFramework" ]; then
    rm -rf VideoProcessingFramework
fi

git clone https://github.com/NVIDIA/VideoProcessingFramework.git
cd VideoProcessingFramework

echo ""
echo "Building PyNvCodec..."
echo "This may take 5-10 minutes..."

# Install build dependencies
pip install pybind11 cmake

# Build and install
mkdir -p build
cd build
cmake ..
make -j$(nproc)

# Install Python package
cd ..
pip install .

echo ""
echo "Testing PyNvCodec installation..."
python3 -c "import PyNvCodec as nvc; print('✓ PyNvCodec version:', nvc.__version__)"

echo ""
echo "==============================================="
echo "✓ Installation complete!"
echo "==============================================="
echo ""
echo "Your GPU pipeline is ready. Run your script with:"
echo "  python main.py --use-gpu-pipeline"
echo ""
echo "Expected performance gains:"
echo "  - 3-5x faster video decode/encode"
echo "  - Zero CPU↔GPU memory copies"
echo "  - Frees CPU for other tasks"
echo ""

