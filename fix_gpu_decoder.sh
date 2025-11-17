#!/bin/bash
# Fix PyNvCodec CUDA_ERROR_NO_DEVICE issue on Ubuntu

echo "=== Fixing GPU Decoder CUDA_ERROR_NO_DEVICE ==="

# 1. Find libnvcuvid.so location
echo "Searching for libnvcuvid.so..."
NVCUVID_PATHS=$(find /usr -name "libnvcuvid.so*" 2>/dev/null)

if [ -z "$NVCUVID_PATHS" ]; then
    echo "ERROR: libnvcuvid.so not found!"
    echo "Installing NVIDIA drivers with video support..."
    sudo apt-get update
    sudo apt-get install -y libnvidia-decode-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
    NVCUVID_PATHS=$(find /usr -name "libnvcuvid.so*" 2>/dev/null)
fi

echo "Found: $NVCUVID_PATHS"

# 2. Create symlinks
NVCUVID_PATH=$(echo "$NVCUVID_PATHS" | head -n 1 | xargs dirname)
echo "Creating symlinks in $NVCUVID_PATH..."

if [ -f "$NVCUVID_PATH/libnvcuvid.so.1" ]; then
    echo "libnvcuvid.so.1 exists"
else
    if [ -f "$NVCUVID_PATH/libnvcuvid.so" ]; then
        sudo ln -sf "$NVCUVID_PATH/libnvcuvid.so" "$NVCUVID_PATH/libnvcuvid.so.1"
        echo "Created symlink: libnvcuvid.so.1"
    fi
fi

# 3. Export LD_LIBRARY_PATH
echo "Setting LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH=$NVCUVID_PATH:$LD_LIBRARY_PATH
echo "export LD_LIBRARY_PATH=$NVCUVID_PATH:\$LD_LIBRARY_PATH" >> ~/.bashrc

# 4. Test CUDA
echo "Testing CUDA access..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 5. Test PyNvCodec
echo "Testing PyNvCodec..."
python3 -c "
try:
    import PyNvCodec as nvc
    print('PyNvCodec imported successfully')
except Exception as e:
    print(f'PyNvCodec error: {e}')
"

echo "=== Fix applied. Try running: ==="
echo "export LD_LIBRARY_PATH=$NVCUVID_PATH:\$LD_LIBRARY_PATH"
echo "python3 run_mjpeg_stream_gpu.py"
