#!/bin/bash
# ============================================================================
# PyNvCodec Installation Script with CMake Policy Fixes
# ============================================================================
# Builds PyNvCodec from source with fixes for common issues:
# - CMake CMP0148 policy warning
# - Python binding generation
# - Library path configuration
# ============================================================================

set -e

COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[0;31m'
COLOR_BLUE='\033[0;34m'
COLOR_NC='\033[0m'

log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_NC} $1"
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_NC} $1"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_NC} $1"
}

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    log_error "cmake not found. Install with: sudo apt install cmake"
    exit 1
fi

if ! command -v git &> /dev/null; then
    log_error "git not found. Install with: sudo apt install git"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    log_error "No virtual environment active. Activate with: source rf-detr-venv-310/bin/activate"
    exit 1
fi

log_success "Prerequisites OK"

# Clone repository
PYNVCODEC_DIR="PyNvCodec_build"
log_info "Cloning PyNvCodec..."

if [ -d "$PYNVCODEC_DIR" ]; then
    log_info "Removing existing PyNvCodec directory..."
    rm -rf "$PYNVCODEC_DIR"
fi

git clone https://github.com/NVIDIA/PyNvCodec.git "$PYNVCODEC_DIR"
cd "$PYNVCODEC_DIR"

log_success "Repository cloned"

# Apply CMake policy fix
log_info "Applying CMake CMP0148 policy fix..."

cat > cmake_policy_fix.patch << 'PATCH_EOF'
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -1,5 +1,8 @@
 cmake_minimum_required(VERSION 3.10)
 
+# Fix CMake policy CMP0148 warning (FindPythonInterp deprecation)
+cmake_policy(SET CMP0148 OLD)
+
 project(PyNvCodec)
 
 set(CMAKE_CXX_STANDARD 17)
PATCH_EOF

# Try to apply patch (may fail if file structure changed)
if patch -p1 --dry-run < cmake_policy_fix.patch &> /dev/null; then
    patch -p1 < cmake_policy_fix.patch
    log_success "CMake policy fix applied"
else
    log_info "Patch not applicable (may be already fixed or structure changed)"
    # Manually add policy to CMakeLists.txt if needed
    if ! grep -q "CMP0148" CMakeLists.txt; then
        log_info "Manually adding CMP0148 policy..."
        sed -i '1a\\n# Fix CMake policy CMP0148 warning\ncmake_policy(SET CMP0148 OLD)\n' CMakeLists.txt
    fi
fi

# Build
log_info "Building PyNvCodec..."

mkdir -p build
cd build

PYTHON_EXEC=$(which python)
PYTHON_PREFIX=$(python -c "import sys; print(sys.prefix)")

log_info "Python executable: $PYTHON_EXEC"
log_info "Python prefix: $PYTHON_PREFIX"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$PYTHON_EXEC" \
    -DGENERATE_PYTHON_BINDINGS=ON \
    -DCMAKE_INSTALL_PREFIX="$PYTHON_PREFIX" \
    -DCMAKE_CUDA_ARCHITECTURES="86;89" \
    ..

make -j$(nproc)

log_success "Build complete"

# Install
log_info "Installing PyNvCodec..."
make install

# Verify installation
log_info "Verifying installation..."

cd ../..

python << 'VERIFY_EOF'
try:
    import PyNvCodec as nvc
    print("✓ PyNvCodec imported successfully")
    
    # Try to query encoder info
    try:
        enc_info = nvc.EncodeDeviceInfo()
        print("✓ EncodeDeviceInfo created successfully")
    except Exception as e:
        print(f"⚠ EncodeDeviceInfo query failed: {e}")
        print("  This may be normal if GPU doesn't support NVENC")
    
except ImportError as e:
    print(f"✗ Failed to import PyNvCodec: {e}")
    exit(1)
VERIFY_EOF

if [ $? -eq 0 ]; then
    log_success "PyNvCodec installation verified"
    
    # Cleanup
    log_info "Cleaning up build directory..."
    cd ..
    rm -rf "$PYNVCODEC_DIR"
    log_success "Cleanup complete"
    
    echo ""
    echo "PyNvCodec installed successfully!"
    echo "You can now use hardware-accelerated video decoding/encoding."
else
    log_error "PyNvCodec verification failed"
    log_info "Build directory preserved at: $PYNVCODEC_DIR"
    exit 1
fi
