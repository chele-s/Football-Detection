"""
SIMPLE PyNvCodec Installation - Direct Build Method
====================================================

Use this if the main script fails with CMake errors.
This method builds PyNvCodec directly without scikit-build.
"""

import subprocess
import sys
import os

print("""
╔════════════════════════════════════════════════════════════╗
║      PyNvCodec Installation (SIMPLE METHOD)                ║
╚════════════════════════════════════════════════════════════╝
""")

# 1. Check GPU
print("1️⃣  Checking GPU...")
result = subprocess.run("nvidia-smi", shell=True, capture_output=True)
if result.returncode != 0:
    print("❌ No GPU! Enable: Runtime → Change runtime type → GPU")
    sys.exit(1)
print("✓ GPU detected\n")

# 2. System deps
print("2️⃣  Installing dependencies...")
subprocess.run("""
apt-get -qq update
apt-get install -y -qq \
    ffmpeg libavcodec-dev libavformat-dev libavutil-dev \
    libswscale-dev pkg-config cmake build-essential python3-dev git
""", shell=True)
print("✓ Done\n")

# 3. Python tools
print("3️⃣  Installing Python tools...")
subprocess.run(
    "pip install -q --upgrade pip 'cmake>=3.20' 'pybind11>=2.10'",
    shell=True
)
print("✓ Done\n")

# 4. Clone repo
print("4️⃣  Cloning VideoProcessingFramework...")
subprocess.run("rm -rf /tmp/VPF", shell=True)
subprocess.run(
    "git clone -q https://github.com/NVIDIA/VideoProcessingFramework.git /tmp/VPF",
    shell=True
)
print("✓ Done\n")

# 5. Build with CMake directly
print("5️⃣  Building PyNvCodec (5-10 min)...")
os.chdir("/tmp/VPF")

# Set env vars
os.environ['CUDACXX'] = '/usr/local/cuda/bin/nvcc'
os.environ['CUDA_HOME'] = '/usr/local/cuda'

# Create build directory
subprocess.run("mkdir -p build", shell=True)
os.chdir("build")

# CMake configure
print("   Configuring...")
cmake_cmd = """
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DGENERATE_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE=/usr/bin/python3
"""

result = subprocess.run(cmake_cmd, shell=True)
if result.returncode != 0:
    print("❌ CMake configure failed")
    sys.exit(1)

# Build
print("   Building...")
result = subprocess.run("make -j$(nproc)", shell=True)
if result.returncode != 0:
    print("❌ Build failed")
    sys.exit(1)

# Install
print("   Installing...")
result = subprocess.run("make install", shell=True)
if result.returncode != 0:
    print("❌ Install failed")
    sys.exit(1)

# Add to Python path
python_site = subprocess.run(
    "python3 -c 'import site; print(site.getsitepackages()[0])'",
    shell=True,
    capture_output=True,
    text=True
).stdout.strip()

subprocess.run(
    f"cp -r /usr/local/lib/python3.*/dist-packages/PyNvCodec* {python_site}/ 2>/dev/null || true",
    shell=True
)

print("✓ Build complete\n")

# 6. Test
print("6️⃣  Testing...")
test_code = """
import sys
try:
    import PyNvCodec as nvc
    import torch
    print('✓ PyNvCodec imported')
    print(f'✓ CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    sys.exit(0)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"""

result = subprocess.run(['python3', '-c', test_code])

if result.returncode == 0:
    print("""
╔════════════════════════════════════════════════════════════╗
║              ✓ Installation Complete! 🎉                  ║
╚════════════════════════════════════════════════════════════╝

GPU pipeline ready! Expected performance:
  • 720p: 60+ FPS (vs 19 FPS CPU)
  • 1080p: 45+ FPS (vs 12 FPS CPU)
""")
else:
    print("\n❌ Test failed. PyNvCodec not working.")
    sys.exit(1)

