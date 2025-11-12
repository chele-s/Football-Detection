"""
Install PyNvCodec on Google Colab
Run this in a Colab notebook cell:

!python install_pynvcodec_colab.py
"""

import subprocess
import sys
import os

def run_cmd(cmd, description):
    """Run command and print output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    return True

def main():
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║         PyNvCodec Installation for Google Colab            ║
    ║   GPU-Accelerated Video Decode/Encode (NVDEC/NVENC)       ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    # Check GPU
    print("\n1. Checking GPU...")
    gpu_check = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
    if gpu_check.returncode != 0:
        print("❌ No GPU detected. Enable GPU: Runtime → Change runtime type → GPU")
        return False
    
    print("✓ GPU detected:")
    # Extract GPU name
    for line in gpu_check.stdout.split('\n'):
        if 'Tesla' in line or 'T4' in line or 'V100' in line or 'A100' in line:
            print(f"  {line.strip()}")
            break
    
    # Install system dependencies
    if not run_cmd(
        "apt-get update && apt-get install -y ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev pkg-config",
        "2. Installing system dependencies (FFmpeg, codecs)"
    ):
        return False
    
    # Install Python build tools
    if not run_cmd(
        "pip install --upgrade pip cmake pybind11",
        "3. Installing Python build tools"
    ):
        return False
    
    # Clone VideoProcessingFramework
    print("\n4. Cloning NVIDIA Video Processing Framework...")
    if os.path.exists("/tmp/VideoProcessingFramework"):
        run_cmd("rm -rf /tmp/VideoProcessingFramework", "Removing old build")
    
    if not run_cmd(
        "cd /tmp && git clone https://github.com/NVIDIA/VideoProcessingFramework.git",
        "Cloning repository"
    ):
        return False
    
    # Build PyNvCodec
    print("\n5. Building PyNvCodec (this takes ~5 minutes)...")
    build_commands = """
    cd /tmp/VideoProcessingFramework && \
    export CUDACXX=/usr/local/cuda/bin/nvcc && \
    pip install .
    """
    
    result = subprocess.run(build_commands, shell=True, capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"❌ Build failed: {result.stderr}")
        print("\nTrying alternative installation method...")
        
        # Alternative: Install pre-built wheel (if available)
        alt_result = subprocess.run(
            "pip install PyNvCodec",
            shell=True, 
            capture_output=True, 
            text=True
        )
        
        if alt_result.returncode != 0:
            print("❌ Installation failed. Check CUDA version and GPU compatibility.")
            return False
    
    # Test installation
    print("\n6. Testing PyNvCodec...")
    test_code = """
import PyNvCodec as nvc
import torch
print('✓ PyNvCodec imported successfully')
print(f'  Version: {nvc.__version__ if hasattr(nvc, "__version__") else "N/A"}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"""
    
    test_result = subprocess.run(
        f'python -c "{test_code}"',
        shell=True,
        capture_output=True,
        text=True
    )
    
    print(test_result.stdout)
    
    if test_result.returncode != 0:
        print(f"❌ Test failed: {test_result.stderr}")
        return False
    
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              ✓ Installation Complete!                     ║
    ╚════════════════════════════════════════════════════════════╝
    
    Your GPU video pipeline is ready! 🚀
    
    Expected performance gains:
      • 3-5x faster video decode/encode
      • Zero CPU↔GPU memory transfers  
      • Frees CPU for other tasks
    
    Next steps:
      1. Update your config to use GPU pipeline
      2. Run your stream pipeline
      3. Enjoy 60+ FPS on 720p video! 🎉
    
    Note: T4 GPU in Colab supports:
      - NVDEC: H.264, H.265, VP9 decode
      - NVENC: H.264, H.265 encode at 60+ FPS
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

