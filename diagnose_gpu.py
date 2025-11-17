#!/usr/bin/env python3
"""
Extended GPU diagnostics for RF-DETR Football Detection
Performs comprehensive checks and provides troubleshooting guidance
"""

import sys
import subprocess
from pathlib import Path

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print_section("NVIDIA Driver")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", 
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        info = result.stdout.strip().split(", ")
        print(f"✓ Driver Version: {info[0]}")
        print(f"✓ GPU Model: {info[1]}")
        print(f"✓ VRAM: {info[2]}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ nvidia-smi not found or failed")
        print("  → Install NVIDIA drivers: sudo apt install nvidia-driver-550")
        return False

def check_nvenc_nvdec():
    """Check NVENC/NVDEC support"""
    print_section("Hardware Video Acceleration")
    try:
        # Check encoder
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=encoder.max_sessions", 
             "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        enc_sessions = result.stdout.strip()
        
        # Check decoder
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=decoder.max_sessions", 
             "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        dec_sessions = result.stdout.strip()
        
        if enc_sessions != "0":
            print(f"✓ NVENC: Supported (max {enc_sessions} sessions)")
        else:
            print("✗ NVENC: NOT supported")
            print("  → GPU may not have video encode capability")
            
        if dec_sessions != "0":
            print(f"✓ NVDEC: Supported (max {dec_sessions} sessions)")
        else:
            print("✗ NVDEC: NOT supported")
            print("  → GPU may not have video decode capability")
            
        # Check for Video Encoder/Decoder in detailed query
        result = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True,
            text=True
        )
        
        if "Video Encoder" not in result.stdout:
            print("\n⚠ WARNING: Video Encoder not visible in nvidia-smi -q")
            if Path("/.dockerenv").exists():
                print("  → Running in Docker without video capability")
                print("  → Add to docker run: --gpus '\"capabilities=compute,graphics,utility,video\"'")
        
        return True
    except Exception as e:
        print(f"✗ Could not query NVENC/NVDEC: {e}")
        return False

def check_cuda():
    """Check CUDA installation"""
    print_section("CUDA Toolkit")
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True
        )
        version_line = [l for l in result.stdout.split("\n") if "release" in l][0]
        print(f"✓ CUDA Toolkit: {version_line.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        print("✗ nvcc not found")
        print("  → Install CUDA: sudo apt install nvidia-cuda-toolkit")
        print("  → Or download from: https://developer.nvidia.com/cuda-downloads")
        return False

def check_python():
    """Check Python environment"""
    print_section("Python Environment")
    print(f"✓ Python Version: {sys.version.split()[0]}")
    print(f"✓ Python Path: {sys.executable}")
    
    if sys.version_info < (3, 10):
        print("⚠ WARNING: Python < 3.10 detected")
        print("  → Recommended: Python 3.10 for PyNvCodec compatibility")

def check_pytorch():
    """Check PyTorch and CUDA availability"""
    print_section("PyTorch")
    try:
        import torch
        print(f"✓ PyTorch Version: {torch.__version__}")
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"✓ Device Name: {torch.cuda.get_device_name(0)}")
            print(f"✓ Device Count: {torch.cuda.device_count()}")
            
            # Test tensor creation
            try:
                test_tensor = torch.randn(100, 100).cuda()
                print(f"✓ GPU Tensor Creation: OK")
            except Exception as e:
                print(f"✗ GPU Tensor Creation Failed: {e}")
        else:
            print("✗ CUDA not available in PyTorch")
            print("  → Reinstall PyTorch with CUDA:")
            print("  → pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return True
    except ImportError:
        print("✗ PyTorch not installed")
        print("  → Install: pip install torch torchvision")
        return False

def check_opencv():
    """Check OpenCV"""
    print_section("OpenCV")
    try:
        import cv2
        print(f"✓ OpenCV Version: {cv2.__version__}")
        
        # Check for CUDA support in OpenCV (optional)
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                print(f"✓ OpenCV CUDA: Available ({cuda_count} devices)")
            else:
                print("  OpenCV CUDA: Not available (CPU fallback will be used)")
        except:
            print("  OpenCV CUDA: Not available (normal for pip installs)")
        return True
    except ImportError:
        print("✗ OpenCV not installed")
        print("  → Install: pip install opencv-python")
        return False

def check_pynvcodec():
    """Check PyNvCodec installation"""
    print_section("PyNvCodec (Hardware Video Acceleration)")
    try:
        import PyNvCodec as nvc
        print("✓ PyNvCodec: Installed")
        
        # Try to query encoder info
        try:
            gpu_id = 0
            enc_info = nvc.EncodeDeviceInfo()
            if enc_info.IsSupported(gpu_id):
                print(f"✓ NVENC Available: GPU {gpu_id}")
            else:
                print(f"✗ NVENC NOT available on GPU {gpu_id}")
                print("  → Check if GPU supports NVENC: https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix")
        except Exception as e:
            print(f"⚠ Could not query NVENC: {e}")
            
        return True
    except ImportError as e:
        print(f"✗ PyNvCodec not installed: {e}")
        print("  → Build from source: bash scripts/install_pynvcodec.sh")
        print("  → Or see: https://github.com/NVIDIA/PyNvCodec")
        return False

def check_rfdetr():
    """Check RF-DETR"""
    print_section("RF-DETR")
    try:
        import rfdetr
        print(f"✓ RF-DETR Version: {rfdetr.__version__}")
        return True
    except ImportError:
        print("✗ RF-DETR not installed")
        print("  → Install: pip install rfdetr")
        return False

def check_library_paths():
    """Check LD_LIBRARY_PATH configuration"""
    print_section("Library Paths")
    import os
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    
    if "/usr/lib/x86_64-linux-gnu" in ld_path:
        print("✓ LD_LIBRARY_PATH includes /usr/lib/x86_64-linux-gnu")
    else:
        print("⚠ LD_LIBRARY_PATH may not include NVIDIA libraries")
        print("  → Add to ~/.bashrc:")
        print("    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH")
    
    # Check for NVIDIA encode/decode libraries
    lib_dir = Path("/usr/lib/x86_64-linux-gnu")
    
    encode_libs = list(lib_dir.glob("libnvidia-encode.so*"))
    decode_libs = list(lib_dir.glob("libnvidia-decode.so*"))
    
    if encode_libs:
        print(f"✓ libnvidia-encode found: {encode_libs[0].name}")
    else:
        print("✗ libnvidia-encode.so not found")
        print("  → Install: sudo apt install libnvidia-encode-550")
        
    if decode_libs:
        print(f"✓ libnvidia-decode found: {decode_libs[0].name}")
    else:
        print("✗ libnvidia-decode.so not found")
        print("  → Install: sudo apt install libnvidia-decode-550")

def check_model_weights():
    """Check for model weights"""
    print_section("Model Weights")
    model_path = Path("models/best_rf-detr.pth")
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model found: {model_path} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Model not found: {model_path}")
        print("  → Download or copy your trained RF-DETR weights")
        print("  → Update path in configs/model_config.yml")

def check_configs():
    """Check configuration files"""
    print_section("Configuration Files")
    
    configs = [
        "configs/model_config.yml",
        "configs/stream_config.yml"
    ]
    
    for config in configs:
        path = Path(config)
        if path.exists():
            print(f"✓ {config}")
        else:
            print(f"✗ {config} not found")

def main():
    """Run all diagnostics"""
    print("\n" + "=" * 70)
    print("  RF-DETR Football Detection - GPU Diagnostics")
    print("=" * 70)
    
    checks = [
        check_nvidia_driver,
        check_nvenc_nvdec,
        check_cuda,
        check_python,
        check_pytorch,
        check_opencv,
        check_pynvcodec,
        check_rfdetr,
        check_library_paths,
        check_model_weights,
        check_configs,
    ]
    
    results = [check() for check in checks]
    
    # Summary
    print_section("Summary")
    passed = sum(1 for r in results if r)
    total = len(results)
    
    print(f"Checks Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All checks passed! System is ready for production.")
        print("\nNext steps:")
        print("  1. Configure: edit configs/stream_config.yml")
        print("  2. Start: ./start_stream.sh")
        print("  3. View: vlc http://localhost:8554/stream.mjpg")
    else:
        print("\n⚠ Some checks failed. Review the output above for troubleshooting.")
        print("\nCommon fixes:")
        print("  - Driver mismatch: sudo apt install nvidia-utils-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)")
        print("  - Missing PyNvCodec: bash scripts/install_pynvcodec.sh")
        print("  - CUDA unavailable: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    main()
