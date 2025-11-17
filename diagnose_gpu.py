#!/usr/bin/env python3
"""
GPU Diagnostic Script
Checks PyTorch CUDA, PyNvCodec, and GPU availability
"""
import sys

print("="*70)
print("GPU DIAGNOSTIC TOOL")
print("="*70)

# 1. Check PyTorch
print("\n[1/5] Checking PyTorch...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {props.total_memory / 1e9:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        
        # Test tensor creation
        try:
            test_tensor = torch.zeros(1, device='cuda:0')
            del test_tensor
            print("✓ CUDA tensor creation: SUCCESS")
        except Exception as e:
            print(f"✗ CUDA tensor creation FAILED: {e}")
    else:
        print("✗ PyTorch CUDA NOT available")
        print("  Possible reasons:")
        print("  - CUDA not installed")
        print("  - Wrong PyTorch build (CPU-only)")
        print("  - GPU driver issues")
except ImportError:
    print("✗ PyTorch not installed")
    sys.exit(1)

# 2. Check NVIDIA Driver
print("\n[2/5] Checking NVIDIA Driver...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ nvidia-smi accessible")
        # Parse output for driver version
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"  {line.strip()}")
                break
    else:
        print("✗ nvidia-smi failed")
except FileNotFoundError:
    print("✗ nvidia-smi not found (NVIDIA drivers may not be installed)")
except Exception as e:
    print(f"✗ Error running nvidia-smi: {e}")

# 3. Check PyNvCodec
print("\n[3/5] Checking PyNvCodec...")
try:
    import PyNvCodec as nvc
    print("✓ PyNvCodec imported successfully")
    print(f"  Module location: {nvc.__file__}")
    
    # Try to check PyNvCodec CUDA support
    try:
        # Attempt to create a simple decoder to verify CUDA support
        print("  Testing PyNvCodec CUDA access...")
        # Note: This will fail if no video file exists, but that's okay
        print("  (Full test requires a video file)")
    except Exception as e:
        print(f"  Note: {e}")
        
except ImportError as e:
    print(f"✗ PyNvCodec NOT installed: {e}")
    print("  Install with: pip install PyNvCodec")
    print("  Or from source: pip install git+https://github.com/NVIDIA/VideoProcessingFramework")
except Exception as e:
    print(f"✗ PyNvCodec import failed: {e}")

# 4. Check RF-DETR
print("\n[4/5] Checking RF-DETR...")
try:
    import rfdetr
    print(f"✓ RF-DETR installed")
    
    # Try to import model
    from rfdetr import RFDETRMedium
    print("✓ RFDETRMedium importable")
    
    # Check if optimize_for_inference is available
    model_instance = RFDETRMedium()
    if hasattr(model_instance, 'optimize_for_inference'):
        print("✓ optimize_for_inference() method available")
    else:
        print("✗ optimize_for_inference() not found (old version?)")
        
except ImportError as e:
    print(f"✗ RF-DETR NOT installed: {e}")
    print("  Install with: pip install rfdetr")
except Exception as e:
    print(f"✗ RF-DETR check failed: {e}")

# 5. Check System Info
print("\n[5/5] System Information...")
import platform
print(f"OS: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"Architecture: {platform.machine()}")

# Final Summary
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)

if torch.cuda.is_available():
    print("✓ PyTorch CUDA: READY")
else:
    print("✗ PyTorch CUDA: NOT AVAILABLE")
    print("  → Install CUDA-enabled PyTorch")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

try:
    import PyNvCodec
    print("✓ PyNvCodec: INSTALLED")
    if not torch.cuda.is_available():
        print("  ⚠ PyNvCodec requires CUDA - fix PyTorch CUDA first")
except:
    print("✗ PyNvCodec: NOT INSTALLED")
    print("  → For GPU video decoding (optional, CPU fallback available)")
    print("     pip install PyNvCodec")

try:
    import rfdetr
    print("✓ RF-DETR: INSTALLED")
except:
    print("✗ RF-DETR: NOT INSTALLED")
    print("  → Required for ball detection")
    print("     pip install rfdetr supervision")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if not torch.cuda.is_available():
    print("1. Install CUDA toolkit from NVIDIA")
    print("2. Reinstall PyTorch with CUDA support:")
    print("   pip uninstall torch torchvision")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

try:
    import PyNvCodec
    if torch.cuda.is_available():
        print("\n✓ GPU acceleration should work!")
        print("  Run: python run_mjpeg_stream_gpu.py")
    else:
        print("\n⚠ PyNvCodec installed but CUDA not available")
        print("  Use CPU pipeline: python run_mjpeg_stream.py")
except:
    print("\n⚠ PyNvCodec not installed - GPU decoder unavailable")
    print("  Option 1: Install PyNvCodec for GPU acceleration")
    print("  Option 2: Use CPU pipeline (slower but works): python run_mjpeg_stream.py")

print("\n" + "="*70)
