#!/usr/bin/env python3
"""
Quick verification script for GPU pipeline setup

Run this to check if your system is ready for GPU-accelerated video processing.
"""

import sys


def check_cuda():
    """Check CUDA availability"""
    print("\n1️⃣  Checking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   ✅ CUDA available")
            print(f"   ✅ GPU: {device_name}")
            print(f"   ✅ Memory: {device_memory:.1f} GB")
            return True
        else:
            print("   ❌ CUDA not available")
            print("      Enable GPU: Runtime → Change runtime type → GPU")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        print("      Install: pip install torch")
        return False


def check_pynvcodec():
    """Check PyNvCodec installation"""
    print("\n2️⃣  Checking PyNvCodec...")
    try:
        import PyNvCodec as nvc
        version = nvc.__version__ if hasattr(nvc, '__version__') else "unknown"
        print(f"   ✅ PyNvCodec installed (version: {version})")
        return True
    except ImportError:
        print("   ❌ PyNvCodec not installed")
        print("      Install: python install_pynvcodec_colab.py")
        return False


def check_ffmpeg():
    """Check FFmpeg availability"""
    print("\n3️⃣  Checking FFmpeg...")
    import subprocess
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract version
            first_line = result.stdout.split('\n')[0]
            print(f"   ✅ FFmpeg installed: {first_line}")
            
            # Check for NVENC support
            result_codecs = subprocess.run(
                ['ffmpeg', '-codecs'],
                capture_output=True,
                text=True,
                timeout=5
            )
            has_h264_nvenc = 'h264_nvenc' in result_codecs.stdout
            has_hevc_nvenc = 'hevc_nvenc' in result_codecs.stdout
            
            if has_h264_nvenc or has_hevc_nvenc:
                print(f"   ✅ NVENC support detected")
            else:
                print(f"   ⚠️  NVENC not detected (optional)")
            
            return True
        else:
            print("   ❌ FFmpeg not working")
            return False
    except FileNotFoundError:
        print("   ❌ FFmpeg not installed")
        print("      Install: apt-get install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        print("   ⚠️  FFmpeg check timed out")
        return False


def check_project_setup():
    """Check if project structure is correct"""
    print("\n4️⃣  Checking project setup...")
    import os
    
    required_files = [
        'app/utils/gpu_video_io.py',
        'app/pipelines/gpu_stream_pipeline.py',
        'app/pipelines/auto_pipeline.py',
        'install_pynvcodec_colab.py'
    ]
    
    all_found = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} not found")
            all_found = False
    
    return all_found


def check_gpu_pipeline():
    """Check if GPU pipeline can be imported"""
    print("\n5️⃣  Checking GPU pipeline import...")
    try:
        from app.pipelines import GPU_PIPELINE_AVAILABLE, AutoPipeline
        if GPU_PIPELINE_AVAILABLE:
            print(f"   ✅ GPU pipeline available")
            return True
        else:
            print(f"   ⚠️  GPU pipeline not available (PyNvCodec missing)")
            return False
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False


def test_simple_tensor_ops():
    """Test basic tensor operations"""
    print("\n6️⃣  Testing GPU tensor operations...")
    try:
        import torch
        
        # Create test tensor on GPU
        tensor = torch.randn(3, 640, 480).cuda()
        
        # Crop
        cropped = tensor[:, 100:300, 100:300]
        
        # Resize
        resized = torch.nn.functional.interpolate(
            cropped.unsqueeze(0),
            size=(224, 224),
            mode='bilinear'
        ).squeeze(0)
        
        print(f"   ✅ Tensor operations working")
        print(f"      Original: {tensor.shape}")
        print(f"      Cropped: {cropped.shape}")
        print(f"      Resized: {resized.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Tensor operations failed: {e}")
        return False


def print_summary(results):
    """Print summary and recommendations"""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✅ All checks passed! Your system is ready for GPU pipeline.")
        print("\nNext steps:")
        print("  1. Run: python example_gpu_usage.py")
        print("  2. Expected FPS: 60+ on 720p video (T4 GPU)")
        print("  3. Read: GPU_PIPELINE_GUIDE.md for details")
    else:
        print("❌ Some checks failed. See recommendations below:\n")
        
        if not results['cuda']:
            print("📌 CUDA not available:")
            print("   - Enable GPU in Colab: Runtime → Change runtime type → GPU")
            print("   - Or install CUDA locally: https://developer.nvidia.com/cuda-downloads")
        
        if not results['pynvcodec']:
            print("\n📌 PyNvCodec not installed:")
            print("   - Run: python install_pynvcodec_colab.py (for Colab)")
            print("   - Or: bash install_pynvcodec.sh (for Linux)")
        
        if not results['ffmpeg']:
            print("\n📌 FFmpeg not installed:")
            print("   - Run: apt-get install ffmpeg")
        
        if not results['project']:
            print("\n📌 Project files missing:")
            print("   - Ensure you've copied all GPU pipeline files")
        
        print("\n📖 For detailed setup guide, see: GPU_PIPELINE_GUIDE.md")
    
    print("="*70)
    
    return all_passed


def main():
    print("="*70)
    print("GPU Pipeline Setup Verification")
    print("="*70)
    
    results = {
        'cuda': check_cuda(),
        'pynvcodec': check_pynvcodec(),
        'ffmpeg': check_ffmpeg(),
        'project': check_project_setup(),
        'gpu_pipeline': check_gpu_pipeline(),
        'tensor_ops': test_simple_tensor_ops()
    }
    
    all_ok = print_summary(results)
    
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

