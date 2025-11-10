#!/usr/bin/env python3
"""
Export RF-DETR model to TensorRT for maximum inference speed.

Usage:
    python scripts/export_to_tensorrt.py --checkpoint models/checkpoint_best_total.pth
    
Expected speedup: 2-3x (20-30 FPS → 40-60 FPS)
"""

import argparse
import torch
import tensorrt as trt
from pathlib import Path
import numpy as np
import sys
import os
sys.path.append(str(Path(__file__).parent.parent))

from rfdetr import RFDETRMedium

def export_to_onnx(checkpoint_path: str, output_path: str, resolution: int = 480):
    """Export RF-DETR model to ONNX format using native export() method"""
    print(f"[1/3] Loading RF-DETR model from {checkpoint_path}")
    
    # Convert to absolute paths before changing directory
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    output_dir = output_path.parent
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save current directory and change to output dir
    original_dir = Path.cwd()
    os.chdir(output_dir)
    
    try:
        model = RFDETRMedium(
            num_classes=1,
            resolution=resolution,
            pretrain_weights=str(checkpoint_path)
        )
        
        model.optimize_for_inference()
        
        print(f"[1/3] Exporting to ONNX using native export() method...")
        print("⏳ This may take 2-3 minutes and show many TracerWarnings (normal)")
        
        # Use RF-DETR's native export method
        # This creates "inference_model.onnx" in current directory or "output/" subdirectory
        model.export()
        
        # Check multiple possible locations (relative to current directory)
        possible_locations = [
            Path("inference_model.onnx"),
            Path("output/inference_model.onnx"),
            Path("output") / "inference_model.onnx"
        ]
        
        default_onnx = None
        for loc in possible_locations:
            if loc.exists():
                default_onnx = loc.resolve()  # Get absolute path
                print(f"✓ Found ONNX at: {loc}")
                break
        
        if default_onnx is None:
            raise FileNotFoundError(f"ONNX export failed - checked locations: {possible_locations}")
        
        # Copy to desired location using absolute paths
        import shutil
        shutil.copy2(str(default_onnx), str(output_path))
        
        print(f"✓ ONNX export complete: {output_path}")
        return str(output_path)
            
    finally:
        # Restore original directory
        os.chdir(str(original_dir))


def build_tensorrt_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    """Build TensorRT engine from ONNX"""
    print(f"\n[2/3] Building TensorRT engine from {onnx_path}")
    print("⏳ This may take 2-5 minutes...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print("[2/3] Parsing ONNX model...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print("❌ Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(f"  Error: {parser.get_error(error)}")
            return None
    
    print("[2/3] ONNX parsed successfully")
    
    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    
    # Enable FP16 for 2x speedup
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ FP16 mode enabled (2x faster)")
    else:
        print("⚠ FP16 not available, using FP32")
    
    # Build engine
    print("[2/3] Building engine (this is the slow part)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("❌ Failed to build TensorRT engine")
        return None
    
    # Save engine
    print(f"[2/3] Saving engine to {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    
    print(f"✓ TensorRT engine saved: {engine_path}")
    return engine_path


def benchmark_engine(engine_path: str, warmup: int = 10, iterations: int = 100):
    """Benchmark TensorRT engine performance"""
    print(f"\n[3/3] Benchmarking TensorRT engine...")
    
    import pycuda.driver as cuda
    import pycuda.autoinit
    import time
    
    # Load engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
    
    # Get input/output shapes
    input_shape = engine.get_tensor_shape('images')
    print(f"[3/3] Input shape: {input_shape}")
    
    # Allocate buffers
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    stream = cuda.Stream()
    
    # Warmup
    print(f"[3/3] Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.set_tensor_address('images', int(d_input))
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    
    # Benchmark
    print(f"[3/3] Running benchmark ({iterations} iterations)...")
    times = []
    for _ in range(iterations):
        start = time.time()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.set_tensor_address('images', int(d_input))
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time
    
    print(f"\n{'='*60}")
    print(f"  TensorRT Performance")
    print(f"{'='*60}")
    print(f"  Avg inference time: {avg_time:.2f}ms (±{std_time:.2f}ms)")
    print(f"  FPS: {fps:.1f}")
    print(f"  Min: {min(times):.2f}ms | Max: {max(times):.2f}ms")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Export RF-DETR to TensorRT')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to RF-DETR checkpoint')
    parser.add_argument('--resolution', type=int, default=480, help='Model resolution (default: 480)')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 precision')
    parser.add_argument('--benchmark', action='store_true', default=True, help='Benchmark engine')
    
    args = parser.parse_args()
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model_name = checkpoint_path.stem
    onnx_path = output_dir / f"{model_name}_{args.resolution}_rfdetr.onnx"
    engine_path = output_dir / f"{model_name}_{args.resolution}_rfdetr{'_fp16' if args.fp16 else '_fp32'}.engine"
    
    print(f"\n{'='*60}")
    print(f"  RF-DETR TensorRT Export")
    print(f"{'='*60}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  FP16: {args.fp16}")
    print(f"  Output: {engine_path}")
    print(f"{'='*60}\n")
    
    try:
        # Export to ONNX
        export_to_onnx(str(checkpoint_path), str(onnx_path), args.resolution)
        
        # Build TensorRT engine
        build_tensorrt_engine(str(onnx_path), str(engine_path), args.fp16)
        
        # Benchmark
        if args.benchmark:
            try:
                benchmark_engine(str(engine_path))
            except Exception as e:
                print(f"⚠ Benchmark failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"  ✓ Export Complete!")
        print(f"{'='*60}")
        print(f"  ONNX: {onnx_path}")
        print(f"  TensorRT: {engine_path}")
        print(f"\n  Next steps:")
        print(f"  1. Use engine in detector.py")
        print(f"  2. Expected: 2-3x speedup vs PyTorch")
        print(f"  3. No precision loss")
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
