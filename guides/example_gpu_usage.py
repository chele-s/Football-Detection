"""
Example: Using GPU-accelerated pipeline

This example shows how to use the GPU pipeline for maximum performance.
Expected: 3-5x faster than CPU pipeline (60+ FPS on 720p with T4 GPU)
"""

import logging
from app.utils import load_config
from app.pipelines import AutoPipeline, GPU_PIPELINE_AVAILABLE

# Setup logging to see performance metrics
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


def main():
    """Run GPU-accelerated pipeline"""
    
    logger.info("="*70)
    logger.info("GPU-Accelerated Pipeline Example")
    logger.info("="*70)
    
    # Check GPU availability
    if not GPU_PIPELINE_AVAILABLE:
        logger.warning("⚠️  GPU pipeline not available!")
        logger.warning("   Install PyNvCodec: python install_pynvcodec_colab.py")
        logger.warning("   Falling back to CPU pipeline...")
    else:
        logger.info("✓ GPU pipeline available")
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # ===== OPTION 1: Auto-detect (Recommended) =====
    # Automatically uses GPU pipeline if available, otherwise CPU
    logger.info("\n--- Using AutoPipeline (auto-detect) ---")
    pipeline = AutoPipeline(config, prefer_gpu=True)
    
    # ===== OPTION 2: Force GPU pipeline =====
    # Uncomment to explicitly use GPU (will error if PyNvCodec not installed)
    # from app.pipelines import GPUStreamPipeline
    # logger.info("\n--- Using GPUStreamPipeline (explicit) ---")
    # pipeline = GPUStreamPipeline(config)
    
    # ===== OPTION 3: Force CPU pipeline =====
    # Uncomment to explicitly use CPU (for comparison/debugging)
    # from app.pipelines import StreamPipeline
    # logger.info("\n--- Using StreamPipeline (CPU) ---")
    # pipeline = StreamPipeline(config)
    
    # Configure input/output
    # Examples:
    input_source = "path/to/your/video.mp4"
    # input_source = "rtsp://your-ip:port/stream"
    # input_source = "https://www.youtube.com/watch?v=..."
    
    output_destination = "output/result.mp4"
    # output_destination = "rtmp://your-server/live/stream"
    
    logger.info(f"\nInput:  {input_source}")
    logger.info(f"Output: {output_destination}")
    logger.info("")
    
    # Run pipeline
    try:
        pipeline.run(
            input_source=input_source,
            output_destination=output_destination
        )
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopped by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        raise
    
    logger.info("\n✅ Pipeline completed!")


def benchmark_comparison():
    """
    Compare CPU vs GPU pipeline performance.
    
    This will run both pipelines on the same video and compare FPS.
    """
    from app.pipelines import StreamPipeline, GPUStreamPipeline
    import time
    
    config = load_config('config/config.yaml')
    test_video = "path/to/test/video.mp4"
    
    logger.info("="*70)
    logger.info("CPU vs GPU Pipeline Benchmark")
    logger.info("="*70)
    
    # Test CPU pipeline
    logger.info("\n1️⃣  Testing CPU pipeline...")
    cpu_pipeline = StreamPipeline(config)
    
    start_time = time.time()
    cpu_pipeline.run(test_video, output_destination=None)  # No output, just benchmark
    cpu_time = time.time() - start_time
    
    cpu_fps = cpu_pipeline.performance_stats['total_frames'] / cpu_time
    
    logger.info(f"✓ CPU Pipeline: {cpu_fps:.1f} FPS")
    
    # Test GPU pipeline
    if GPU_PIPELINE_AVAILABLE:
        logger.info("\n2️⃣  Testing GPU pipeline...")
        gpu_pipeline = GPUStreamPipeline(config)
        
        start_time = time.time()
        gpu_pipeline.run(test_video, output_destination=None)
        gpu_time = time.time() - start_time
        
        gpu_fps = gpu_pipeline.performance_stats['total_frames'] / gpu_time
        
        logger.info(f"✓ GPU Pipeline: {gpu_fps:.1f} FPS")
        
        # Results
        speedup = gpu_fps / cpu_fps
        logger.info("\n" + "="*70)
        logger.info("RESULTS")
        logger.info("="*70)
        logger.info(f"CPU:     {cpu_fps:.1f} FPS")
        logger.info(f"GPU:     {gpu_fps:.1f} FPS")
        logger.info(f"Speedup: {speedup:.2f}x faster 🚀")
        logger.info("="*70)
    else:
        logger.warning("\n⚠️  GPU pipeline not available for comparison")


if __name__ == "__main__":
    # Run main example
    main()
    
    # Uncomment to run benchmark comparison
    # benchmark_comparison()

