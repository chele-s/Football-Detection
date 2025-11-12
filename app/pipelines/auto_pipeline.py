"""
Auto-detect and select optimal pipeline (GPU vs CPU)
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def create_pipeline(config: Dict[str, Any], prefer_gpu: bool = True):
    """
    Auto-detect and create optimal pipeline.
    
    Args:
        config: Pipeline configuration
        prefer_gpu: If True, use GPU pipeline when available
    
    Returns:
        StreamPipeline or GPUStreamPipeline instance
    """
    
    # Check if GPU pipeline is requested and available
    use_gpu = False
    
    if prefer_gpu:
        try:
            import torch
            from app.utils import GPU_AVAILABLE
            
            if GPU_AVAILABLE and torch.cuda.is_available():
                # Check if NVDEC/NVENC are accessible
                try:
                    import PyNvCodec as nvc
                    use_gpu = True
                    logger.info("✓ GPU pipeline available (PyNvCodec + CUDA)")
                except ImportError:
                    logger.warning("PyNvCodec not installed. Falling back to CPU pipeline.")
                    logger.warning("Install with: python install_pynvcodec_colab.py")
            else:
                if not torch.cuda.is_available():
                    logger.info("CUDA not available. Using CPU pipeline.")
                else:
                    logger.info("GPU video I/O not available. Using CPU pipeline.")
        
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            logger.info("Falling back to CPU pipeline")
    
    # Create appropriate pipeline
    if use_gpu:
        from app.pipelines.gpu_stream_pipeline import GPUStreamPipeline
        logger.info("="*70)
        logger.info("🚀 USING GPU-ACCELERATED PIPELINE")
        logger.info("  - NVDEC hardware video decode")
        logger.info("  - PyTorch GPU tensor operations")
        logger.info("  - NVENC hardware video encode")
        logger.info("  - Expected: 3-5x performance boost")
        logger.info("="*70)
        return GPUStreamPipeline(config)
    else:
        from app.pipelines.stream_pipeline import StreamPipeline
        logger.info("="*70)
        logger.info("Using CPU pipeline")
        logger.info("="*70)
        return StreamPipeline(config)


class AutoPipeline:
    """
    Wrapper that automatically selects optimal pipeline.
    
    Usage:
        pipeline = AutoPipeline(config, prefer_gpu=True)
        pipeline.run(input_source, output_destination)
    """
    
    def __init__(self, config: Dict[str, Any], prefer_gpu: bool = True):
        self.pipeline = create_pipeline(config, prefer_gpu)
    
    def run(self, input_source: str, output_destination: Optional[str] = None):
        """Run the pipeline"""
        return self.pipeline.run(input_source, output_destination)
    
    def __getattr__(self, name):
        """Forward all other attributes to the underlying pipeline"""
        return getattr(self.pipeline, name)

