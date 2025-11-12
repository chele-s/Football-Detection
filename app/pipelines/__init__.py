from .batch_pipeline import BatchPipeline
from .stream_pipeline import StreamPipeline
from .auto_pipeline import AutoPipeline, create_pipeline

# GPU pipeline (optional, requires PyNvCodec)
try:
    from .gpu_stream_pipeline import GPUStreamPipeline
    GPU_PIPELINE_AVAILABLE = True
    __all__ = ['BatchPipeline', 'StreamPipeline', 'GPUStreamPipeline', 
               'AutoPipeline', 'create_pipeline', 'GPU_PIPELINE_AVAILABLE']
except ImportError:
    GPU_PIPELINE_AVAILABLE = False
    __all__ = ['BatchPipeline', 'StreamPipeline', 'AutoPipeline', 
               'create_pipeline', 'GPU_PIPELINE_AVAILABLE']
