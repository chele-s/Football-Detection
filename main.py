import argparse
import sys
import logging
import signal
import torch
from pathlib import Path
from typing import Optional

from app.utils import load_config, merge_configs
from app.pipelines import BatchPipeline, StreamPipeline

logger = logging.getLogger(__name__)

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.kill_now = True


def setup_logging(debug: bool = False):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('football_tracker.log', mode='a')
        ]
    )
    logger.info("Logging initialized")

def print_system_info():
    logger.info("="*60)
    logger.info("FOOTBALL TRACKER SYSTEM")
    logger.info("="*60)
    logger.info(f"Python version: {sys.version.split()[0]}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: YES")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    else:
        logger.info("CUDA available: NO (CPU only)")
    
    logger.info("="*60)

def validate_config_files(model_config_path: str, mode_config_path: str) -> bool:
    model_path = Path(model_config_path)
    mode_path = Path(mode_config_path)
    
    if not model_path.exists():
        logger.error(f"Model config not found: {model_config_path}")
        return False
    
    if not mode_path.exists():
        logger.error(f"Mode config not found: {mode_config_path}")
        return False
    
    return True

def run_batch_mode(config: dict, input_path: str, output_path: str):
    tracking_path = config['output'].get('tracking_path')
    
    if not Path(input_path).exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    logger.info("Starting BATCH mode")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    
    if tracking_path:
        logger.info(f"Tracking data: {tracking_path}")
    
    try:
        pipeline = BatchPipeline(config)
        pipeline.process_video(input_path, output_path, tracking_path)
        logger.info("Batch processing completed successfully")
    except Exception as e:
        logger.exception(f"Batch processing failed: {e}")
        sys.exit(1)

def run_stream_mode(config: dict, input_url: str, output_url: Optional[str]):
    logger.info("Starting STREAM mode")
    logger.info(f"Input: {input_url}")
    logger.info(f"Output: {output_url if output_url else 'PREVIEW ONLY'}")
    logger.info(f"Debug mode: {config['stream'].get('debug_mode', False)}")
    
    killer = GracefulKiller()
    
    try:
        pipeline = StreamPipeline(config)
        pipeline.run(input_url, output_url)
        
        if killer.kill_now:
            logger.info("Stream stopped by user")
        else:
            logger.info("Stream completed")
    except KeyboardInterrupt:
        logger.info("Stream interrupted by user")
    except Exception as e:
        logger.exception(f"Stream processing failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Football Tracker Pipeline - Advanced ball tracking system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Batch mode:
    python main.py batch --input video.mp4 --output output.mp4
  
  Stream mode:
    python main.py stream --input rtmp://localhost/live --output rtmp://localhost/output
    python main.py stream --input "https://youtube.com/watch?v=xxx" --debug
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['batch', 'stream'],
        help='Operation mode: batch (file processing) or stream (real-time)'
    )
    
    parser.add_argument(
        '--model-config',
        default='configs/model_config.yml',
        help='Path to model config file (default: configs/model_config.yml)'
    )
    
    parser.add_argument(
        '--input',
        help='Input path/URL (overrides config file)'
    )
    
    parser.add_argument(
        '--output',
        help='Output path/URL (overrides config file)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (shows preview window and verbose logging)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        help='Detection confidence threshold (0.0-1.0)'
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Force specific device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--no-logging-file',
        action='store_true',
        help='Disable logging to file'
    )
    
    args = parser.parse_args()
    
    setup_logging(args.debug)
    print_system_info()
    
    try:
        mode_config_path = 'configs/batch_config.yml' if args.mode == 'batch' else 'configs/stream_config.yml'
        
        if not validate_config_files(args.model_config, mode_config_path):
            sys.exit(1)
        
        logger.info(f"Loading configuration files...")
        model_config = load_config(args.model_config)
        mode_config = load_config(mode_config_path)
        config = merge_configs(model_config, mode_config)
        
        if args.confidence:
            config['model']['confidence'] = args.confidence
            logger.info(f"Confidence threshold override: {args.confidence}")
        
        if args.device:
            config['model']['device'] = args.device
            logger.info(f"Device override: {args.device}")
        
        if args.debug:
            config['debug'] = True
            if args.mode == 'stream':
                config['stream']['debug_mode'] = True
                config['stream']['show_stats'] = True
        
        if args.mode == 'batch':
            input_path = args.input or config['input']['video_path']
            output_path = args.output or config['output']['video_path']
            run_batch_mode(config, input_path, output_path)
        
        elif args.mode == 'stream':
            input_url = args.input or config['stream']['input_url']
            output_url = args.output or config['stream'].get('output_url')
            
            if args.debug:
                output_url = None
            
            run_stream_mode(config, input_url, output_url)
        
        logger.info("Application finished successfully")
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutdown requested... exiting")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)
