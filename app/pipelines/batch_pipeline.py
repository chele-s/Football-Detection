import cv2
import numpy as np
import json
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
from collections import deque

from app.inference import BallDetector
from app.tracking import BallTracker
from app.camera import VirtualCamera
from app.utils import VideoReader, VideoWriter

logger = logging.getLogger(__name__)


class BatchPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        
        logger.info("Initializing BatchPipeline")
        
        self.detector = BallDetector(
            model_path=config['model']['path'],
            confidence_threshold=config['model']['confidence'],
            device=config['model'].get('device', 'cuda'),
            half_precision=config['model'].get('half_precision', True),
            imgsz=config['model'].get('imgsz', 640)
        )
        
        model_info = self.detector.get_model_info()
        logger.info(f"Detector initialized: {model_info}")
        
        self.tracker = BallTracker(
            max_lost_frames=config['tracking']['max_lost_frames'],
            min_confidence=config['tracking']['min_confidence'],
            adaptive_noise=True
        )
        
        self.virtual_camera = None
        
        self.ball_class_id = config['model'].get('ball_class_id', 0)
        self.save_tracking_data = config.get('save_tracking_data', True)
        self.save_visualizations = config.get('save_visualizations', False)
        
        self.stats = {
            'total_frames': 0,
            'detected_frames': 0,
            'tracked_frames': 0,
            'lost_frames': 0,
            'processing_times': deque(maxlen=1000),
            'start_time': None,
            'end_time': None
        }
    
    def _setup_logging(self):
        log_level = logging.DEBUG if self.config.get('debug', False) else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        tracking_output_path: Optional[str] = None
    ):
        self._validate_input(input_path)
        
        logger.info(f"Processing video: {input_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        reader = VideoReader(input_path)
        logger.info(f"Video loaded: {reader.width}x{reader.height} @ {reader.fps}fps, {reader.total_frames} frames")
        
        self.virtual_camera = VirtualCamera(
            frame_width=reader.width,
            frame_height=reader.height,
            output_width=self.config['output']['width'],
            output_height=self.config['output']['height'],
            dead_zone_percent=self.config['camera']['dead_zone'],
            anticipation_factor=self.config['camera']['anticipation'],
            zoom_padding=self.config['camera']['zoom_padding'],
            use_pid=True,
            prediction_steps=5
        )
        
        self._prepare_output_directory(output_path)
        
        writer = VideoWriter(
            output_path,
            width=self.config['output']['width'],
            height=self.config['output']['height'],
            fps=reader.fps,
            codec=self.config['output'].get('codec', 'mp4v')
        )
        
        tracking_data = []
        frame_idx = 0
        self.stats['start_time'] = time.time()
        
        pbar = tqdm(
            total=reader.total_frames,
            desc="Processing",
            unit="frame",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        try:
            while True:
                frame_start = time.time()
                
                ret, frame = reader.read()
                if not ret:
                    break
                
                detection = self.detector.predict_ball_only(frame, self.ball_class_id)
                
                track_result = self.tracker.update(detection)
                
                self.stats['total_frames'] += 1
                
                if track_result is not None:
                    x, y, is_detected = track_result
                    self.stats['tracked_frames'] += 1
                    
                    if is_detected:
                        self.stats['detected_frames'] += 1
                    
                    velocity = self.tracker.get_velocity()
                    x1, y1, x2, y2 = self.virtual_camera.update(x, y, velocity_hint=velocity)
                    
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.shape[:2] != (self.config['output']['height'], self.config['output']['width']):
                        cropped = cv2.resize(
                            cropped,
                            (self.config['output']['width'], self.config['output']['height'])
                        )
                    
                    if self.save_visualizations:
                        cropped = self._add_visualization_overlay(cropped, track_result, frame_idx)
                    
                    writer.write(cropped)
                    
                    if self.save_tracking_data:
                        tracker_state = self.tracker.get_state()
                        tracking_data.append({
                            'frame': frame_idx,
                            'ball_x': float(x),
                            'ball_y': float(y),
                            'detected': is_detected,
                            'confidence': tracker_state['avg_confidence'],
                            'velocity': list(velocity),
                            'velocity_magnitude': tracker_state['velocity_magnitude'],
                            'crop': [int(x1), int(y1), int(x2), int(y2)]
                        })
                else:
                    self.stats['lost_frames'] += 1
                    x1, y1, x2, y2 = self.virtual_camera.get_current_crop()
                    cropped = frame[y1:y2, x1:x2]
                    
                    if cropped.shape[:2] != (self.config['output']['height'], self.config['output']['width']):
                        cropped = cv2.resize(
                            cropped,
                            (self.config['output']['width'], self.config['output']['height'])
                        )
                    
                    writer.write(cropped)
                
                frame_time = (time.time() - frame_start) * 1000
                self.stats['processing_times'].append(frame_time)
                
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    avg_time = np.mean(self.stats['processing_times'])
                    pbar.set_postfix({
                        'ms/frame': f'{avg_time:.1f}',
                        'detected': f"{self.stats['detected_frames']}/{frame_idx}"
                    })
                
                pbar.update(1)
        
        finally:
            pbar.close()
            reader.release()
            writer.release()
            self.stats['end_time'] = time.time()
        
        self._save_outputs(tracking_data, output_path, tracking_output_path)
        self._log_final_statistics()
    
    def _validate_input(self, input_path: str):
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {input_path}")
        logger.info(f"Input validation passed: {input_path}")
    
    def _prepare_output_directory(self, output_path: str):
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Output directory prepared: {output_dir}")
    
    def _add_visualization_overlay(self, frame: np.ndarray, track_result, frame_idx: int) -> np.ndarray:
        x, y, is_detected = track_result
        status = "DETECTED" if is_detected else "TRACKED"
        color = (0, 255, 0) if is_detected else (0, 255, 255)
        
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def _save_outputs(self, tracking_data: List[Dict], output_path: str, tracking_output_path: Optional[str]):
        if self.save_tracking_data and tracking_output_path:
            Path(tracking_output_path).parent.mkdir(parents=True, exist_ok=True)
            
            metadata = {
                'video_output': str(output_path),
                'total_frames': self.stats['total_frames'],
                'detected_frames': self.stats['detected_frames'],
                'tracked_frames': self.stats['tracked_frames'],
                'lost_frames': self.stats['lost_frames'],
                'detection_rate': self.stats['detected_frames'] / self.stats['total_frames'] if self.stats['total_frames'] > 0 else 0,
                'tracking_rate': self.stats['tracked_frames'] / self.stats['total_frames'] if self.stats['total_frames'] > 0 else 0
            }
            
            output_data = {
                'metadata': metadata,
                'tracking_data': tracking_data
            }
            
            with open(tracking_output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Tracking data saved to: {tracking_output_path}")
        
        logger.info(f"Video saved to: {output_path}")
    
    def _log_final_statistics(self):
        logger.info("="*60)
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info("="*60)
        
        total_time = self.stats['end_time'] - self.stats['start_time']
        fps = self.stats['total_frames'] / total_time if total_time > 0 else 0
        
        logger.info(f"Total frames: {self.stats['total_frames']}")
        logger.info(f"Detected frames: {self.stats['detected_frames']} ({self.stats['detected_frames']/self.stats['total_frames']*100:.1f}%)")
        logger.info(f"Tracked frames: {self.stats['tracked_frames']} ({self.stats['tracked_frames']/self.stats['total_frames']*100:.1f}%)")
        logger.info(f"Lost frames: {self.stats['lost_frames']} ({self.stats['lost_frames']/self.stats['total_frames']*100:.1f}%)")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Average FPS: {fps:.2f}")
        
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            min_time = np.min(self.stats['processing_times'])
            max_time = np.max(self.stats['processing_times'])
            logger.info(f"Avg processing time: {avg_time:.2f}ms/frame")
            logger.info(f"Min/Max processing time: {min_time:.2f}ms / {max_time:.2f}ms")
        
        detector_stats = self.detector.get_stats()
        logger.info(f"Detector stats: {detector_stats}")
        
        tracker_stats = self.tracker.get_stats()
        logger.info(f"Tracker stats: {tracker_stats}")
        
        camera_stats = self.virtual_camera.get_stats()
        logger.info(f"Camera stats: {camera_stats}")
        
        logger.info("="*60)
