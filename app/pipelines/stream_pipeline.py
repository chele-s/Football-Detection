import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
from collections import deque

from app.inference import BallDetector
from app.tracking import BallTracker
from app.camera import VirtualCamera
from app.utils import VideoReader, FFMPEGWriter, RTMPClient

logger = logging.getLogger(__name__)


class StreamPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        
        logger.info("Initializing StreamPipeline")
        
        self.detector = BallDetector(
            model_path=config['model']['path'],
            confidence_threshold=config['model']['confidence'],
            iou_threshold=config['model'].get('iou_threshold', 0.45),
            device=config['model'].get('device', 'cuda'),
            half_precision=config['model'].get('half_precision', True),
            imgsz=config['model'].get('imgsz', 640),
            multi_scale=config['model'].get('multi_scale', False),
            warmup_iterations=config['model'].get('warmup_iterations', 3)
        )
        
        model_info = self.detector.get_model_info()
        logger.info(f"Model loaded: {model_info}")
        
        self.tracker = BallTracker(
            max_lost_frames=config['tracking']['max_lost_frames'],
            min_confidence=config['tracking']['min_confidence'],
            adaptive_noise=True
        )
        
        self.virtual_camera = None
        
        self.ball_class_id = config['model'].get('ball_class_id', 0)
        
        self.fps_history = deque(maxlen=30)
        self.target_fps = config['stream'].get('target_fps', 30)
        self.show_stats = config['stream'].get('show_stats', True)
        self.debug_mode = config['stream'].get('debug_mode', False)
        
        self.performance_stats = {
            'inference_times': deque(maxlen=100),
            'tracking_times': deque(maxlen=100),
            'camera_times': deque(maxlen=100),
            'total_frames': 0
        }
    
    def _setup_logging(self):
        log_level = logging.DEBUG if self.config.get('stream', {}).get('debug_mode', False) else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    def run(self, input_source: str, output_destination: Optional[str] = None):
        if RTMPClient.is_youtube_url(input_source):
            logger.info("Extracting YouTube stream URL...")
            stream_url = RTMPClient.get_youtube_stream_url(input_source)
            if stream_url is None:
                raise ValueError("Could not get YouTube URL")
            input_source = stream_url
            logger.info("YouTube URL obtained successfully")
        
        logger.info(f"Connecting to input: {input_source}")
        reader = VideoReader(input_source)
        logger.info(f"Connected! Resolution: {reader.width}x{reader.height} @ {reader.fps}fps")
        
        self.virtual_camera = VirtualCamera(
            frame_width=reader.width,
            frame_height=reader.height,
            output_width=self.config['output']['width'],
            output_height=self.config['output']['height'],
            dead_zone_percent=self.config['camera']['dead_zone'],
            anticipation_factor=self.config['camera']['anticipation'],
            zoom_padding=self.config['camera']['zoom_padding'],
            smoothing_freq=self.target_fps,
            use_pid=True,
            prediction_steps=5
        )
        
        writer = None
        if output_destination and not self.debug_mode:
            logger.info(f"Starting output stream to: {output_destination}")
            writer = FFMPEGWriter(
                output_url=output_destination,
                width=self.config['output']['width'],
                height=self.config['output']['height'],
                fps=self.target_fps,
                bitrate=self.config['stream'].get('bitrate', '4000k'),
                preset=self.config['stream'].get('preset', 'ultrafast')
            )
        
        frame_count = 0
        last_time = time.time()
        
        logger.info("Starting main loop... (Press 'q' to quit)")
        
        try:
            while True:
                loop_start = time.time()
                
                ret, frame = reader.read()
                if not ret:
                    logger.info("End of stream")
                    break
                
                t_inference_start = time.time()
                det_result = self.detector.predict_ball_only(frame, self.ball_class_id, return_candidates=True)
                if isinstance(det_result, tuple) and len(det_result) == 2 and (
                    det_result[0] is None or isinstance(det_result[0], tuple)
                ):
                    detection, detections_list = det_result
                else:
                    detection = det_result
                    detections_list = None
                t_inference = (time.time() - t_inference_start) * 1000
                self.performance_stats['inference_times'].append(t_inference)
                
                t_tracking_start = time.time()
                track_result = self.tracker.update(detection, detections_list)
                t_tracking = (time.time() - t_tracking_start) * 1000
                self.performance_stats['tracking_times'].append(t_tracking)
                
                t_camera_start = time.time()
                if track_result is not None:
                    x, y, is_detected = track_result
                    velocity = self.tracker.get_velocity()
                    x1, y1, x2, y2 = self.virtual_camera.update(x, y, time.time(), velocity_hint=velocity)
                else:
                    x1, y1, x2, y2 = self.virtual_camera.get_current_crop()
                    is_detected = False
                t_camera = (time.time() - t_camera_start) * 1000
                self.performance_stats['camera_times'].append(t_camera)
                
                cropped = frame[y1:y2, x1:x2]
                
                if cropped.shape[:2] != (self.config['output']['height'], self.config['output']['width']):
                    cropped = cv2.resize(
                        cropped,
                        (self.config['output']['width'], self.config['output']['height'])
                    )
                
                if self.show_stats:
                    current_fps = 1.0 / (time.time() - loop_start) if (time.time() - loop_start) > 0 else 0
                    self.fps_history.append(current_fps)
                    avg_fps = sum(self.fps_history) / len(self.fps_history)
                    
                    status = "DETECTED" if (track_result and is_detected) else "PREDICTED" if track_result else "LOST"
                    color = (0, 255, 0) if status == "DETECTED" else (0, 255, 255) if status == "PREDICTED" else (0, 0, 255)
                    
                    avg_inference = np.mean(self.performance_stats['inference_times']) if self.performance_stats['inference_times'] else 0
                    avg_tracking = np.mean(self.performance_stats['tracking_times']) if self.performance_stats['tracking_times'] else 0
                    avg_camera = np.mean(self.performance_stats['camera_times']) if self.performance_stats['camera_times'] else 0
                    
                    cv2.putText(cropped, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(cropped, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(cropped, f"Inference: {avg_inference:.1f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(cropped, f"Tracking: {avg_tracking:.1f}ms", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    cv2.putText(cropped, f"Camera: {avg_camera:.1f}ms", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    if track_result:
                        tracker_state = self.tracker.get_state()
                        vel_mag = tracker_state['velocity_magnitude']
                        cv2.putText(cropped, f"Velocity: {vel_mag:.1f}px/s", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        cv2.putText(cropped, f"Conf: {tracker_state['avg_confidence']:.2f}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                if writer:
                    writer.write(cropped)
                
                if self.debug_mode:
                    cv2.imshow('Stream Output', cropped)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Stopped by user")
                        break
                
                frame_count += 1
                self.performance_stats['total_frames'] = frame_count
                
                if frame_count % 100 == 0:
                    elapsed = time.time() - last_time
                    fps = 100 / elapsed
                    tracker_stats = self.tracker.get_stats()
                    camera_stats = self.virtual_camera.get_stats()
                    
                    logger.info(f"Frames: {frame_count} | FPS: {fps:.2f} | "
                               f"Tracking: {self.tracker.is_tracking} | "
                               f"Predictions: {tracker_stats['predictions_used']} | "
                               f"PID corrections: {camera_stats['pid_corrections']}")
                    last_time = time.time()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            reader.release()
            if writer:
                writer.release()
            if self.debug_mode:
                cv2.destroyAllWindows()
            
            self._log_final_stats(frame_count)
    
    def _log_final_stats(self, total_frames: int):
        logger.info("="*50)
        logger.info("FINAL STATISTICS")
        logger.info("="*50)
        logger.info(f"Total frames processed: {total_frames}")
        
        if self.performance_stats['inference_times']:
            avg_inf = np.mean(self.performance_stats['inference_times'])
            logger.info(f"Avg inference time: {avg_inf:.2f}ms")
        
        if self.performance_stats['tracking_times']:
            avg_track = np.mean(self.performance_stats['tracking_times'])
            logger.info(f"Avg tracking time: {avg_track:.2f}ms")
        
        if self.performance_stats['camera_times']:
            avg_cam = np.mean(self.performance_stats['camera_times'])
            logger.info(f"Avg camera time: {avg_cam:.2f}ms")
        
        tracker_stats = self.tracker.get_stats()
        logger.info(f"Tracker stats: {tracker_stats}")
        
        camera_stats = self.virtual_camera.get_stats()
        logger.info(f"Camera stats: {camera_stats}")
        
        filter_stats = self.virtual_camera.position_filter.get_stats()
        logger.info(f"Filter stats: {filter_stats}")
        
        logger.info("="*50)
