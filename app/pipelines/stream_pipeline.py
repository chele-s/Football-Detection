import cv2
import numpy as np
import time
import logging
import os
import math
from typing import Optional, Dict, Any
from collections import deque

from app.inference import BallDetector
from app.tracking import BallTracker
from app.camera import VirtualCamera
from app.utils import VideoReader, FFMPEGWriter, RTMPClient

logger = logging.getLogger(__name__)

class SmoothZoom:
    def __init__(self, min_zoom: float = 1.0, max_zoom: float = 2.5, stiffness: float = 0.08, damping: float = 0.70, max_rate: float = 0.08):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.k = stiffness
        self.c = damping
        self.max_rate = max_rate
        self.z = min_zoom
        self.v = 0.0
        self.target = min_zoom
    def set_target(self, target: float):
        self.target = max(self.min_zoom, min(self.max_zoom, target))
    def update(self) -> float:
        e = self.target - self.z
        a = self.k * e - self.c * self.v
        self.v += a
        if self.v > self.max_rate:
            self.v = self.max_rate
        elif self.v < -self.max_rate:
            self.v = -self.max_rate
        self.z += self.v
        if self.z < self.min_zoom:
            self.z = self.min_zoom
            self.v = 0.0
        elif self.z > self.max_zoom:
            self.z = self.max_zoom
            self.v = 0.0
        return self.z

def has_display() -> bool:
    """Check if display is available for cv2.imshow"""
    if os.environ.get('DISPLAY') is None and os.name != 'nt':
        return False
    try:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('test')
        return True
    except:
        return False


class StreamPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_logging()
        self.display_available = has_display()
        
        logger.info("Initializing StreamPipeline")
        
        if not self.display_available:
            logger.info("No display detected - running in headless mode (cv2.imshow disabled)")
        
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
            iou_threshold=config['tracking'].get('iou_threshold', 0.3),
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
            smoothing_min_cutoff=self.config['camera'].get('smoothing_min_cutoff', 0.6),
            smoothing_beta=self.config['camera'].get('smoothing_beta', 0.004),
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
        camera_initialized = False
        current_zoom_level = 1.0
        target_zoom_level = 1.0
        max_zoom_level = 1.7
        frames_tracking = 0
        lost_search_center = None
        zoom = SmoothZoom(min_zoom=1.0, max_zoom=max_zoom_level)
        diag = int(math.hypot(reader.width, reader.height))
        prev_crop = None
        roi_active = False
        roi_stable_frames = 0
        roi_ready_frames = 45
        roi_fail_count = 0
        roi_fail_max = 90
        roi_last_valid_pos = None
        
        logger.info("Starting main loop... (Press 'q' to quit)")
        print(f"\n{'='*60}")
        print(f"📐 Frame resolution: {reader.width}x{reader.height}")
        print(f"🚫 DEAD ZONE 1: x>{reader.width*0.70:.0f} AND y<{reader.height*0.40:.0f} (top-right)")
        print(f"🚫 DEAD ZONE 2: x<{reader.width*0.08:.0f} AND y<{reader.height*0.40:.0f} (top-left)")
        print(f"{'='*60}\n")
        
        try:
            while True:
                loop_start = time.time()
                
                ret, frame = reader.read()
                if not ret:
                    logger.info("End of stream")
                    break
                
                if True:
                    t_inference_start = time.time()
                    use_roi = False
                    offx, offy = 0, 0
                    if prev_crop is not None and roi_active:
                        rx1, ry1, rx2, ry2 = prev_crop
                        rx1 = max(0, min(reader.width-2, int(rx1)))
                        ry1 = max(0, min(reader.height-2, int(ry1)))
                        rx2 = max(rx1+2, min(reader.width, int(rx2)))
                        ry2 = max(ry1+2, min(reader.height, int(ry2)))
                        frame_in = frame[ry1:ry2, rx1:rx2]
                        use_roi = True
                        offx, offy = rx1, ry1
                        det_result = self.detector.predict_ball_only(frame_in, self.ball_class_id, use_temporal_filtering=False, return_candidates=True)
                    else:
                        det_result = self.detector.predict_ball_only(frame, self.ball_class_id, return_candidates=True)
                    t_inference = (time.time() - t_inference_start) * 1000
                    self.performance_stats['inference_times'].append(t_inference)
                if isinstance(det_result, tuple) and len(det_result) == 2 and (
                    det_result[0] is None or isinstance(det_result[0], tuple)
                ):
                    detection, detections_list = det_result
                else:
                    detection = det_result
                    detections_list = None
                
                dead_zones = [
                    {'x1': reader.width * 0.70, 'y1': 0, 'x2': reader.width, 'y2': reader.height * 0.40, 'name': 'top-right'},
                    {'x1': 0, 'y1': 0, 'x2': reader.width * 0.08, 'y2': reader.height * 0.40, 'name': 'top-left'},
                ]
                
                if detection is not None:
                    dx, dy, dw, dh, dconf = detection
                    if frame_count % 30 == 0:
                        print(f"🎯 RAW DETECTION: x={dx:.0f}, y={dy:.0f}, conf={dconf:.2f}")
                    
                    for zone in dead_zones:
                        if zone['x1'] <= dx <= zone['x2'] and zone['y1'] <= dy <= zone['y2']:
                            print(f"❌ DEAD ZONE '{zone['name']}': x={dx:.0f}, y={dy:.0f} BLOCKED")
                            detection = None
                            break
                
                if detections_list:
                    original_count = len(detections_list)
                    filtered_detections = []
                    for d in detections_list:
                        dx, dy = d[0], d[1]
                        in_dead_zone = False
                        for zone in dead_zones:
                            if zone['x1'] <= dx <= zone['x2'] and zone['y1'] <= dy <= zone['y2']:
                                in_dead_zone = True
                                break
                        if not in_dead_zone:
                            filtered_detections.append(d)
                    detections_list = filtered_detections if filtered_detections else None
                    
                    if original_count != len(filtered_detections) and frame_count % 30 == 0:
                        print(f"🚫 Dead zones filtered {original_count - len(filtered_detections)} candidates")
                if use_roi:
                    if detection is not None:
                        bx, by, bw, bh, bc = detection
                        detection = (bx + offx, by + offy, bw, bh, bc)
                    if detections_list:
                        mapped = []
                        for d in detections_list:
                            mapped.append((d[0] + offx, d[1] + offy, d[2], d[3], d[4], d[5]))
                        detections_list = mapped
                    # Check if detection is viable
                    detection_viable = False
                    if detection is not None:
                        bx, by, bw, bh, bconf = detection
                        # Check for erratic movement
                        if roi_last_valid_pos is not None:
                            dx = bx - roi_last_valid_pos[0]
                            dy = by - roi_last_valid_pos[1]
                            jump_dist = math.hypot(dx, dy)
                            # Viable if movement is reasonable (not teleporting)
                            detection_viable = jump_dist < diag * 0.3
                        else:
                            detection_viable = True
                        
                        # Check for multiple conflicting detections
                        if detection_viable and detections_list and len(detections_list) > 3:
                            # Too many detections = confusion
                            detection_viable = False
                        
                    if detection_viable:
                        roi_last_valid_pos = (bx, by)
                        roi_fail_count = 0
                    else:
                        roi_fail_count += 1
                    
                    if roi_fail_count >= roi_fail_max:
                        logger.info(f"ROI unstable - switching to full-frame")
                        roi_active = False
                        roi_fail_count = 0
                        roi_last_valid_pos = None
                
                # Spatial filter: exclude stands/lights regions (top + sides)
                # ONLY apply when ROI is NOT active AND zoom < 1.4
                # When zoomed in or ROI active, we're already focused on playing field
                apply_spatial_filter = (not roi_active) and (current_zoom_level < 1.2) and (frames_tracking < 10)
                
                if apply_spatial_filter:
                    if detection is not None:
                        bx, by, bw, bh, bconf = detection
                        if by < reader.height * 0.20 or bx < reader.width * 0.10 or bx > reader.width * 0.90:
                            detection = None
                    
                    if detections_list:
                        filtered_dets = []
                        for d in detections_list:
                            dx, dy = d[0], d[1]
                            if dy >= reader.height * 0.20 and dx >= reader.width * 0.10 and dx <= reader.width * 0.90:
                                filtered_dets.append(d)
                        detections_list = filtered_dets if filtered_dets else None
                
                if detections_list and len(detections_list) >= 4:
                    if detection is not None:
                        det_conf = detection[4]
                        other_high_conf = [d for d in detections_list if len(d) >= 5 and d[4] > det_conf * 0.75]
                        
                        if len(other_high_conf) >= 2:
                            if frame_count % 30 == 0:
                                print(f"⚠ Multiple candidates: {len(detections_list)} total, {len(other_high_conf)} high-conf - REJECTING ALL")
                            detection = None
                            detections_list = None
                
                t_tracking_start = time.time()
                track_result = self.tracker.update(detection, detections_list)
                t_tracking = (time.time() - t_tracking_start) * 1000
                self.performance_stats['tracking_times'].append(t_tracking)
                
                t_camera_start = time.time()
                
                if not camera_initialized and track_result:
                    x, y, is_tracking = track_result
                    self.virtual_camera.reset()
                    crop_coords = self.virtual_camera.update(x, y, time.time(), velocity_hint=self.tracker.get_velocity(), detector_stable=True)
                    camera_initialized = True
                elif track_result:
                    x, y, is_tracking = track_result
                    
                    if not is_tracking:
                        roi_active = False
                        roi_fail_count = 0
                        roi_last_valid_pos = None
                        frames_tracking = 0
                    else:
                        frames_tracking += 1
                    
                    state = self.tracker.get_state()
                    vmag = state['velocity_magnitude'] if state else 0.0
                    
                    detector_stable = True
                    if state and 'stats' in state:
                        total = state['stats'].get('total_updates', 1)
                        erratic = state['stats'].get('erratic_detections', 0)
                        if total > 0:
                            erratic_rate = erratic / total
                            detector_stable = erratic_rate < 0.08
                    
                    use_x, use_y = x, y
                    
                    if is_tracking:
                        lost_search_center = None
                    
                    crop_half_h = self.virtual_camera.effective_height // 2
                    crop_half_w = self.virtual_camera.effective_width // 2
                    
                    safe_x = max(crop_half_w + 20, min(use_x, reader.width - crop_half_w - 20))
                    safe_y = max(crop_half_h + 20, min(use_y, reader.height - crop_half_h - 20))
                    
                    if not is_tracking:
                        prev_y = self.virtual_camera.current_center_y
                        min_allowed_y = int(reader.height * 0.35)
                        safe_y = max(min_allowed_y, prev_y)
                    
                    vhx, vhy = self.tracker.get_velocity()
                    crop_coords = self.virtual_camera.update(safe_x, safe_y, time.time(), velocity_hint=(vhx, vhy), detector_stable=detector_stable)
                    
                    if is_tracking and frames_tracking >= 8:
                        if vmag > 900:
                            target_zoom_level = 1.15
                        elif vmag > 600:
                            target_zoom_level = 1.30
                        elif vmag > 350:
                            target_zoom_level = 1.45
                        else:
                            target_zoom_level = 1.60
                        
                        if roi_stable_frames < roi_ready_frames:
                            roi_stable_frames += 1
                        
                        if not roi_active and roi_stable_frames >= roi_ready_frames:
                            roi_active = True
                            roi_fail_count = 0
                            roi_last_valid_pos = None
                    else:
                        target_zoom_level = 1.0
                        roi_stable_frames = max(0, roi_stable_frames - 2)
                else:
                    frames_tracking = 0
                    target_zoom_level = 1.0
                    roi_active = False
                    roi_stable_frames = 0
                    
                    if lost_search_center is None:
                        current_crop = self.virtual_camera.get_current_crop()
                        cx = (current_crop[0] + current_crop[2]) // 2
                        cy = (current_crop[1] + current_crop[3]) // 2
                        lost_search_center = (cx, cy)
                    
                    search_target_x = int(reader.width * 0.50)
                    search_target_y = int(reader.height * 0.40)
                    
                    lost_search_center = (
                        int(lost_search_center[0] * 0.96 + search_target_x * 0.04),
                        int(lost_search_center[1] * 0.96 + search_target_y * 0.04)
                    )
                    
                    crop_coords = self.virtual_camera.update(lost_search_center[0], lost_search_center[1], time.time(), velocity_hint=(0, 0), detector_stable=True)
                zoom.set_target(target_zoom_level)
                current_zoom_level = zoom.update()
                
                x1, y1, x2, y2 = crop_coords
                
                if current_zoom_level > 1.01:
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    zoomed_width = int(crop_width / current_zoom_level)
                    zoomed_height = int(crop_height / current_zoom_level)
                    
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    x1 = max(0, cx - zoomed_width // 2)
                    y1 = max(0, cy - zoomed_height // 2)
                    x2 = min(reader.width, x1 + zoomed_width)
                    y2 = min(reader.height, y1 + zoomed_height)
                    
                    if x2 - x1 < zoomed_width:
                        x1 = max(0, x2 - zoomed_width)
                    if y2 - y1 < zoomed_height:
                        y1 = max(0, y2 - zoomed_height)
                
                x1 = max(0, min(reader.width - 2, int(x1)))
                y1 = max(0, min(reader.height - 2, int(y1)))
                x2 = max(x1 + 2, min(reader.width, int(x2)))
                y2 = max(y1 + 2, min(reader.height, int(y2)))
                
                prev_crop = (x1, y1, x2, y2)
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
                    
                    status = "DETECTED" if (track_result and is_tracking) else "PREDICTED" if track_result else "LOST"
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
                
                if self.debug_mode and self.display_available:
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
            if self.debug_mode and self.display_available:
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
