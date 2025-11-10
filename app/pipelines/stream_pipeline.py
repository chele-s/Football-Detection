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
    def __init__(self, min_zoom: float = 1.0, max_zoom: float = 2.5, stiffness: float = 0.06, damping: float = 0.60, max_rate: float = 0.11, max_rate_in: float = 0.14, max_rate_out: float = 0.10, accel_limit: float = 0.05):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.k = stiffness
        self.c = damping
        self.max_rate = max_rate
        self.max_rate_in = max_rate_in
        self.max_rate_out = max_rate_out
        self.accel_limit = accel_limit
        self.z = min_zoom
        self.v = 0.0
        self.target = min_zoom
    def set_target(self, target: float):
        self.target = max(self.min_zoom, min(self.max_zoom, target))
    def update(self) -> float:
        e = self.target - self.z
        a = self.k * e - self.c * self.v
        if self.accel_limit is not None:
            if a > self.accel_limit:
                a = self.accel_limit
            elif a < -self.accel_limit:
                a = -self.accel_limit
        self.v += a
        if e >= 0:
            mr = self.max_rate_in if self.max_rate_in is not None else self.max_rate
            if self.v > mr:
                self.v = mr
            if self.v < -mr:
                self.v = -mr
        else:
            mr = self.max_rate_out if self.max_rate_out is not None else self.max_rate
            if self.v > mr:
                self.v = mr
            if self.v < -mr:
                self.v = -mr
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
        detection_count = 0
        lost_count = 0
        current_zoom_level = 1.0
        target_zoom_level = 1.0
        max_zoom_level = 1.8
        frames_tracking = 0
        frames_required_for_zoom = 4
        zoom = SmoothZoom(min_zoom=1.0, max_zoom=max_zoom_level)
        diag = int(math.hypot(reader.width, reader.height))
        stable_step_px = max(6, int(diag * 0.030))
        jump_reset_px = max(36, int(diag * 0.050))
        cooldown_max = 18
        cooldown = 0
        last_stable = None
        stability_score = 0.0
        anchor = None
        max_pan_step = max(8, int(diag * 0.012))
        anchor_ready_px = max(28, int(diag * 0.055))
        zoom_lock_count = 0
        zoom_lock_max = 36
        hold_zoom_level = 1.0
        last_raw = None
        last_vec = (0.0, 0.0)
        natural_counter = 0
        natural_step_px = max(10, int(diag * 0.020))
        min_dir_cos = 0.4
        recent_positions = []
        close_counter = 0
        close_thresh = max(14, int(diag * 0.050))
        last_det = None
        prev_zoom_center = None
        prev_crop = None
        max_crop_step_base = max(4, int(diag * 0.008))
        reacq_points = []
        area_min = max(64, int(0.00002 * reader.width * reader.height))
        area_max = int(0.030 * reader.width * reader.height)
        det_consist = 0
        det_consist_need = 4
        roi_active = False
        roi_stable_frames = 0
        roi_ready_frames = 35
        roi_fail_count = 0
        roi_fail_max = 120  # 4 seconds at 30fps
        roi_last_valid_pos = None
        bloom_counter = 0
        bloom_max = 12
        far_reacquire_count = 0
        far_reacquire_need = 6
        det_skip = 0
        
        logger.info("Starting main loop... (Press 'q' to quit)")
        
        try:
            while True:
                loop_start = time.time()
                
                ret, frame = reader.read()
                if not ret:
                    logger.info("End of stream")
                    break
                
                do_inference = True
                if frames_tracking >= 8 and stability_score >= 0.60 and cooldown == 0:
                    if det_skip > 0:
                        det_skip -= 1
                        do_inference = False
                    else:
                        det_skip = 1
                if do_inference:
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
                else:
                    det_result = (None, None)
                    t_inference = 0.0
                self.performance_stats['inference_times'].append(t_inference)
                if isinstance(det_result, tuple) and len(det_result) == 2 and (
                    det_result[0] is None or isinstance(det_result[0], tuple)
                ):
                    detection, detections_list = det_result
                else:
                    detection = det_result
                    detections_list = None
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
                    
                    # Only increment fail count if no viable detection
                    if not detection_viable:
                        roi_fail_count += 1
                    else:
                        roi_fail_count = 0
                    
                    if roi_fail_count >= roi_fail_max:
                        logger.info(f"ROI lost ball for {roi_fail_max} frames - switching to full-frame detection")
                        roi_active = False
                        roi_fail_count = 0
                        roi_last_valid_pos = None
                
                # Spatial filter: exclude stands/lights regions (top + sides)
                # ONLY apply when ROI is NOT active AND zoom < 1.4
                # When zoomed in or ROI active, we're already focused on playing field
                apply_spatial_filter = (not roi_active) and (current_zoom_level < 1.4)
                
                if apply_spatial_filter:
                    # TOP: Reject detections in upper 25% of frame (stands/lights)
                    # SIDES: Reject detections in outer 15% on left and right (side stands)
                    if detection is not None:
                        bx, by, bw, bh, bconf = detection
                        # Exclude top region
                        if by < reader.height * 0.25:
                            detection = None
                        # Exclude left side region
                        elif bx < reader.width * 0.15:
                            detection = None
                        # Exclude right side region
                        elif bx > reader.width * 0.85:
                            detection = None
                    
                    # Filter detections_list as well
                    if detections_list:
                        filtered_dets = []
                        for d in detections_list:
                            dx, dy = d[0], d[1]
                            # Keep only detections in valid playing field area
                            if (dy >= reader.height * 0.25 and 
                                dx >= reader.width * 0.15 and 
                                dx <= reader.width * 0.85):
                                filtered_dets.append(d)
                        detections_list = filtered_dets if filtered_dets else None
                
                t_tracking_start = time.time()
                track_result = self.tracker.update(detection, detections_list)
                t_tracking = (time.time() - t_tracking_start) * 1000
                self.performance_stats['tracking_times'].append(t_tracking)
                
                t_camera_start = time.time()
                ball_detection = detection
                if not camera_initialized and track_result:
                    x, y, is_tracking = track_result
                    self.virtual_camera.reset()
                    last_stable = (x, y)
                    anchor = (x, y)
                    crop_coords = self.virtual_camera.update(x, y, time.time(), velocity_hint=self.tracker.get_velocity())
                    camera_initialized = True
                elif track_result:
                    x, y, is_tracking = track_result
                    if not is_tracking:
                        roi_active = False
                        roi_fail_count = 0
                        roi_last_valid_pos = None
                    state = self.tracker.get_state()
                    vmag = state['velocity_magnitude'] if state else 0.0
                    kalman_ok = state['kalman_stable'] if state else True
                    if last_stable is None:
                        last_stable = (x, y)
                    if anchor is None:
                        anchor = (x, y)
                    use_x, use_y = x, y
                    det_ok = False
                    if ball_detection:
                        bx, by, bw, bh, bconf = ball_detection
                        step_ok = True
                        if last_det is not None:
                            dx_det = bx - last_det[0]
                            dy_det = by - last_det[1]
                            step_det = math.hypot(dx_det, dy_det)
                            step_ok = step_det <= close_thresh * 1.3
                        aspect = (bw / bh) if bh > 0 else 1.0
                        area = bw * bh
                        det_raw_ok = (bconf >= 0.24) and step_ok and (0.65 <= aspect <= 1.45) and (area >= area_min and area <= area_max)
                        if det_raw_ok:
                            det_consist += 1
                        else:
                            det_consist = max(det_consist - 1, 0)
                        far_thresh = max(int(diag * 0.22), 180)
                        d_far = math.hypot(bx - (anchor[0] if anchor else bx), by - (anchor[1] if anchor else by))
                        if d_far > far_thresh:
                            far_reacquire_count += 1
                        else:
                            far_reacquire_count = max(far_reacquire_count - 1, 0)
                        if det_raw_ok:
                            reacq_points.append((bx, by))
                            if len(reacq_points) > 10:
                                reacq_points.pop(0)
                        cluster_ok = True
                        if (not is_tracking) and (far_reacquire_count < far_reacquire_need) and len(reacq_points) >= 4:
                            mx = sum(p[0] for p in reacq_points) / len(reacq_points)
                            my = sum(p[1] for p in reacq_points) / len(reacq_points)
                            dists = [math.hypot(p[0]-mx, p[1]-my) for p in reacq_points]
                            count_in = sum(1 for d in dists if d <= close_thresh * 1.2)
                            cluster_ok = count_in >= 3
                        consist_need = det_consist_need + 1 if not is_tracking else det_consist_need
                        det_ok = det_raw_ok and det_consist >= consist_need and (d_far <= far_thresh or far_reacquire_count >= far_reacquire_need) and cluster_ok
                        if det_ok:
                            x, y = bx, by
                            is_tracking = True
                            reacq_points.clear()
                    if is_tracking and kalman_ok and cooldown == 0:
                        d = math.hypot(x - last_stable[0], y - last_stable[1])
                        if d > jump_reset_px:
                            cooldown = cooldown_max
                            frames_tracking = 0
                            stability_score = max(stability_score - 0.2, 0.0)
                            use_x, use_y = last_stable
                        else:
                            if d <= stable_step_px:
                                frames_tracking += 1
                                stability_score = min(stability_score + 0.12, 1.0)
                                a = 0.18
                                last_stable = (last_stable[0] * (1 - a) + x * a, last_stable[1] * (1 - a) + y * a)
                            else:
                                frames_tracking = max(frames_tracking - 1, 0)
                                stability_score = max(stability_score - 0.10, 0.0)
                    else:
                        if cooldown > 0:
                            cooldown -= 1
                            use_x, use_y = last_stable if last_stable else (x, y)
                        if not is_tracking:
                            frames_tracking = max(frames_tracking - 1, 0)
                            stability_score = max(stability_score - 0.05, 0.0)
                    if anchor is not None and is_tracking:
                        dx = use_x - anchor[0]
                        dy = use_y - anchor[1]
                        dist = math.hypot(dx, dy)
                        cur_step = int(max_pan_step * (1.0 + max(0.0, current_zoom_level - 1.0) * 1.5 + min(vmag / 450.0, 0.8)))
                        if dist > cur_step and dist > 1e-6:
                            r = cur_step / dist
                            anchor = (anchor[0] + dx * r, anchor[1] + dy * r)
                        else:
                            # Reduced alpha for smoother anchor movement
                            a_anch = 0.04 + 0.08 * max(0.0, current_zoom_level - 1.0)
                            if a_anch > 0.18:
                                a_anch = 0.18
                            anchor = (anchor[0] * (1.0 - a_anch) + use_x * a_anch, anchor[1] * (1.0 - a_anch) + use_y * a_anch)
                        use_x, use_y = anchor
                    if ball_detection:
                        last_det = (ball_detection[0], ball_detection[1])
                    if last_raw is not None:
                        dxn = x - last_raw[0]
                        dyn = y - last_raw[1]
                        step = math.hypot(dxn, dyn)
                        lvx, lvy = last_vec
                        lvnorm = math.hypot(lvx, lvy)
                        dnorm = math.hypot(dxn, dyn)
                        cosd = (lvx*dxn + lvy*dyn)/(lvnorm*dnorm) if (lvnorm>1e-6 and dnorm>1e-6) else 1.0
                        if step <= natural_step_px or cosd >= min_dir_cos:
                            natural_counter += 1
                        else:
                            natural_counter = max(natural_counter - 1, 0)
                        last_vec = (0.8*lvx + 0.2*dxn, 0.8*lvy + 0.2*dyn)
                        if cosd <= -0.4 and step > close_thresh*0.8:
                            bloom_counter = bloom_max
                    last_raw = (x, y)
                    vhx, vhy = self.tracker.get_velocity()
                    # CRITICAL: Only update camera when we have REAL detection, not predictions
                    # This prevents jitter from Kalman predictions
                    if is_tracking and ball_detection is not None:
                        crop_coords = self.virtual_camera.update(use_x, use_y, time.time(), velocity_hint=(vhx, vhy))
                    else:
                        # Freeze camera when predicting or lost - prevents jitter
                        crop_coords = self.virtual_camera.get_current_crop()
                    detection_count += 1
                    lost_count = 0
                    recent_positions.append((x, y))
                    if len(recent_positions) > 12:
                        recent_positions.pop(0)
                    if len(recent_positions) >= 5:
                        total = 0.0
                        cnt = 0
                        for i in range(len(recent_positions) - 1):
                            dxs = recent_positions[i + 1][0] - recent_positions[i][0]
                            dys = recent_positions[i + 1][1] - recent_positions[i][1]
                            total += math.hypot(dxs, dys)
                            cnt += 1
                        avg_step = total / cnt if cnt > 0 else 0.0
                        if avg_step <= close_thresh:
                            close_counter += 1
                        else:
                            close_counter = max(close_counter - 1, 0)
                    zoom_gate_ok = (det_ok) or (is_tracking and cooldown == 0 and (stability_score >= 0.40 or frames_tracking >= frames_required_for_zoom or natural_counter >= 2 or close_counter >= 2))
                    if zoom_gate_ok:
                        if vmag > 950:
                            target_zoom_level = 1.20
                        elif vmag > 650:
                            target_zoom_level = 1.35
                        elif vmag > 380:
                            target_zoom_level = 1.50
                        else:
                            target_zoom_level = 1.65
                        zoom_lock_count = zoom_lock_max
                        hold_zoom_level = target_zoom_level
                        if det_ok:
                            roi_stable_frames += 1
                        else:
                            roi_stable_frames = max(roi_stable_frames - 1, 0)
                        if (not roi_active) and roi_stable_frames >= roi_ready_frames:
                            roi_active = True
                            roi_fail_count = 0
                            roi_last_valid_pos = None  # Reset on ROI activation
                    else:
                        if not is_tracking:
                            zoom_lock_count = 0
                            hold_zoom_level = 1.0
                            target_zoom_level = 1.0
                        elif zoom_lock_count > 0:
                            zoom_lock_count -= 1
                            hold_zoom_level = max(1.2, hold_zoom_level * 0.98)
                            target_zoom_level = hold_zoom_level
                        else:
                            target_zoom_level = 1.0
                else:
                    crop_coords = self.virtual_camera.get_current_crop()
                    lost_count += 1
                    frames_tracking = 0
                    target_zoom_level = 1.0
                    zoom_lock_count = 0
                    hold_zoom_level = 1.0
                    roi_active = False
                zoom.set_target(target_zoom_level)
                current_zoom_level = zoom.update()
                x1, y1, x2, y2 = crop_coords
                if track_result:
                    if last_raw is not None and anchor is not None:
                        follow_cx, follow_cy = anchor[0], anchor[1]
                        wz = 0.70
                        zoom_cx = int(wz*last_raw[0] + (1.0-wz)*follow_cx)
                        zoom_cy = int(wz*last_raw[1] + (1.0-wz)*follow_cy)
                    else:
                        zoom_cx = int(0.8*x + 0.2*(anchor[0] if anchor else x))
                        zoom_cy = int(0.8*y + 0.2*(anchor[1] if anchor else y))
                else:
                    zoom_cx = (x1 + x2) // 2
                    zoom_cy = (y1 + y2) // 2
                safe_margin = max(8, int(diag * 0.02))
                if zoom_cx < safe_margin: zoom_cx = safe_margin
                if zoom_cx > reader.width - safe_margin: zoom_cx = reader.width - safe_margin
                if zoom_cy < safe_margin: zoom_cy = safe_margin
                if zoom_cy > reader.height - safe_margin: zoom_cy = reader.height - safe_margin
                if prev_zoom_center is not None:
                    zfac = max(0.0, min(1.0, current_zoom_level - 1.0))
                    az = 0.18 + 0.16 * zfac
                    zoom_cx = int(prev_zoom_center[0] * (1.0 - az) + zoom_cx * az)
                    zoom_cy = int(prev_zoom_center[1] * (1.0 - az) + zoom_cy * az)
                prev_zoom_center = (zoom_cx, zoom_cy)
                if current_zoom_level > 1.0:
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    zoomed_width = int(crop_width / current_zoom_level)
                    zoomed_height = int(crop_height / current_zoom_level)
                    x1 = max(0, zoom_cx - zoomed_width // 2)
                    y1 = max(0, zoom_cy - zoomed_height // 2)
                    x2 = min(reader.width, x1 + zoomed_width)
                    y2 = min(reader.height, y1 + zoomed_height)
                    if x2 - x1 < zoomed_width:
                        x1 = max(0, x2 - zoomed_width)
                    if y2 - y1 < zoomed_height:
                        y1 = max(0, y2 - zoomed_height)
                    if track_result:
                        rel_x = int(x - x1)
                        rel_y = int(y - y1)
                        mx = int(zoomed_width * 0.35)
                        my = int(zoomed_height * 0.35)
                        if rel_x < mx:
                            shift = mx - rel_x
                            s = max(1, int(shift * 0.45))
                            x1 = max(0, min(reader.width - zoomed_width, x1 - s))
                            x2 = x1 + zoomed_width
                        elif rel_x > zoomed_width - mx:
                            shift = rel_x - (zoomed_width - mx)
                            s = max(1, int(shift * 0.45))
                            x1 = max(0, min(reader.width - zoomed_width, x1 + s))
                            x2 = x1 + zoomed_width
                        if rel_y < my:
                            shift = my - rel_y
                            s = max(1, int(shift * 0.45))
                            y1 = max(0, min(reader.height - zoomed_height, y1 - s))
                            y2 = y1 + zoomed_height
                        elif rel_y > zoomed_height - my:
                            shift = rel_y - (zoomed_height - my)
                            s = max(1, int(shift * 0.45))
                            y1 = max(0, min(reader.height - zoomed_height, y1 + s))
                            y2 = y1 + zoomed_height
                x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
                
                # CRITICAL: Heavy smoothing on crop coordinates to eliminate jitter
                if prev_crop is not None:
                    pcx1, pcy1, pcx2, pcy2 = prev_crop
                    # Exponential smoothing with very low alpha (0.15 = keep 85% of previous crop)
                    alpha_crop = 0.15
                    x1 = int(pcx1 * (1.0 - alpha_crop) + x1 * alpha_crop)
                    y1 = int(pcy1 * (1.0 - alpha_crop) + y1 * alpha_crop)
                    x2 = int(pcx2 * (1.0 - alpha_crop) + x2 * alpha_crop)
                    y2 = int(pcy2 * (1.0 - alpha_crop) + y2 * alpha_crop)
                    
                    # Additional step limiter as backup
                    step_lim = int(max_crop_step_base * (1.0 + max(0.0, current_zoom_level - 1.0) * 0.8))
                    if abs(x1 - pcx1) > step_lim:
                        x1 = pcx1 + step_lim if x1 > pcx1 else pcx1 - step_lim
                        x2 = x1 + (pcx2 - pcx1)
                    if abs(y1 - pcy1) > step_lim:
                        y1 = pcy1 + step_lim if y1 > pcy1 else pcy1 - step_lim
                        y2 = y1 + (pcy2 - pcy1)
                if x1 < 0: x1 = 0
                if y1 < 0: y1 = 0
                if x2 > reader.width: x2 = reader.width
                if y2 > reader.height: y2 = reader.height
                if x2 <= x1: x2 = min(reader.width, x1 + 2)
                if y2 <= y1: y2 = min(reader.height, y1 + 2)
                prev_crop = (x1, y1, x2, y2)
                x1o, y1o, x2o, y2o = x1, y1, x2, y2
                t_camera = (time.time() - t_camera_start) * 1000
                self.performance_stats['camera_times'].append(t_camera)

                cropped = frame[y1o:y2o, x1o:x2o]
                
                if cropped.shape[:2] != (self.config['output']['height'], self.config['output']['width']):
                    cropped = cv2.resize(
                        cropped,
                        (self.config['output']['width'], self.config['output']['height'])
                    )
                
                if bloom_counter > 0:
                    intensity = bloom_counter / bloom_max
                    blurred = cv2.GaussianBlur(cropped, (0, 0), sigmaX=6, sigmaY=6)
                    cropped = cv2.addWeighted(cropped, 1.0, blurred, 0.35*intensity, 0)
                    bloom_counter -= 1
                
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
