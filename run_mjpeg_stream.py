import cv2
import numpy as np
import time
import torch
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import math
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    from app.Inference import BallDetector
except ModuleNotFoundError:
    from app.inference import BallDetector
from app.tracking import BallTracker
from app.utils import VideoReader, load_config, merge_configs, MJPEGServer
from app.utils.hls_streamer import HLSStreamer
from app.camera import VirtualCamera


def prepare_detection_frame(image: np.ndarray, max_width: int = None, max_height: int = None):
    """Resize frame for detection if it exceeds processing limits."""
    h, w = image.shape[:2]
    scale_factor = 1.0

    if max_width and max_width > 0 and w > max_width:
        scale_factor = max(scale_factor, w / max_width)
    if max_height and max_height > 0 and h > max_height:
        scale_factor = max(scale_factor, h / max_height)

    if scale_factor > 1.0:
        target_w = max(1, int(round(w / scale_factor)))
        target_h = max(1, int(round(h / scale_factor)))
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        scale_w = w / target_w
        scale_h = h / target_h
    else:
        resized = image
        scale_w = scale_h = 1.0

    return resized, scale_w, scale_h


def prepare_detection_tensor(image: np.ndarray, max_width: int = None, max_height: int = None, device: str = 'cuda'):
    """Resize frame for detection using GPU to avoid CPU bottleneck."""
    h, w = image.shape[:2]
    scale_factor = 1.0

    if max_width and max_width > 0 and w > max_width:
        scale_factor = max(scale_factor, w / max_width)
    if max_height and max_height > 0 and h > max_height:
        scale_factor = max(scale_factor, h / max_height)

    if scale_factor > 1.0:
        target_w = max(1, int(round(w / scale_factor)))
        target_h = max(1, int(round(h / scale_factor)))
        scale_w = w / target_w
        scale_h = h / target_h
        tensor = torch.from_numpy(image)
        # 2. Move to GPU
        tensor = tensor.to(device, non_blocking=True)
        # 3. Permute to (C, H, W) -> (3, H, W)
        tensor = tensor.permute(2, 0, 1)
        # 4. BGR to RGB (swap channels 0 and 2)
        tensor = tensor[[2, 1, 0], :, :]
        # 5. Normalize to 0-1 float
        tensor = tensor.float().div_(255.0)
        # 6. Add batch dim for interpolate: (1, 3, H, W)
        tensor = tensor.unsqueeze(0)
        
        # 7. Resize on GPU
        tensor = torch.nn.functional.interpolate(
            tensor, 
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 8. Remove batch dim: (3, H, W)
        tensor = tensor.squeeze(0)
        
    else:
        # No resize needed, but still convert to tensor for GPU pipeline
        scale_w = scale_h = 1.0
        tensor = torch.from_numpy(image).to(device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1) # (C, H, W)
        tensor = tensor[[2, 1, 0], :, :] # BGR -> RGB
        tensor = tensor.float().div_(255.0)

    return tensor, scale_w, scale_h


class SmoothZoom:
    def __init__(self, min_zoom: float = 1.0, max_zoom: float = 2.5, stiffness: float = 0.08, damping: float = 0.35, max_rate: float = 0.25, max_rate_in: float = None, max_rate_out: float = None, accel_limit: float = None):
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

def main():
    logger.info("="*60)
    logger.info("🎥 MJPEG Stream Server - Football Detection")
    logger.info("="*60)
    
    # Load configs
    logger.info("\n[1/5] Loading configurations...")
    model_config = load_config('configs/model_config.yml')
    stream_config = load_config('configs/stream_config.yml')
    config = merge_configs(model_config, stream_config)
    
    # Initialize detector (auto-detect model type)
    logger.info("[2/5] Loading RF-DETR model...")
    model_path = config['model']['path']
    
    if model_path and model_path.endswith('.onnx'):
        from app.inference.detector_onnx import BallDetectorONNX
        logger.info("    Using ONNX Runtime backend")
        detector = BallDetectorONNX(
            onnx_path=model_path,
            confidence_threshold=config['model']['confidence'],
            imgsz=config['model'].get('imgsz', 640)
        )
    elif model_path and model_path.endswith('.engine'):
        from app.inference.detector_tensorrt import BallDetectorTensorRT
        logger.info("    Using TensorRT backend")
        detector = BallDetectorTensorRT(
            engine_path=model_path,
            confidence_threshold=config['model']['confidence'],
            imgsz=config['model'].get('imgsz', 640)
        )
    else:
        logger.info("    Using PyTorch backend")
        detector = BallDetector(
            model_path=model_path,
            confidence_threshold=config['model']['confidence'],
            iou_threshold=config['model'].get('iou_threshold', 0.45),
            device=config['model'].get('device', 'cuda'),
            half_precision=config['model'].get('half_precision', True),
            imgsz=config['model'].get('imgsz', 640),
            multi_scale=config['model'].get('multi_scale', False),
            warmup_iterations=config['model'].get('warmup_iterations', 3)
        )
    
    # Initialize tracker with optimized parameters for ball tracking
    logger.info("[3/5] Initializing ball tracker...")
    tracking_config = config.get('tracking', {})
    tracker = BallTracker(
        max_lost_frames=tracking_config.get('max_lost_frames', 10),
        min_confidence=tracking_config.get('min_confidence', 0.45),
        iou_threshold=tracking_config.get('iou_threshold', 0.10),
        adaptive_noise=True,
        allow_chaos_mode=tracking_config.get('allow_chaos_mode', False),
        allow_jitter_mode=tracking_config.get('allow_jitter_mode', False),
        detection_smoothing=tracking_config.get('detection_smoothing', 0.25)
    )
    print(f"   → Tracker config: max_lost={tracker.max_lost_frames}, min_conf={tracker.min_confidence:.2f}, iou={tracker.iou_threshold:.2f}")
    
    # Start MJPEG server
    mjpeg_port = config.get('stream', {}).get('mjpeg_port', 8554)
    hls_output_dir = os.path.join(os.getcwd(), 'hls_output')
    
    logger.info(f"[4/5] Starting Stream Server on port {mjpeg_port}...")
    mjpeg_server = MJPEGServer(port=mjpeg_port, hls_dir=hls_output_dir)
    mjpeg_server.start()
    

    
    logger.info("✅ Stream Server started!")
    logger.info(f"📺 MJPEG Stream: http://localhost:{mjpeg_port}/stream.mjpg")
    logger.info(f"📺 HLS Stream:   http://localhost:{mjpeg_port}/hls/stream.m3u8")
    logger.info("💡 En Colab, usa ngrok para exponer el puerto 8554")
    
    # Open video
    output_config = config.get('output', {})
    target_output_width = int(output_config.get('width', 1920))
    target_output_height = int(output_config.get('height', 1080))

    processing_config = config.get('processing', {})
    processing_width = int(processing_config.get('detection_width', 1920))
    processing_height = int(processing_config.get('detection_height', 1080))
    logger.info(f"   → Detection resolution cap: {processing_width}x{processing_height}")

    video_path = config.get('stream', {}).get('input_url', '/content/football.mp4')
    logger.info(f"\n[5/5] Opening video: {video_path}")
    
    use_ffmpeg = True
    try:
        reader = FFmpegVideoLoader(video_path)
        logger.info(f"✅ Video opened with FFmpeg Pipe (CUDA): {reader.width}x{reader.height} @ {reader.fps:.1f}fps")
    except Exception as e:
        logger.warning(f"⚠ FFmpeg Loader failed, falling back to OpenCV...")
        # Enable auto-reconnect for streams (assumed if http/rtsp/rtmp)
        is_stream = video_path.startswith(('http', 'rtsp', 'rtmp'))
        reader = VideoReader(video_path, reconnect=is_stream)
        use_ffmpeg = False
        logger.info(f"✅ Video opened with OpenCV: {reader.width}x{reader.height} @ {reader.fps:.1f}fps")

    # Initialize HLS Streamer
    # Cap HLS resolution to 1920 width to ensure compatibility and performance
    # The source video is 4608x1728 which is too large for standard H.264 levels and NVENC
    # hls_width = int(config.get('output', {}).get('width', 1920))
    # hls_height = int(config.get('output', {}).get('height', 1080))
    
    # if hls_width > 1920:
    #     aspect_ratio = hls_height / hls_width
    #     hls_width = 1920
    #     hls_height = int(hls_width * aspect_ratio)
    #     # Ensure even dimensions for video encoding
    #     hls_height = hls_height - (hls_height % 2)
        
    # # Use source FPS if available, otherwise default to 30
    # source_fps = getattr(reader, 'fps', 30.0)
    # logger.info(f"Source FPS: {source_fps:.2f}")
    # logger.info(f"HLS Resolution: {hls_width}x{hls_height}")
    
    # hls_streamer = HLSStreamer(output_dir=hls_output_dir, width=hls_width, height=hls_height, fps=int(source_fps))
    # hls_streamer.start()
    
    # Re-initialize tracker with correct frame dimensions
    tracker = BallTracker(
        frame_width=reader.width,
        frame_height=reader.height,
        max_lost_frames=tracking_config.get('max_lost_frames', 10),
        min_confidence=tracking_config.get('min_confidence', 0.45),
        iou_threshold=tracking_config.get('iou_threshold', 0.10),
        adaptive_noise=True,
        allow_chaos_mode=tracking_config.get('allow_chaos_mode', False),
        allow_jitter_mode=tracking_config.get('allow_jitter_mode', False),
        detection_smoothing=tracking_config.get('detection_smoothing', 0.25)
    )
    logger.info(f"   → Tracker config: max_lost={tracker.max_lost_frames}, min_conf={tracker.min_confidence:.2f}, iou={tracker.iou_threshold:.2f}")

    # Initialize virtual camera
    # Base crop size for normal view - will zoom progressively when tracking
    camera_config = config.get('camera', {})
    base_output_width = int(camera_config.get('base_output_width', target_output_width))
    base_output_height = int(camera_config.get('base_output_height', target_output_height))
    base_output_width = min(base_output_width, reader.width)
    base_output_height = min(base_output_height, reader.height)

    virtual_camera = VirtualCamera(
        frame_width=reader.width,
        frame_height=reader.height,
        output_width=base_output_width,
        output_height=base_output_height,
        dead_zone_percent=0.10,
        anticipation_factor=0.35,
        zoom_padding=camera_config.get('zoom_padding', 1.2),
        smoothing_freq=camera_config.get('smoothing_freq', 30.0),
        smoothing_min_cutoff=camera_config.get('smoothing_min_cutoff', 0.5),
        smoothing_beta=camera_config.get('smoothing_beta', 0.05),
        use_pid=True,
        prediction_steps=5
    )
    print(f"   → Camera base crop: {base_output_width}x{base_output_height}")
    print(f"   → Final stream output: {target_output_width}x{target_output_height}")
    print(f"   → Professional cameraman mode: Smooth zoom when tracking ball")

    effects_config = config.get('effects', {})
    bloom_enabled = bool(effects_config.get('enable_bloom', False))
    bloom_strength = float(effects_config.get('bloom_strength', 0.35))
    bloom_sigma = float(effects_config.get('bloom_sigma', 6.0))
    bloom_duration_frames = int(effects_config.get('bloom_duration_frames', 12))
    
    ball_class_id = config['model'].get('ball_class_id', 0)
    
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting processing loop...")
    logger.info("="*60)
    logger.info("Press Ctrl+C to stop\n")
    
    frame_count = 0
    last_log_time = time.time()
    fps_history = []
    camera_initialized = False
    detection_count = 0
    lost_count = 0
    crop_coords = (0, 0, reader.width, reader.height)
    
    # Fast zoom system
    current_zoom_level = 1.0
    target_zoom_level = 1.0
    max_zoom_level = 1.8
    frames_tracking = 0
    frames_required_for_zoom = 4
    lost_search_center = None  # Gradual expansion center when lost
    
    # Loop detection and recovery
    last_tracking_state = False
    tracking_state_changes = []
    loop_detection_window = 120  # frames (increased to ~4 seconds)
    loop_threshold = 20  # changes in window (increased to allow more fluctuation)
    consecutive_det_frames = 0  # consecutive frames with detection
    target_zoom_before_loss = 1.0

    zoom = SmoothZoom(min_zoom=1.0, max_zoom=max_zoom_level, stiffness=0.060, damping=0.60, max_rate=0.11, max_rate_in=0.14, max_rate_out=0.10, accel_limit=0.05)
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
    recent_dets = []
    reacq_points = []
    area_min = max(64, int(0.00002 * reader.width * reader.height))
    area_max = int(0.030 * reader.width * reader.height)
    center_lp = None
    zoom_target_lp = 1.0
    roi_active = False
    roi_stable_frames = 0
    roi_ready_frames = 40 
    roi_fail_count = 0
    roi_fail_max = 150  
    roi_last_valid_pos = None
    bloom_counter = 0
    bloom_max = bloom_duration_frames if bloom_enabled else 0
    far_reacquire_count = 0
    far_reacquire_need = 6
    last_reliable_position = None
    lost_pan_limit = max(18, int(diag * 0.012))
    det_skip = 0
    
    try:
        while True:
            loop_start = time.time()
            det_ok = False
            
            ret, frame = reader.read()
            if not ret or frame is None:
                logger.warning("📹 Video ended, restarting...")
                reader.release()
                if use_ffmpeg:
                    try:
                        reader = FFmpegVideoLoader(video_path)
                    except:
                        reader = VideoReader(video_path)
                        use_ffmpeg = False
                else:
                    reader = VideoReader(video_path)
                continue
            follow_cx, follow_cy = None, None
            
            start_inf = time.time()
            use_roi = False
            offx, offy = 0, 0
            do_inference = True
            if frames_tracking >= 8 and stability_score >= 0.60 and cooldown == 0:
                if det_skip > 0:
                    det_skip -= 1
                    do_inference = False
                else:
                    det_skip = 1
            if do_inference:
                if prev_crop is not None and roi_active:
                    rx1, ry1, rx2, ry2 = prev_crop
                    rx1 = max(0, min(reader.width-2, int(rx1)))
                    ry1 = max(0, min(reader.height-2, int(ry1)))
                    rx2 = max(rx1+2, min(reader.width, int(rx2)))
                    ry2 = max(ry1+2, min(reader.height, int(ry2)))
                    frame_in = frame[ry1:ry2, rx1:rx2]
                    use_roi = True
                    offx, offy = rx1, ry1
                    # GPU-accelerated resizing
                    model_input, scale_w, scale_h = prepare_detection_tensor(
                        frame_in,
                        processing_width,
                        processing_height
                    )
                    det_result = detector.predict_ball_only(
                        model_input,
                        ball_class_id,
                        use_temporal_filtering=False,
                        return_candidates=True
                    )
                else:
                    # GPU-accelerated resizing
                    model_input, scale_w, scale_h = prepare_detection_tensor(
                        frame,
                        processing_width,
                        processing_height
                    )
                    det_result = detector.predict_ball_only(
                        model_input, 
                        ball_class_id,
                        use_temporal_filtering=True,
                        return_candidates=True
                    )
                if det_result[0] is not None:
                    x, y, w_box, h_box, conf = det_result[0]
                    det_result = (
                        (x * scale_w, y * scale_h, w_box * scale_w, h_box * scale_h, conf),
                        det_result[1]
                    )
                if det_result[1]:
                    det_result = (
                        det_result[0],
                        [
                            (d[0] * scale_w, d[1] * scale_h, d[2] * scale_w, d[3] * scale_h, d[4], d[5])
                            for d in det_result[1]
                        ]
                    )
                inf_time = (time.time() - start_inf) * 1000
            else:
                det_result = (None, None)
                inf_time = 0.0
            
            ball_detection, all_detections = det_result
            if use_roi:
                if ball_detection is not None:
                    bx, by, bw, bh, bc = ball_detection
                    ball_detection = (bx + offx, by + offy, bw, bh, bc)
                if all_detections:
                    mapped = []
                    for d in all_detections:
                        mapped.append((d[0] + offx, d[1] + offy, d[2], d[3], d[4], d[5]))
                    all_detections = mapped
                if roi_active:
                    # Check if detection is viable
                    detection_viable = False
                    if ball_detection is not None:
                        bx, by, bw, bh, bconf = ball_detection
                        # Check for erratic movement (Teleportation Guard)
                        if roi_last_valid_pos is not None:
                            dx = bx - roi_last_valid_pos[0]
                            dy = by - roi_last_valid_pos[1]
                            jump_dist = math.hypot(dx, dy)
                            
                            # Teleportation Guard: Reject large jumps unless confidence is super high
                            # 300px is about 15% of width
                            jump_limit = reader.width * 0.15
                            if jump_dist > jump_limit and bconf < 0.80:
                                logger.warning(f"🛡 Teleportation Guard: Rejected jump of {jump_dist:.1f}px with conf {bconf:.2f}")
                                detection_viable = False
                            elif jump_dist < diag * 0.3:
                                detection_viable = True
                            else:
                                detection_viable = False
                        else:
                            detection_viable = True
                        
                        # Check for multiple conflicting detections
                        if detection_viable and all_detections and len(all_detections) > 3:
                            # Too many detections = confusion
                            detection_viable = False
                        
                        if detection_viable:
                            roi_last_valid_pos = (bx, by)
                    
                    # Only increment fail count if no viable detection
                    if not detection_viable:
                        roi_fail_count += 1
                        ball_detection = None # Ensure we don't pass bad detection to tracker
                    else:
                        roi_fail_count = 0
                    
                    if roi_fail_count >= roi_fail_max:
                        logger.warning(f"⚠ ROI lost ball for {roi_fail_max} frames - switching to full-frame detection")
                        roi_active = False
                        roi_fail_count = 0
                        roi_stable_frames = 0
                        roi_last_valid_pos = None
            apply_spatial_filter = (not roi_active) and (current_zoom_level < 1.4)
            
            if apply_spatial_filter:
                if ball_detection is not None:
                    bx, by, bw, bh, bconf = ball_detection
                    
                    if by < reader.height * 0.25:
                        ball_detection = None
                    
                    elif bx < reader.width * 0.15:
                        ball_detection = None
                    
                    elif bx > reader.width * 0.85:
                        ball_detection = None
                
                if all_detections:
                    filtered_dets = []
                    for d in all_detections:
                        dx, dy = d[0], d[1]
                        if (dy >= reader.height * 0.25 and 
                            dx >= reader.width * 0.15 and 
                            dx <= reader.width * 0.85):
                            filtered_dets.append(d)
                    all_detections = filtered_dets if filtered_dets else None
            
            track_result = tracker.update(ball_detection, all_detections)
            
            if not camera_initialized and track_result:
                x, y, is_tracking = track_result
                virtual_camera.reset()
                last_stable = (x, y)
                anchor = (x, y)
                crop_coords = virtual_camera.update(x, y, time.time(), velocity_hint=tracker.get_velocity())
                camera_initialized = True
                logger.info(f"[CAMERA] Initialized at ball position: ({x:.1f}, {y:.1f})")
            elif track_result:
                x, y, is_tracking = track_result
                
                det_ok = False
                if ball_detection:
                    det_ok = True
                
                if is_tracking != last_tracking_state:
                    tracking_state_changes.append(frame_count)
                    tracking_state_changes = [f for f in tracking_state_changes if frame_count - f < loop_detection_window]
                    
                    if len(tracking_state_changes) >= loop_threshold:
                        logger.warning(f"⚠ LOOP DETECTED - Resetting to center")
                        center_x = reader.width // 2
                        center_y = reader.height // 2
                        virtual_camera.reset()
                        crop_coords = virtual_camera.update(center_x, center_y, time.time(), velocity_hint=(0, 0))
                        roi_active = False
                        roi_stable_frames = 0
                        roi_fail_count = 0
                        lost_search_center = None
                        target_zoom_level = 1.0
                        tracking_state_changes.clear()
                        consecutive_det_frames = 0
                last_tracking_state = is_tracking
                
                if det_ok and ball_detection:
                    consecutive_det_frames += 1
                else:
                    consecutive_det_frames = 0
                
                if not is_tracking:
                    if roi_active:
                        logger.info(f"⚠ ROI deactivated - back to full-frame detection")
                    roi_active = False
                    roi_fail_count = 0
                    roi_stable_frames = 0
                    roi_last_valid_pos = None
                else:
                    roi_stable_frames += 1
                    if (not roi_active) and roi_stable_frames >= roi_ready_frames:
                        roi_active = True
                        roi_fail_count = 0
                        roi_last_valid_pos = None
                        logger.info(f"✓ ROI activated - now detecting only in zoom region for performance")
                    if is_tracking:
                        last_reliable_position = (x, y)
                    else:
                        if last_reliable_position is not None:
                            blend = 0.25
                            x = last_reliable_position[0] * (1.0 - blend) + x * blend
                            y = last_reliable_position[1] * (1.0 - blend) + y * blend
                        current_cx = virtual_camera.current_center_x
                        current_cy = virtual_camera.current_center_y
                        dx = x - current_cx
                        dy = y - current_cy
                        if abs(dx) > lost_pan_limit:
                            x = current_cx + math.copysign(lost_pan_limit, dx)
                        if abs(dy) > lost_pan_limit:
                            y = current_cy + math.copysign(lost_pan_limit, dy)
                
                state = tracker.get_state()
                vmag = state['velocity_magnitude'] if state else 0.0
                kalman_ok = state['kalman_stable'] if state else True
                if last_stable is None:
                    last_stable = (x, y)
                if anchor is None:
                    anchor = (x, y)
                use_x, use_y = x, y
                if ball_detection:
                    last_det = (ball_detection[0], ball_detection[1])
                    recent_dets.append((ball_detection[0], ball_detection[1]))
                    if len(recent_dets) > 10:
                        recent_dets.pop(0)
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

                dist_anchor_ball = math.hypot((anchor[0] if anchor else x) - x, (anchor[1] if anchor else y) - y)
                zoom_gate_ok = (det_ok) or (is_tracking and cooldown == 0 and (stability_score >= 0.40 or frames_tracking >= frames_required_for_zoom or natural_counter >= 2 or close_counter >= 2))
                if zoom_gate_ok:
                    det_zoom = None
                    if det_ok and len(recent_dets) >= 3:
                        ttot = 0.0
                        tcnt = 0
                        for i in range(len(recent_dets) - 1):
                            ttot += math.hypot(recent_dets[i+1][0]-recent_dets[i][0], recent_dets[i+1][1]-recent_dets[i][1])
                            tcnt += 1
                        avg_det_step = ttot / tcnt if tcnt > 0 else 0.0
                        if avg_det_step <= close_thresh * 0.6:
                            det_zoom = 1.70
                        elif avg_det_step <= close_thresh * 1.1:
                            det_zoom = 1.55
                        else:
                            det_zoom = 1.35
                    if roi_active and consecutive_det_frames < 3:
                        target_zoom_before_loss = max(1.0, target_zoom_before_loss * 0.97)
                        target_zoom_level = target_zoom_before_loss
                        zoom_lock_count = max(zoom_lock_count - 2, 0)  # Faster unlock
                    else:
                        if det_zoom is not None:
                            target_zoom_level = min(max_zoom_level, det_zoom)
                        else:
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
                        target_zoom_before_loss = target_zoom_level
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
            
            dz = target_zoom_level - zoom_target_lp
            az_t = 0.28 if abs(dz) > 0.25 else 0.18
            zoom_target_lp = zoom_target_lp * (1.0 - az_t) + target_zoom_level * az_t
            zoom.set_target(zoom_target_lp)
            current_zoom_level = zoom.update()
            
            x1, y1, x2, y2 = crop_coords
            if track_result:
                if follow_cx is not None and follow_cy is not None:
                    wz = 0.70
                    ax = (anchor[0] if anchor else follow_cx)
                    ay = (anchor[1] if anchor else follow_cy)
                    zoom_cx = int(wz*follow_cx + (1.0-wz)*ax)
                    zoom_cy = int(wz*follow_cy + (1.0-wz)*ay)
                else:
                    zoom_cx = int(0.8*x + 0.2*(anchor[0] if anchor else x))
                    zoom_cy = int(0.8*y + 0.2*(anchor[1] if anchor else y))
            else:
                zoom_cx = (x1 + x2) // 2
                zoom_cy = (y1 + y2) // 2
            safe_margin = max(8, int(diag * 0.02))
            if zoom_cx < safe_margin:
                zoom_cx = safe_margin
            if zoom_cx > reader.width - safe_margin:
                zoom_cx = reader.width - safe_margin
            if zoom_cy < safe_margin:
                zoom_cy = safe_margin
            if zoom_cy > reader.height - safe_margin:
                zoom_cy = reader.height - safe_margin

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
                    rel_x = x - x1
                    rel_y = y - y1
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
            
            if prev_crop is not None:
                pcx1, pcy1, pcx2, pcy2 = prev_crop
                
                if det_ok:
                    alpha_x = 0.12
                    alpha_y = 0.10
                else:
                    # Heavy smoothing when predicting to prevent shake
                    alpha_x = 0.02
                    alpha_y = 0.02
                
                x1 = int(pcx1 * (1.0 - alpha_x) + x1 * alpha_x)
                y1 = int(pcy1 * (1.0 - alpha_y) + y1 * alpha_y)
                x2 = int(pcx2 * (1.0 - alpha_x) + x2 * alpha_x)
                y2 = int(pcy2 * (1.0 - alpha_y) + y2 * alpha_y)
                
                step_lim_x = int(max_crop_step_base * (1.0 + max(0.0, current_zoom_level - 1.0) * 0.8))
                step_lim_y = int(step_lim_x * 0.5) if not det_ok else step_lim_x
                
                if abs(x1 - pcx1) > step_lim_x:
                    x1 = pcx1 + step_lim_x if x1 > pcx1 else pcx1 - step_lim_x
                    x2 = x1 + (pcx2 - pcx1)
                if abs(y1 - pcy1) > step_lim_y:
                    y1 = pcy1 + step_lim_y if y1 > pcy1 else pcy1 - step_lim_y
                    y2 = y1 + (pcy2 - pcy1)
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > reader.width: x2 = reader.width
            if y2 > reader.height: y2 = reader.height
            if x2 <= x1: x2 = min(reader.width, x1 + 2)
            if y2 <= y1: y2 = min(reader.height, y1 + 2)
            prev_crop = (x1, y1, x2, y2)
            cropped = frame[y1:y2, x1:x2].copy()
            
            
            if track_result:
                x, y, is_tracking = track_result
                rel_x = int(x - x1)
                rel_y = int(y - y1)
                
                if is_tracking:
                    cv2.circle(cropped, (rel_x, rel_y), 15, (0, 255, 0), 3)
                    cv2.circle(cropped, (rel_x, rel_y), 5, (0, 255, 0), -1)
                    cv2.line(cropped, (rel_x - 20, rel_y), (rel_x + 20, rel_y), (0, 255, 0), 2)
                    cv2.line(cropped, (rel_x, rel_y - 20), (rel_x, rel_y + 20), (0, 255, 0), 2)
                else:
                    cv2.circle(cropped, (rel_x, rel_y), 15, (0, 255, 255), 3)
                    cv2.circle(cropped, (rel_x, rel_y), 5, (0, 255, 255), -1)
                
                if ball_detection:
                    bx, by, bw, bh, conf = ball_detection
                    bbox_x = int(bx - x1)
                    bbox_y = int(by - y1)
                    bbox_w = int(bw)
                    bbox_h = int(bh)
                    
                    cv2.rectangle(cropped,
                                (bbox_x - bbox_w//2, bbox_y - bbox_h//2),
                                (bbox_x + bbox_w//2, bbox_y + bbox_h//2),
                                (0, 255, 0), 2)
                    
                    cv2.putText(cropped, f"Ball: {conf:.2f}",
                               (bbox_x - bbox_w//2, bbox_y - bbox_h//2 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if (not track_result) and ball_detection:
                bx, by, bw, bh, conf = ball_detection
                bbox_x = int(bx - x1)
                bbox_y = int(by - y1)
                bbox_w = int(bw)
                bbox_h = int(bh)
                cv2.rectangle(cropped,
                              (bbox_x - bbox_w//2, bbox_y - bbox_h//2),
                              (bbox_x + bbox_w//2, bbox_y + bbox_h//2),
                              (0, 255, 255), 2)
                cv2.putText(cropped, f"Ball: {conf:.2f}",
                           (bbox_x - bbox_w//2, bbox_y - bbox_h//2 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if bloom_enabled and bloom_counter > 0:
                intensity = bloom_counter / bloom_max
                blurred = cv2.GaussianBlur(cropped, (0, 0), sigmaX=bloom_sigma, sigmaY=bloom_sigma)
                cropped = cv2.addWeighted(cropped, 1.0, blurred, bloom_strength * intensity, 0)
                bloom_counter -= 1

            loop_time = (time.time() - loop_start) * 1000
            fps = 1000 / loop_time if loop_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            
            avg_fps = np.mean(fps_history)
            
            stats_bg_height = 120
            overlay = cropped.copy()
            cv2.rectangle(overlay, (5, 5), (350, stats_bg_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, cropped, 0.5, 0, cropped)
            
            cv2.putText(cropped, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cropped, f"Inference: {inf_time:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if track_result:
                x, y, is_tracking = track_result
                status_text = "TRACKING" if is_tracking else "PREDICTING"
                status_color = (0, 255, 0) if is_tracking else (0, 255, 255)
                cv2.putText(cropped, f"Status: {status_text}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            else:
                cv2.putText(cropped, "Status: LOST", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(cropped, f"Detections: {detection_count}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show zoom indicator
            if current_zoom_level > 1.05:
                zoom_text = f"ZOOM: {current_zoom_level:.1f}x"
                zoom_color = (0, 255, 255) if current_zoom_level < max_zoom_level else (0, 255, 0)
            else:
                zoom_text = "ZOOM: OFF"
                zoom_color = (128, 128, 128)
            
            cv2.putText(cropped, zoom_text, (cropped.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, zoom_color, 2)
            
            if track_result:
                x, y, is_tracking = track_result
                if is_tracking and frames_tracking > 0:
                    track_color = (0, 255, 0) if frames_tracking >= frames_required_for_zoom else (255, 255, 0)
                    track_text = f"Lock: {min(frames_tracking, frames_required_for_zoom)}/{frames_required_for_zoom}"
                    cv2.putText(cropped, track_text, (cropped.shape[1] - 150, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)
                elif not is_tracking:
                    cv2.putText(cropped, "PRED (0/15)", (cropped.shape[1] - 150, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            resize_interp = cv2.INTER_CUBIC if (target_output_width > cropped.shape[1]) else cv2.INTER_LINEAR
            target_frame_time = 1.0 / source_fps
            if process_time < target_frame_time:
                time.sleep(target_frame_time - process_time)
            
            if frame_count % 30 == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                actual_fps = 30 / elapsed if elapsed > 0 else 0
                
                tracking_status = "ACTIVE" if track_result and track_result[2] else "LOST" if not track_result else "PRED"
                if track_result:
                    x, y, _ = track_result
                    print(f"[STREAM] Frame {frame_count:4d} | FPS: {actual_fps:5.1f} | Inf: {inf_time:5.1f}ms | Track: {tracking_status} | Zoom: {current_zoom_level:.2f}x | Lock: {frames_tracking}")
                else:
                    print(f"[STREAM] Frame {frame_count:4d} | FPS: {actual_fps:5.1f} | Inf: {inf_time:5.1f}ms | Track: {tracking_status} | Lost: {lost_count}")
                
                last_log_time = current_time
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping stream...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'reader' in locals() and reader:
            reader.release()
        if 'mjpeg_server' in locals() and mjpeg_server:
            mjpeg_server.stop()
        if 'hls_streamer' in locals() and hls_streamer:
            hls_streamer.stop()
            pass
        print("✅ Resources released.")

if __name__ == "__main__":
    main()
