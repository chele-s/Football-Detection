#!/usr/bin/env python3
import cv2
import numpy as np
import time
import torch
from pathlib import Path
import math

from app.inference import BallDetector
from app.tracking import BallTracker
from app.utils import VideoReader, load_config, merge_configs, MJPEGServer
from app.camera import VirtualCamera


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
    print("="*60)
    print("🎥 MJPEG Stream Server - Football Detection")
    print("="*60)
    
    # Load configs
    print("\n[1/5] Loading configurations...")
    model_config = load_config('configs/model_config.yml')
    stream_config = load_config('configs/stream_config.yml')
    config = merge_configs(model_config, stream_config)
    
    # Initialize detector
    print("[2/5] Loading RF-DETR model...")
    detector = BallDetector(
        model_path=config['model']['path'],
        confidence_threshold=config['model']['confidence'],
        iou_threshold=config['model'].get('iou_threshold', 0.45),
        device=config['model'].get('device', 'cuda'),
        half_precision=config['model'].get('half_precision', True),
        imgsz=config['model'].get('imgsz', 640),
        multi_scale=config['model'].get('multi_scale', False),
        warmup_iterations=config['model'].get('warmup_iterations', 3)
    )
    
    # Initialize tracker with optimized parameters for ball tracking
    print("[3/5] Initializing ball tracker...")
    tracker = BallTracker(
        max_lost_frames=config['tracking'].get('max_lost_frames', 10),
        min_confidence=config['tracking'].get('min_confidence', 0.10),
        iou_threshold=config['tracking'].get('iou_threshold', 0.20),
        adaptive_noise=True
    )
    print(f"   → Tracker config: max_lost={tracker.max_lost_frames}, min_conf={tracker.min_confidence:.2f}, iou={tracker.iou_threshold:.2f}")
    
    # Start MJPEG server
    mjpeg_port = config.get('stream', {}).get('mjpeg_port', 8554)
    print(f"[4/5] Starting MJPEG server on port {mjpeg_port}...")
    mjpeg_server = MJPEGServer(port=mjpeg_port)
    mjpeg_server.start()
    print("✅ MJPEG server started!")
    print("📺 Stream URL: http://localhost:8554/stream.mjpg")
    print("💡 En Colab, usa ngrok para exponer el puerto 8554")
    
    # Open video
    video_path = config.get('stream', {}).get('input_url', '/content/football.mp4')
    print(f"\n[5/5] Opening video: {video_path}")
    reader = VideoReader(video_path)
    print(f"✅ Video opened: {reader.width}x{reader.height} @ {reader.fps:.1f}fps")
    
    # Initialize virtual camera
    # Base crop size for normal view - will zoom progressively when tracking
    base_output_width = 960
    base_output_height = 540
    camera_config = config.get('camera', {})
    
    virtual_camera = VirtualCamera(
        frame_width=reader.width,
        frame_height=reader.height,
        output_width=base_output_width,
        output_height=base_output_height,
        dead_zone_percent=0.10,
        anticipation_factor=0.35,
        zoom_padding=camera_config.get('zoom_padding', 1.2),
        smoothing_freq=camera_config.get('smoothing_freq', 30.0),
        smoothing_min_cutoff=camera_config.get('smoothing_min_cutoff', 1.5),
        smoothing_beta=camera_config.get('smoothing_beta', 0.002),
        use_pid=True,
        prediction_steps=5
    )
    print(f"   → Camera config: {base_output_width}x{base_output_height} base crop")
    print(f"   → Professional cameraman mode: Smooth zoom when tracking ball")
    
    ball_class_id = config['model'].get('ball_class_id', 0)
    
    print("\n" + "="*60)
    print("🚀 Starting processing loop...")
    print("="*60)
    print("Press Ctrl+C to stop\n")
    
    frame_count = 0
    last_log_time = time.time()
    fps_history = []
    camera_initialized = False
    detection_count = 0
    lost_count = 0
    
    # Fast zoom system
    current_zoom_level = 1.0
    target_zoom_level = 1.0
    max_zoom_level = 1.8
    frames_tracking = 0
    frames_required_for_zoom = 4

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
    max_crop_step_base = max(8, int(diag * 0.018))
    recent_dets = []
    reacq_points = []
    area_min = max(64, int(0.00002 * reader.width * reader.height))
    area_max = int(0.030 * reader.width * reader.height)
    center_lp = None
    zoom_target_lp = 1.0
    roi_active = False
    roi_stable_frames = 0
    roi_ready_frames = 35
    roi_fail_count = 0
    roi_fail_max = 6
    bloom_counter = 0
    bloom_max = 12
    far_reacquire_count = 0
    far_reacquire_need = 6
    det_skip = 0
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = reader.read()
            if not ret or frame is None:
                print("📹 Video ended, restarting...")
                reader.release()
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
                    det_result = detector.predict_ball_only(
                        frame_in,
                        ball_class_id,
                        use_temporal_filtering=False,
                        return_candidates=True
                    )
                else:
                    det_result = detector.predict_ball_only(
                        frame, 
                        ball_class_id,
                        use_temporal_filtering=True,
                        return_candidates=True
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
                    if ball_detection is None:
                        roi_fail_count += 1
                    else:
                        roi_fail_count = 0
                    if roi_fail_count >= roi_fail_max:
                        roi_active = False
                        roi_fail_count = 0
            
            # Spatial filter: exclude upper region (stands/lights)
            # Only reject detections in the TOP 25% of frame where stands/lights are
            if ball_detection is not None:
                bx, by, bw, bh, bconf = ball_detection
                if by < reader.height * 0.25:
                    ball_detection = None
            
            # Filter all_detections as well
            if all_detections:
                filtered_dets = []
                for d in all_detections:
                    dx, dy = d[0], d[1]
                    if dy >= reader.height * 0.25:
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
                print(f"[CAMERA] Initialized at ball position: ({x:.1f}, {y:.1f})")
            elif track_result:
                x, y, is_tracking = track_result
                if not is_tracking:
                    # Freeze camera during search to avoid jitter from predictions
                    roi_active = False
                    roi_fail_count = 0
                else:
                    # Tracker is confident - activate ROI after a few frames
                    roi_stable_frames += 1
                    if (not roi_active) and roi_stable_frames >= roi_ready_frames:
                        roi_active = True
                        roi_fail_count = 0
                state = tracker.get_state()
                vmag = state['velocity_magnitude'] if state else 0.0
                kalman_ok = state['kalman_stable'] if state else True
                if last_stable is None:
                    last_stable = (x, y)
                if anchor is None:
                    anchor = (x, y)
                use_x, use_y = x, y
                # Update last_det for visual overlay and det_ok flag
                det_ok = False
                if ball_detection:
                    bx, by, bw, bh, bconf = ball_detection
                    last_det = (bx, by)
                    det_ok = True

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
                    cur_step = int(max_pan_step * (1.0 + max(0.0, current_zoom_level - 1.0) * 2.0 + min(vmag / 350.0, 1.2)))
                    if dist > cur_step and dist > 1e-6:
                        r = cur_step / dist
                        anchor = (anchor[0] + dx * r, anchor[1] + dy * r)
                    else:
                        a_anch = 0.08 + 0.14 * max(0.0, current_zoom_level - 1.0)
                        if a_anch > 0.32:
                            a_anch = 0.32
                        anchor = (anchor[0] * (1.0 - a_anch) + use_x * a_anch, anchor[1] * (1.0 - a_anch) + use_y * a_anch)
                    use_x, use_y = anchor

                if is_tracking:
                    if center_lp is None:
                        center_lp = (use_x, use_y)
                    zfac_c = max(0.0, min(1.0, current_zoom_level - 1.0))
                    ac = 0.18 - 0.10 * zfac_c
                    if ac < 0.08:
                        ac = 0.08
                    if ac > 0.22:
                        ac = 0.22
                    cx = center_lp[0] * (1.0 - ac) + use_x * ac
                    cy = center_lp[1] * (1.0 - ac) + use_y * ac
                    center_lp = (cx, cy)
                    use_x, use_y = center_lp

                if ball_detection and last_det is not None:
                    dvx = ball_detection[0] - last_det[0]
                    dvy = ball_detection[1] - last_det[1]
                    vhx, vhy = dvx, dvy
                else:
                    vhx, vhy = tracker.get_velocity()
                if is_tracking:
                    crop_coords = virtual_camera.update(use_x, use_y, time.time(), velocity_hint=(vhx, vhy))
                else:
                    crop_coords = virtual_camera.get_current_crop()
                detection_count += 1
                lost_count = 0

                follow_cx, follow_cy = None, None
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
                    dirn = math.hypot(dxn, dyn)
                    if dirn > 1e-6:
                        ux, uy = dxn/dirn, dyn/dirn
                        lead_px = int(12 + 24*max(0.0, current_zoom_level - 1.0) + min(vmag/18.0, 32.0))
                        fx = (anchor[0] if anchor else x) + ux*lead_px
                        fy = (anchor[1] if anchor else y) + uy*lead_px
                        follow_cx, follow_cy = int(max(0, min(reader.width-1, fx))), int(max(0, min(reader.height-1, fy)))
                    if cosd <= -0.4 and step > close_thresh*0.8:
                        bloom_counter = bloom_max
                last_raw = (x, y)
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
                # No tracking - keep current position and zoom out
                crop_coords = virtual_camera.get_current_crop()
                lost_count += 1
                frames_tracking = 0
                target_zoom_level = 1.0
                zoom_lock_count = 0
                hold_zoom_level = 1.0
                roi_active = False
            
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
            # Clamp zoom center to safe margins to avoid edge jitter/dark areas
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
                step_lim = int(max_crop_step_base * (1.0 + max(0.0, current_zoom_level - 1.0) * 1.5))
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
            else:
                center_x = cropped.shape[1] // 2
                center_y = cropped.shape[0] // 2
                cv2.putText(cropped, "SEARCHING...",
                           (center_x - 100, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

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
            
            if bloom_counter > 0:
                intensity = bloom_counter / bloom_max
                blurred = cv2.GaussianBlur(cropped, (0, 0), sigmaX=6, sigmaY=6)
                cropped = cv2.addWeighted(cropped, 1.0, blurred, 0.35*intensity, 0)
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
            
            display_frame = cv2.resize(cropped, (854, 480), interpolation=cv2.INTER_LINEAR)
            
            mjpeg_server.update_frame(display_frame)
            
            frame_count += 1
            
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
        reader.release()
        mjpeg_server.stop()
        print("✅ Stream stopped")

if __name__ == "__main__":
    main()
