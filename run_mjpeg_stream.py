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
    def __init__(self, min_zoom: float = 1.0, max_zoom: float = 2.5, stiffness: float = 0.08, damping: float = 0.35, max_rate: float = 0.25):
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
        min_confidence=config['tracking'].get('min_confidence', 0.20),
        iou_threshold=config['tracking'].get('iou_threshold', 0.25),
        adaptive_noise=True
    )
    print(f"   → Tracker config: max_lost={tracker.max_lost_frames}, min_conf={tracker.min_confidence:.2f}, iou={tracker.iou_threshold:.2f}")
    
    # Start MJPEG server
    print("[4/5] Starting MJPEG server on port 8554...")
    mjpeg_server = MJPEGServer(port=8554)
    mjpeg_server.start()
    print("✅ MJPEG server started!")
    print("📺 Stream URL: http://localhost:8554/stream.mjpg")
    print("💡 En Colab, usa ngrok para exponer el puerto 8554")
    
    # Open video
    video_path = '/content/football.mp4'
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
        dead_zone_percent=camera_config.get('dead_zone', 0.10),  # Small dead zone = responsive following
        anticipation_factor=camera_config.get('anticipation', 0.3),  # Moderate anticipation
        zoom_padding=camera_config.get('zoom_padding', 1.2),  # Less padding for tighter frame
        smoothing_freq=camera_config.get('smoothing_freq', 30.0),
        smoothing_min_cutoff=camera_config.get('smoothing_min_cutoff', 1.0),  # Less smoothing = more responsive
        smoothing_beta=camera_config.get('smoothing_beta', 0.007),
        use_pid=True,  # Enable PID control for smooth tracking
        prediction_steps=5  # Some prediction for smooth movement
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
    max_zoom_level = 2.5
    frames_tracking = 0
    frames_required_for_zoom = 10

    zoom = SmoothZoom(min_zoom=1.0, max_zoom=max_zoom_level, stiffness=0.085, damping=0.42, max_rate=0.16)
    diag = int(math.hypot(reader.width, reader.height))
    stable_step_px = max(6, int(diag * 0.008))
    jump_reset_px = max(36, int(diag * 0.050))
    cooldown_max = 18
    cooldown = 0
    last_stable = None
    stability_score = 0.0
    anchor = None
    max_pan_step = max(10, int(diag * 0.016))
    anchor_ready_px = max(32, int(diag * 0.060))
    zoom_lock_count = 0
    zoom_lock_max = 24
    hold_zoom_level = 1.0
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = reader.read()
            if not ret or frame is None:
                print("📹 Video ended, restarting...")
                reader.release()
                reader = VideoReader(video_path)
                continue
            
            # Detect ball
            start_inf = time.time()
            det_result = detector.predict_ball_only(
                frame, 
                ball_class_id, 
                return_candidates=True
            )
            inf_time = (time.time() - start_inf) * 1000
            
            # Parse detection  
            # det_result is always (detection_tuple, all_detections) when return_candidates=True
            ball_detection, all_detections = det_result
            
            # Update tracker
            track_result = tracker.update(ball_detection, all_detections)
            
            # Initialize camera on first valid detection
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
                state = tracker.get_state()
                vmag = state['velocity_magnitude'] if state else 0.0
                kalman_ok = state['kalman_stable'] if state else True
                if last_stable is None:
                    last_stable = (x, y)
                if anchor is None:
                    anchor = (x, y)
                use_x, use_y = x, y

                if is_tracking and kalman_ok and cooldown == 0:
                    d = math.hypot(x - last_stable[0], y - last_stable[1])
                    if d > jump_reset_px:
                        cooldown = cooldown_max
                        frames_tracking = 0
                        stability_score = max(stability_score - 0.4, 0.0)
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
                    frames_tracking = 0 if not is_tracking else frames_tracking
                    stability_score = max(stability_score - 0.15, 0.0)

                if anchor is not None:
                    dx = use_x - anchor[0]
                    dy = use_y - anchor[1]
                    dist = math.hypot(dx, dy)
                    if dist > max_pan_step and dist > 1e-6:
                        r = max_pan_step / dist
                        anchor = (anchor[0] + dx * r, anchor[1] + dy * r)
                    else:
                        anchor = (use_x, use_y)
                    use_x, use_y = anchor

                crop_coords = virtual_camera.update(use_x, use_y, time.time(), velocity_hint=tracker.get_velocity())
                detection_count += 1
                lost_count = 0

                dist_anchor_ball = math.hypot((anchor[0] if anchor else x) - x, (anchor[1] if anchor else y) - y)
                zoom_gate_ok = (stability_score >= 0.50 or frames_tracking >= frames_required_for_zoom) and cooldown == 0 and dist_anchor_ball <= anchor_ready_px
                if zoom_gate_ok:
                    if vmag > 950:
                        target_zoom_level = 1.30
                    elif vmag > 650:
                        target_zoom_level = 1.50
                    elif vmag > 380:
                        target_zoom_level = 1.75
                    else:
                        target_zoom_level = 2.15
                    zoom_lock_count = zoom_lock_max
                    hold_zoom_level = target_zoom_level
                else:
                    if zoom_lock_count > 0:
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
            
            # Fast zoom transition (2 frames)
            zoom.set_target(target_zoom_level)
            current_zoom_level = zoom.update()
            
            x1, y1, x2, y2 = crop_coords
            if track_result:
                zoom_cx = int(anchor[0] if anchor else (x1 + x2) // 2)
                zoom_cy = int(anchor[1] if anchor else (y1 + y2) // 2)
            else:
                zoom_cx = (x1 + x2) // 2
                zoom_cy = (y1 + y2) // 2

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
            
            cropped = frame[y1:y2, x1:x2].copy()
            
            # Draw detection on cropped frame
            if track_result:
                x, y, is_tracking = track_result
                # Convert absolute coordinates to relative (within crop)
                rel_x = int(x - x1)
                rel_y = int(y - y1)
                
                # Draw tracking indicator
                if is_tracking:
                    # Active tracking - green circle and crosshair
                    cv2.circle(cropped, (rel_x, rel_y), 15, (0, 255, 0), 3)
                    cv2.circle(cropped, (rel_x, rel_y), 5, (0, 255, 0), -1)
                    # Crosshair
                    cv2.line(cropped, (rel_x - 20, rel_y), (rel_x + 20, rel_y), (0, 255, 0), 2)
                    cv2.line(cropped, (rel_x, rel_y - 20), (rel_x, rel_y + 20), (0, 255, 0), 2)
                else:
                    # Predicted position - yellow circle
                    cv2.circle(cropped, (rel_x, rel_y), 15, (0, 255, 255), 3)
                    cv2.circle(cropped, (rel_x, rel_y), 5, (0, 255, 255), -1)
                
                # Draw bounding box if we have detection
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
                    
                    # Confidence label
                    cv2.putText(cropped, f"Ball: {conf:.2f}",
                               (bbox_x - bbox_w//2, bbox_y - bbox_h//2 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # No tracking - draw "LOST" indicator in center
                center_x = cropped.shape[1] // 2
                center_y = cropped.shape[0] // 2
                cv2.putText(cropped, "SEARCHING...",
                           (center_x - 100, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Calculate FPS
            loop_time = (time.time() - loop_start) * 1000
            fps = 1000 / loop_time if loop_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            
            avg_fps = np.mean(fps_history)
            
            # Overlay stats with background for readability
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
            
            # Show detection stats and zoom level
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
            
            # Show tracking confidence
            if track_result:
                x, y, is_tracking = track_result
                if is_tracking and frames_tracking > 0:
                    # Active tracking - show progress
                    track_color = (0, 255, 0) if frames_tracking >= frames_required_for_zoom else (255, 255, 0)
                    track_text = f"Lock: {min(frames_tracking, frames_required_for_zoom)}/{frames_required_for_zoom}"
                    cv2.putText(cropped, track_text, (cropped.shape[1] - 150, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 2)
                elif not is_tracking:
                    # Predicting - show warning
                    cv2.putText(cropped, "PRED (0/15)", (cropped.shape[1] - 150, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            # Upscale for better visibility - 854x480 target
            display_frame = cv2.resize(cropped, (854, 480), interpolation=cv2.INTER_LINEAR)
            
            # Update MJPEG server
            mjpeg_server.update_frame(display_frame)
            
            frame_count += 1
            
            # Log every 30 frames
            if frame_count % 30 == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                actual_fps = 30 / elapsed if elapsed > 0 else 0
                
                # Detailed tracking stats
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
