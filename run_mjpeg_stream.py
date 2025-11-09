#!/usr/bin/env python3
import cv2
import numpy as np
import time
import torch
from pathlib import Path

from app.inference import BallDetector
from app.tracking import BallTracker
from app.utils import VideoReader, load_config, merge_configs, MJPEGServer
from app.camera import VirtualCamera

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
        max_lost_frames=config['tracking'].get('max_lost_frames', 10),  # Shorter recovery window
        min_confidence=config['tracking'].get('min_confidence', 0.08),  # Lower threshold for marginal detections
        iou_threshold=config['tracking'].get('iou_threshold', 0.2),  # More permissive for fast moving ball
        adaptive_noise=True  # Kalman filter adapts to ball movement
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
    # Use 960x540 for better FPS - tight zoom like a cameraman following the ball
    output_width = 960
    output_height = 540
    camera_config = config.get('camera', {})
    
    virtual_camera = VirtualCamera(
        frame_width=reader.width,
        frame_height=reader.height,
        output_width=output_width,
        output_height=output_height,
        dead_zone_percent=camera_config.get('dead_zone', 0.08),  # Smaller dead zone = more responsive
        anticipation_factor=camera_config.get('anticipation', 0.5),  # High anticipation for fast ball
        zoom_padding=camera_config.get('zoom_padding', 1.1),  # TIGHT zoom - minimal padding
        smoothing_freq=camera_config.get('smoothing_freq', 30.0),
        smoothing_min_cutoff=camera_config.get('smoothing_min_cutoff', 0.8),  # Less smoothing = more responsive
        smoothing_beta=camera_config.get('smoothing_beta', 0.005),
        use_pid=True,  # Enable PID control for smooth tracking
        prediction_steps=8  # Predict more frames ahead for anticipation
    )
    print(f"   → Camera config: {output_width}x{output_height}, dead_zone={0.08:.2f}, zoom_padding={1.1:.1f}")
    
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
                # Initialize camera centered on first ball detection
                virtual_camera.reset()
                # Update camera position
                crop_coords = virtual_camera.update(x, y, time.time())
                camera_initialized = True
                print(f"[CAMERA] Initialized at ball position: ({x:.1f}, {y:.1f})")
            elif track_result:
                x, y, is_tracking = track_result
                crop_coords = virtual_camera.update(x, y, time.time())
                detection_count += 1
                lost_count = 0  # Reset lost counter
            else:
                # No tracking - keep current position
                crop_coords = virtual_camera.get_current_crop()
                lost_count += 1
            
            # Crop frame
            x1, y1, x2, y2 = crop_coords
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
            
            # Show "CAMERAMAN MODE" indicator
            zoom_text = "ZOOM: TIGHT (1.1x)"
            cv2.putText(cropped, zoom_text, (cropped.shape[1] - 220, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Resize for even better FPS - 720x405 is perfect for streaming
            display_frame = cv2.resize(cropped, (720, 405), interpolation=cv2.INTER_LINEAR)
            
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
                    print(f"[STREAM] Frame {frame_count:4d} | FPS: {actual_fps:5.1f} | Inf: {inf_time:5.1f}ms | Track: {tracking_status} ({x:.0f},{y:.0f}) | Dets: {detection_count}")
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
