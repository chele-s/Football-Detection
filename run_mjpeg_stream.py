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
    
    # Initialize tracker
    print("[3/5] Initializing ball tracker...")
    tracker = BallTracker(
        max_lost_frames=config['tracking']['max_lost_frames'],
        min_confidence=config['tracking']['min_confidence'],
        iou_threshold=config['tracking'].get('iou_threshold', 0.3),
        adaptive_noise=True
    )
    
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
    # Use 1600x900 for high quality tracking window
    output_width = 1600
    output_height = 900
    camera_config = config.get('camera', {})
    
    virtual_camera = VirtualCamera(
        frame_width=reader.width,
        frame_height=reader.height,
        output_width=output_width,
        output_height=output_height,
        dead_zone_percent=camera_config.get('dead_zone', 0.10),
        anticipation_factor=camera_config.get('anticipation', 0.3),
        zoom_padding=camera_config.get('zoom_padding', 1.2),
        smoothing_freq=camera_config.get('smoothing_freq', 30.0),
        smoothing_min_cutoff=camera_config.get('smoothing_min_cutoff', 1.0),
        smoothing_beta=camera_config.get('smoothing_beta', 0.007)
    )
    
    ball_class_id = config['model'].get('ball_class_id', 0)
    
    print("\n" + "="*60)
    print("🚀 Starting processing loop...")
    print("="*60)
    print("Press Ctrl+C to stop\n")
    
    frame_count = 0
    last_log_time = time.time()
    fps_history = []
    
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
            
            # Update virtual camera
            if track_result:
                x, y, is_tracking = track_result
                crop_coords = virtual_camera.update(x, y, time.time())
            else:
                crop_coords = virtual_camera.get_current_crop()
            
            # Crop frame
            x1, y1, x2, y2 = crop_coords
            cropped = frame[y1:y2, x1:x2].copy()
            
            # Draw detection on cropped frame
            if ball_detection and track_result:
                bx, by, bw, bh, conf = ball_detection
                rel_x = int(bx - x1)
                rel_y = int(by - y1)
                rel_w = int(bw)
                rel_h = int(bh)
                
                cv2.rectangle(cropped, 
                             (rel_x - rel_w//2, rel_y - rel_h//2),
                             (rel_x + rel_w//2, rel_y + rel_h//2),
                             (0, 255, 0), 2)
                cv2.circle(cropped, (rel_x, rel_y), 5, (0, 255, 0), -1)
                cv2.putText(cropped, f"{conf:.2f}", 
                           (rel_x + 10, rel_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate FPS
            loop_time = (time.time() - loop_start) * 1000
            fps = 1000 / loop_time if loop_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            
            avg_fps = np.mean(fps_history)
            
            # Overlay stats
            cv2.putText(cropped, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cropped, f"Inference: {inf_time:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if track_result:
                cv2.putText(cropped, "Tracking: ACTIVE", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(cropped, "Tracking: LOST", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Update MJPEG server with high quality
            # Keep 1280x720 for better quality (browser will scale if needed)
            mjpeg_server.update_frame(cropped)
            
            frame_count += 1
            
            # Log every 30 frames
            if frame_count % 30 == 0:
                current_time = time.time()
                elapsed = current_time - last_log_time
                actual_fps = 30 / elapsed if elapsed > 0 else 0
                print(f"[STREAM] Frame {frame_count:4d} | FPS: {actual_fps:5.1f} | Inf: {inf_time:5.1f}ms | Loop: {loop_time:5.1f}ms")
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
