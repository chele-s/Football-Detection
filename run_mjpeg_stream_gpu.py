#!/usr/bin/env python3
"""
GPU-Accelerated MJPEG Stream Server
Zero-copy pipeline: NVDEC → PyTorch → RF-DETR → Display
3-5x faster than CPU version
"""
import cv2
import numpy as np
import time
import torch
from pathlib import Path
import math
import sys

from app.Inference import BallDetector
from app.tracking import BallTracker
from app.utils import load_config, merge_configs, MJPEGServer, RTMPClient
from app.camera import VirtualCamera

# Check GPU availability
try:
    from app.utils import GPU_AVAILABLE, GPUVideoReader, GPUTensorOps
    if not GPU_AVAILABLE:
        print("❌ GPU pipeline not available. Install PyNvCodec first.")
        print("Run: python colab_install_pynvcodec_fixed.py")
        sys.exit(1)
except ImportError:
    print("❌ GPU utils not found. Make sure gpu_video_io.py is in app/utils/")
    sys.exit(1)


class SmoothZoom:
    """Smooth zoom controller (unchanged from CPU version)"""
    def __init__(self, min_zoom: float = 1.0, max_zoom: float = 2.5, 
                 stiffness: float = 0.08, damping: float = 0.35, 
                 max_rate: float = 0.25):
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
    print("🚀 GPU-ACCELERATED MJPEG Stream Server")
    print("="*60)
    
    # Load configs
    print("\n[1/5] Loading configurations...")
    model_config = load_config('configs/model_config.yml')
    stream_config = load_config('configs/stream_config.yml')
    config = merge_configs(model_config, stream_config)
    
    # Initialize detector
    print("[2/5] Loading RF-DETR model...")
    model_path = config['model']['path']
    
    detector = BallDetector(
        model_path=model_path,
        confidence_threshold=config['model']['confidence'],
        iou_threshold=config['model'].get('iou_threshold', 0.45),
        device='cuda',
        half_precision=config['model'].get('half_precision', True),
        imgsz=config['model'].get('imgsz', 640),
        warmup_iterations=3
    )
    print(f"   ✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    
    # Initialize tracker
    print("[3/5] Initializing tracker...")
    tracker = BallTracker(
        max_lost_frames=config['tracking'].get('max_lost_frames', 10),
        min_confidence=config['tracking'].get('min_confidence', 0.10),
        iou_threshold=config['tracking'].get('iou_threshold', 0.20),
        adaptive_noise=True
    )
    
    # Start MJPEG server
    mjpeg_port = config.get('stream', {}).get('mjpeg_port', 8554)
    print(f"[4/5] Starting MJPEG server on port {mjpeg_port}...")
    mjpeg_server = MJPEGServer(port=mjpeg_port)
    mjpeg_server.start()
    print("✅ MJPEG server started!")
    print(f"📺 Stream URL: http://localhost:8080/stream.mjpg")
    
    # Open video with GPU decoder (NVDEC)
    video_path = config.get('stream', {}).get('input_url', '/content/football.mp4')

    if RTMPClient.is_youtube_url(video_path):
        print("YouTube URL detected. Resolviendo stream...")
        stream_url = RTMPClient.get_youtube_stream_url(video_path)
        if stream_url is None:
            raise ValueError("No se pudo obtener el stream de YouTube")
        video_path = stream_url
        print("✓ Stream directo obtenido")

    print(f"\n[5/5] Opening video with GPU decoder (NVDEC)...")
    print(f"   Source: {video_path}")
    
    reader = GPUVideoReader(video_path, device=0)
    print(f"✅ GPU decoder ready: {reader.width}x{reader.height} @ {reader.fps:.1f}fps")
    
    # Initialize virtual camera
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
        smoothing_freq=30.0,
        smoothing_min_cutoff=2.5,
        smoothing_beta=0.001,
        use_pid=True,
        prediction_steps=5
    )
    
    ball_class_id = config['model'].get('ball_class_id', 0)
    
    print("\n" + "="*60)
    print("🎬 Starting GPU pipeline...")
    print("="*60)
    print("⚡ All frames stay in GPU memory (VRAM)")
    print("⚡ Zero CPU↔GPU copies")
    print("Press Ctrl+C to stop\n")
    
    # Pipeline state
    frame_count = 0
    last_log_time = time.time()
    fps_history = []
    camera_initialized = False
    detection_count = 0
    
    # Zoom system
    current_zoom_level = 1.0
    target_zoom_level = 1.0
    max_zoom_level = 1.8
    frames_tracking = 0
    zoom = SmoothZoom(min_zoom=1.0, max_zoom=max_zoom_level)
    
    # ROI optimization
    roi_active = False
    roi_stable_frames = 0
    roi_ready_frames = 40
    roi_fail_count = 0
    roi_fail_max = 150
    roi_last_valid_pos = None
    prev_crop = None
    
    # Performance tracking
    decode_times = []
    inference_times = []
    crop_times = []
    
    device = torch.device('cuda:0')
    
    try:
        while True:
            loop_start = time.time()
            
            # ===== DECODE ON GPU (NVDEC) =====
            t_decode_start = time.time()
            ret, frame_tensor = reader.read()  # Returns torch.Tensor [3, H, W] in VRAM
            t_decode = (time.time() - t_decode_start) * 1000
            decode_times.append(t_decode)
            if len(decode_times) > 100:
                decode_times.pop(0)
            
            if not ret:
                print("✅ End of stream alcanzado. Cerrando loop.")
                break
            
            # frame_tensor is [3, H, W] in VRAM, RGB, float [0..1]
            _, h, w = frame_tensor.shape
            
            # ===== INFERENCE ON GPU =====
            t_inf_start = time.time()
            use_roi = False
            offx, offy = 0, 0
            
            if prev_crop is not None and roi_active:
                # Crop ROI in VRAM (zero-cost indexing)
                rx1, ry1, rx2, ry2 = prev_crop
                rx1 = max(0, min(w-2, int(rx1)))
                ry1 = max(0, min(h-2, int(ry1)))
                rx2 = max(rx1+2, min(w, int(rx2)))
                ry2 = max(ry1+2, min(h, int(ry2)))
                
                frame_roi = GPUTensorOps.crop(frame_tensor, rx1, ry1, rx2, ry2)
                use_roi = True
                offx, offy = rx1, ry1
                det_result = detector.predict_ball_only(
                    frame_roi, ball_class_id,
                    use_temporal_filtering=False,
                    return_candidates=True
                )
            else:
                det_result = detector.predict_ball_only(
                    frame_tensor, ball_class_id,
                    use_temporal_filtering=True,
                    return_candidates=True
                )
            
            t_inf = (time.time() - t_inf_start) * 1000
            inference_times.append(t_inf)
            if len(inference_times) > 100:
                inference_times.pop(0)
            
            ball_detection, all_detections = det_result
            
            # Map ROI coordinates to global
            if use_roi:
                if ball_detection is not None:
                    bx, by, bw, bh, bc = ball_detection
                    ball_detection = (bx + offx, by + offy, bw, bh, bc)
                if all_detections:
                    mapped = []
                    for d in all_detections:
                        mapped.append((d[0] + offx, d[1] + offy, d[2], d[3], d[4], d[5]))
                    all_detections = mapped
                
                # ROI stability check
                detection_viable = False
                if ball_detection is not None:
                    bx, by = ball_detection[0], ball_detection[1]
                    if roi_last_valid_pos is not None:
                        dx = bx - roi_last_valid_pos[0]
                        dy = by - roi_last_valid_pos[1]
                        jump_dist = math.hypot(dx, dy)
                        detection_viable = jump_dist < math.hypot(w, h) * 0.3
                    else:
                        detection_viable = True
                    
                    if detection_viable and all_detections and len(all_detections) > 3:
                        detection_viable = False
                    
                    if detection_viable:
                        roi_last_valid_pos = (bx, by)
                
                if not detection_viable:
                    roi_fail_count += 1
                else:
                    roi_fail_count = 0
                
                if roi_fail_count >= roi_fail_max:
                    print("⚠️  ROI lost - switching to full-frame")
                    roi_active = False
                    roi_fail_count = 0
                    roi_stable_frames = 0
                    roi_last_valid_pos = None
            
            # Spatial filtering (reject stands/lights)
            apply_spatial_filter = (not roi_active) and (current_zoom_level < 1.4)
            if apply_spatial_filter:
                if ball_detection is not None:
                    bx, by = ball_detection[0], ball_detection[1]
                    if by < h * 0.25 or bx < w * 0.15 or bx > w * 0.85:
                        ball_detection = None
                
                if all_detections:
                    filtered = [d for d in all_detections 
                               if d[1] >= h*0.25 and w*0.15 <= d[0] <= w*0.85]
                    all_detections = filtered if filtered else None
            
            # ===== TRACKING (CPU) =====
            track_result = tracker.update(ball_detection, all_detections)
            
            # ===== CAMERA CONTROL =====
            if not camera_initialized and track_result:
                x, y, is_tracking = track_result
                virtual_camera.reset()
                crop_coords = virtual_camera.update(x, y, time.time(), 
                                                   velocity_hint=tracker.get_velocity())
                camera_initialized = True
            elif track_result:
                x, y, is_tracking = track_result
                
                if not is_tracking:
                    roi_active = False
                    roi_stable_frames = 0
                    frames_tracking = 0
                else:
                    frames_tracking += 1
                    roi_stable_frames += 1
                    if not roi_active and roi_stable_frames >= roi_ready_frames:
                        roi_active = True
                        roi_fail_count = 0
                        print("✓ ROI activated")
                
                # Zoom logic
                state = tracker.get_state()
                vmag = state['velocity_magnitude'] if state else 0.0
                
                if is_tracking and frames_tracking >= 8:
                    if vmag > 900:
                        target_zoom_level = 1.20
                    elif vmag > 650:
                        target_zoom_level = 1.35
                    elif vmag > 380:
                        target_zoom_level = 1.50
                    else:
                        target_zoom_level = 1.65
                else:
                    target_zoom_level = 1.0
                
                vhx, vhy = tracker.get_velocity()
                crop_coords = virtual_camera.update(x, y, time.time(), 
                                                   velocity_hint=(vhx, vhy))
                detection_count += 1
            else:
                frames_tracking = 0
                target_zoom_level = 1.0
                roi_active = False
                roi_stable_frames = 0
                crop_coords = virtual_camera.get_current_crop() if camera_initialized else (0, 0, w, h)
            
            zoom.set_target(target_zoom_level)
            current_zoom_level = zoom.update()
            
            x1, y1, x2, y2 = crop_coords
            
            # Apply zoom
            if current_zoom_level > 1.01:
                crop_width = x2 - x1
                crop_height = y2 - y1
                zoomed_width = int(crop_width / current_zoom_level)
                zoomed_height = int(crop_height / current_zoom_level)
                
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                x1 = max(0, cx - zoomed_width // 2)
                y1 = max(0, cy - zoomed_height // 2)
                x2 = min(w, x1 + zoomed_width)
                y2 = min(h, y1 + zoomed_height)
                
                if x2 - x1 < zoomed_width:
                    x1 = max(0, x2 - zoomed_width)
                if y2 - y1 < zoomed_height:
                    y1 = max(0, y2 - zoomed_height)
            
            x1 = max(0, min(w-2, int(x1)))
            y1 = max(0, min(h-2, int(y1)))
            x2 = max(x1+2, min(w, int(x2)))
            y2 = max(y1+2, min(h, int(y2)))
            
            # Smooth crop transitions
            if prev_crop is not None:
                px1, py1, px2, py2 = prev_crop
                alpha = 0.15
                x1 = int(px1 * (1-alpha) + x1 * alpha)
                y1 = int(py1 * (1-alpha) + y1 * alpha)
                x2 = int(px2 * (1-alpha) + x2 * alpha)
                y2 = int(py2 * (1-alpha) + y2 * alpha)
            
            prev_crop = (x1, y1, x2, y2)
            
            # ===== CROP & RESIZE ON GPU =====
            t_crop_start = time.time()
            cropped_tensor = GPUTensorOps.crop_and_resize(
                frame_tensor,
                x1, y1, x2, y2,
                (480, 854),  # Output size for MJPEG
                mode='bilinear'
            )
            t_crop = (time.time() - t_crop_start) * 1000
            crop_times.append(t_crop)
            if len(crop_times) > 100:
                crop_times.pop(0)
            
            # ===== CONVERT TO CPU FOR OVERLAY & MJPEG =====
            # This is the ONLY GPU→CPU copy in the entire pipeline
            display_frame = GPUTensorOps.to_numpy_cpu(cropped_tensor)
            
            # Draw overlays
            if track_result:
                x, y, is_tracking = track_result
                rel_x = int((x - x1) / (x2 - x1) * 854)
                rel_y = int((y - y1) / (y2 - y1) * 480)
                
                color = (0, 255, 0) if is_tracking else (0, 255, 255)
                cv2.circle(display_frame, (rel_x, rel_y), 15, color, 3)
                cv2.circle(display_frame, (rel_x, rel_y), 5, color, -1)
            
            # Stats overlay
            loop_time = (time.time() - loop_start) * 1000
            fps = 1000 / loop_time if loop_time > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            
            avg_fps = np.mean(fps_history)
            avg_decode = np.mean(decode_times) if decode_times else 0
            avg_inference = np.mean(inference_times) if inference_times else 0
            avg_crop = np.mean(crop_times) if crop_times else 0
            
            # Stats background
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (5, 5), (350, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f} [GPU]", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Decode: {avg_decode:.1f}ms" if decode_times else "Decode: N/A", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display_frame, f"Inference: {avg_inference:.1f}ms" if inference_times else "Inference: N/A", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(display_frame, f"Crop: {avg_crop:.1f}ms" if crop_times else "Crop: N/A", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            if track_result:
                status = "TRACKING" if track_result[2] else "PREDICTING"
                color = (0, 255, 0) if track_result[2] else (0, 255, 255)
                cv2.putText(display_frame, f"Status: {status}", (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Zoom indicator
            if current_zoom_level > 1.05:
                zoom_text = f"ZOOM: {current_zoom_level:.1f}x"
                cv2.putText(display_frame, zoom_text, (704, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Send to MJPEG server
            mjpeg_server.update_frame(display_frame)
            
            frame_count += 1
            
            # Periodic logging
            if frame_count % 30 == 0:
                elapsed = time.time() - last_log_time
                actual_fps = 30 / elapsed if elapsed > 0 else 0
                
                status = "TRACK" if track_result and track_result[2] else "PRED" if track_result else "LOST"
                decode_str = f"Decode: {avg_decode:4.1f}ms" if decode_times else "Decode: N/A"
                inf_str = f"Inf: {avg_inference:4.1f}ms" if inference_times else "Inf: N/A"
                print(f"[GPU] Frame {frame_count:4d} | FPS: {actual_fps:5.1f} | {decode_str} | {inf_str} | "
                      f"Status: {status} | Zoom: {current_zoom_level:.2f}x")
                
                last_log_time = time.time()
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping GPU stream...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader.release()
        mjpeg_server.stop()
        print("✅ GPU stream stopped")
        
        # Final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS (GPU PIPELINE)")
        print("="*60)
        print(f"Total frames: {frame_count}")
        if decode_times:
            print(f"Avg decode time: {np.mean(decode_times):.2f}ms")
        else:
            print("Avg decode time: N/A")

        if inference_times:
            print(f"Avg inference time: {np.mean(inference_times):.2f}ms")
        else:
            print("Avg inference time: N/A")

        if crop_times:
            print(f"Avg crop time: {np.mean(crop_times):.2f}ms")
        else:
            print("Avg crop time: N/A")
        
        print(f"Detections: {detection_count}")
        print("="*60)


if __name__ == "__main__":
    main()