import streamlit as st
import cv2
import numpy as np
import time
import threading
from collections import deque
from queue import Queue
import torch
from streamlit_autorefresh import st_autorefresh

from app.inference import BallDetector
from app.tracking import BallTracker
from app.utils import VideoReader, RTMPClient, load_config, merge_configs
from app.camera import VirtualCamera

st.set_page_config(
    page_title="Football Detection",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamProcessor:
    def __init__(self, config):
        self.config = config
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
        
        self.tracker = BallTracker(
            max_lost_frames=config['tracking']['max_lost_frames'],
            min_confidence=config['tracking']['min_confidence'],
            iou_threshold=config['tracking'].get('iou_threshold', 0.3),
            adaptive_noise=True
        )
        
        self.virtual_camera = None
        self.ball_class_id = config['model'].get('ball_class_id', 0)
        
        self.frame_queue = Queue(maxsize=3)
        self.stats_queue = Queue(maxsize=2)
        self.running = False
        self.thread = None
        
        self.fps_history = deque(maxlen=30)
        self.det_history = deque(maxlen=100)
        
    def start(self, video_url: str):
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._process_stream, args=(video_url,))
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _process_stream(self, video_url: str):
        try:
            print(f"[STREAM] Starting stream processing for: {video_url}")
            input_source = video_url
            
            if RTMPClient.is_youtube_url(video_url):
                print("[STREAM] Extracting YouTube stream URL...")
                stream_url = RTMPClient.get_youtube_stream_url(video_url)
                if stream_url is None:
                    print("[STREAM] ERROR: Could not extract YouTube URL")
                    return
                input_source = stream_url
                print(f"[STREAM] YouTube URL extracted successfully")
            
            print(f"[STREAM] Opening video source...")
            reader = VideoReader(input_source)
            print(f"[STREAM] Video source opened: {reader.width}x{reader.height}")
            
            frame_w = reader.width
            frame_h = reader.height
            
            self.virtual_camera = VirtualCamera(
                frame_width=frame_w,
                frame_height=frame_h,
                output_width=1280,
                output_height=720,
                dead_zone_percent=0.10,
                anticipation_factor=0.3,
                zoom_padding=1.5,
                smoothing_freq=20.0,
                use_pid=True,
                prediction_steps=5
            )
            
            frame_count = 0
            last_time = time.time()
            skip_frames = 0
            
            print(f"[STREAM] Starting main loop...")
            
            while self.running:
                ret, frame = reader.read()
                if not ret:
                    print("[STREAM] No more frames (ret=False), stream ended")
                    break
                
                if frame is None:
                    print("[STREAM] Frame is None, skipping")
                    continue
                
                if frame.shape[0] < 100 or frame.shape[1] < 100:
                    print(f"[STREAM] Frame too small {frame.shape}, skipping")
                    continue
                
                start_inf = time.time()
                det_result = self.detector.predict_ball_only(
                    frame, 
                    self.ball_class_id, 
                    return_candidates=True
                )
                inf_time = (time.time() - start_inf) * 1000
                
                if isinstance(det_result, tuple) and len(det_result) == 2:
                    detection, detections_list = det_result
                else:
                    detection = det_result
                    detections_list = None
                
                track_result = self.tracker.update(detection, detections_list)
                
                annotated_frame = frame
                
                if track_result:
                    x, y, is_tracking = track_result
                    
                    box_size = 40
                    x1 = int(x - box_size/2)
                    y1 = int(y - box_size/2)
                    x2 = int(x + box_size/2)
                    y2 = int(y + box_size/2)
                    
                    color = (0, 255, 0) if is_tracking else (0, 165, 255)
                    thickness = 3 if is_tracking else 2
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    cv2.circle(annotated_frame, (int(x), int(y)), 8, color, -1)
                    
                    tracker_state = self.tracker.get_state()
                    avg_conf = tracker_state['avg_confidence']
                    label = f"Ball {avg_conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    vel = tracker_state['velocity_magnitude']
                    cv2.putText(annotated_frame, f"Vel: {vel:.0f}px/s", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.putText(annotated_frame, "No ball detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if track_result:
                    x, y, _ = track_result
                    crop_coords = self.virtual_camera.update(x, y)
                    x1, y1, x2, y2 = crop_coords
                    cropped = annotated_frame[y1:y2, x1:x2]
                    cropped = cv2.resize(cropped, (1280, 720), interpolation=cv2.INTER_LINEAR)
                else:
                    cropped = cv2.resize(annotated_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                
                fps = 1000 / inf_time if inf_time > 0 else 0
                self.fps_history.append(fps)
                
                n_dets = len(detections_list) if detections_list else 0
                self.det_history.append(n_dets)
                
                cv2.putText(cropped, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(cropped, f"Detections: {n_dets}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(cropped, f"Inference: {inf_time:.1f}ms", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                while self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        break
                
                try:
                    self.frame_queue.put_nowait(cropped)
                except:
                    pass
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"[STREAM] Processed {frame_count} frames, FPS: {fps:.1f}, Detections: {n_dets}, Queue size: {self.frame_queue.qsize()}")
                
                if frame_count % 60 == 0:
                    tracker_stats = self.tracker.get_stats()
                    camera_stats = self.virtual_camera.get_stats()
                    
                    stats = {
                        'frame_count': frame_count,
                        'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
                        'avg_detections': np.mean(self.det_history) if self.det_history else 0,
                        'tracking': self.tracker.is_tracking,
                        'predictions_used': tracker_stats['predictions_used'],
                        'successful_tracks': tracker_stats['successful_tracks'],
                        'pid_corrections': camera_stats['pid_corrections']
                    }
                    
                    if self.stats_queue.full():
                        try:
                            self.stats_queue.get_nowait()
                        except:
                            pass
                    self.stats_queue.put(stats)
        
            reader.release()
            print(f"[STREAM] Stream ended normally after {frame_count} frames")
        except Exception as e:
            print(f"[STREAM] ERROR: Stream crashed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            print(f"[STREAM] Stream thread exiting")
    
    def get_frame(self):
        frame = None
        count = 0
        while not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get_nowait()
                count += 1
            except:
                break
        return frame
    
    def get_stats(self):
        if not self.stats_queue.empty():
            return self.stats_queue.get()
        return None

@st.cache_resource
def load_processor():
    model_config = load_config('configs/model_config.yml')
    stream_config = load_config('configs/stream_config.yml')
    config = merge_configs(model_config, stream_config)
    return StreamProcessor(config)

def main():
    st.title("⚽ Football Detection System")
    st.markdown("Real-time ball detection and tracking with RF-DETR")
    
    processor = load_processor()
    
    with st.sidebar:
        st.title("Configuration")
        
        st.info("💡 YouTube bloqueado. Usa videos locales o estos streams de prueba:")
        
        stream_option = st.selectbox(
            "Fuente de video",
            [
                "Custom URL",
                "Test Stream 1 (Mux)",
                "Test Stream 2 (Akamai)",
                "Local File"
            ]
        )
        
        if stream_option == "Test Stream 1 (Mux)":
            video_url = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8"
        elif stream_option == "Test Stream 2 (Akamai)":
            video_url = "https://cph-p2p-msl.akamaized.net/hls/live/2000341/test/master.m3u8"
        elif stream_option == "Local File":
            video_url = st.text_input("Ruta del archivo", value="/content/football.mp4")
        else:
            video_url = st.text_input(
                "Video URL",
                value="https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8",
                help="YouTube URL, local file path, or stream URL"
            )
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.05,
            max_value=0.50,
            value=0.12,
            step=0.01
        )
        
        if st.button("Start Stream", type="primary"):
            if not processor.running:
                processor.detector.set_confidence_threshold(confidence)
                processor.start(video_url)
                st.success("🚀 Stream starting... Wait for video to load")
            else:
                st.warning("Stream is already running")
        
        if st.button("Stop Stream"):
            processor.stop()
            st.warning("Stream stopped")
        
        st.divider()
        
        if processor.running and processor.thread:
            if processor.thread.is_alive():
                st.success("🟢 Stream Running")
            else:
                st.error("🔴 Stream Crashed")
        else:
            st.info("⚪ Stream Not Started")
        
        st.divider()
        
        st.subheader("Model Info")
        model_info = processor.detector.get_model_info()
        
        st.metric("Device", model_info.get('device', 'N/A'))
        st.metric("Model", model_info.get('model_type', 'N/A'))
        st.metric("Image Size", model_info.get('image_size', 'N/A'))
        
        if model_info.get('cuda_available'):
            gpu_name = model_info.get('gpu_name', 'N/A')
            gpu_mem = model_info.get('gpu_memory_gb', 0)
            st.metric("GPU", f"{gpu_name}")
            st.metric("VRAM", f"{gpu_mem:.1f} GB")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Stream")
        frame_placeholder = st.empty()
    
    with col2:
        st.subheader("Statistics")
        stats_placeholder = st.empty()
    
    st_autorefresh(interval=50, key="stream_refresh")
    
    frame = processor.get_frame()
    
    if frame is not None:
        h, w = frame.shape[:2]
        if w > 800:
            scale = 800 / w
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    elif processor.running:
        frame_placeholder.info("⏳ Loading stream...")
    
    stats = processor.get_stats()
    
    if stats:
        with stats_placeholder.container():
            st.metric("Frames Processed", f"{stats['frame_count']:,}")
            st.metric("FPS", f"{stats['avg_fps']:.1f}")
            st.metric("Avg Detections", f"{stats['avg_detections']:.1f}")
            
            st.divider()
            
            tracking_status = "🟢 Active" if stats['tracking'] else "🔴 Lost"
            st.metric("Tracking", tracking_status)
            st.metric("Predictions Used", stats['predictions_used'])
            st.metric("Successful Tracks", stats['successful_tracks'])
            st.metric("PID Corrections", stats['pid_corrections'])
    
    if not processor.running and processor.thread and not processor.thread.is_alive():
        st.error("Stream processing stopped")

if __name__ == "__main__":
    main()
