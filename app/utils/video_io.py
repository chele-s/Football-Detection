import cv2
import numpy as np
import subprocess
import time
import logging
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoReader:
    def __init__(self, source: str, reconnect: bool = False):
        self.source = source
        self.reconnect = reconnect
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            if self.reconnect:
                logger.warning(f"Could not open {source}, entering reconnection loop...")
                self._reconnect()
            else:
                raise ValueError(f"No se pudo abrir: {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.frame_count = 0

    def _reconnect(self):
        while True:
            try:
                logger.info(f"Attempting to reconnect to {self.source}...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.source)
                if self.cap.isOpened():
                    logger.info("Reconnection successful!")
                    # Update properties in case they changed
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                    self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    return
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")
            
            logger.info("Waiting 5 seconds before next attempt...")
            time.sleep(5)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        
        if not ret and self.reconnect:
            logger.warning("Stream disconnected or empty frame, attempting to reconnect...")
            self._reconnect()
            ret, frame = self.cap.read()
            
        if ret:
            self.frame_count += 1
        return ret, frame
    
    def get_progress(self) -> float:
        if self.total_frames > 0:
            return self.frame_count / self.total_frames
        return 0.0
    
    def release(self):
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float = 30.0,
        codec: str = 'mp4v'
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"No se pudo crear: {output_path}")
    
    def write(self, frame: np.ndarray):
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FFMPEGWriter:
    def __init__(
        self,
        output_url: str,
        width: int,
        height: int,
        fps: float = 30.0,
        bitrate: str = '4000k',
        preset: str = 'ultrafast',
        codec: str = 'libx264'
    ):
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        
        command = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',
            '-c:v', codec,
            '-pix_fmt', 'yuv420p',
            '-preset', preset,
            '-tune', 'zerolatency',
            '-g', str(int(max(2, fps) * 2)),
            '-maxrate', bitrate,
            '-bufsize', bitrate,
            '-an',
            '-b:v', bitrate,
            '-f', 'flv' if 'rtmp' in output_url else 'mp4',
            output_url
        ]
        
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    
    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("[FFMPEG] Pipe rota, cerrando...")
    
    def release(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
