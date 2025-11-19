import subprocess
import threading
import os
import shutil
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)

class HLSStreamer:
    def __init__(self, output_dir: str, width: int, height: int, fps: int = 30):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.running = False
        
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                logger.warning(f"Could not clear HLS directory: {e}")
        os.makedirs(output_dir, exist_ok=True)
        
        self.playlist_path = os.path.join(output_dir, 'stream.m3u8')
        
    def start(self):
        if self.running:
            return
        encoders = [
            ('h264_nvenc', ['-preset', 'p1', '-tune', 'zerolatency']),
            ('libx264', ['-preset', 'ultrafast', '-tune', 'zerolatency'])
        ]
        
        for encoder, encoder_opts in encoders:
            cmd = [
                'ffmpeg',
                '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', encoder
            ] + encoder_opts + [
                '-g', str(self.fps * 2),
                '-sc_threshold', '0',
                '-f', 'hls',
                '-hls_time', '1',
                '-hls_list_size', '5',
                '-hls_flags', 'delete_segments',
                '-hls_segment_filename', os.path.join(self.output_dir, 'segment_%03d.ts'),
                self.playlist_path
            ]
            
            try:
                logger.info(f"Attempting to start HLS streamer with {encoder}...")
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE
                )
                
                try:
                    outs, errs = self.process.communicate(timeout=0.5)
                    if self.process.returncode != 0:
                        raise Exception(f"Encoder {encoder} failed: {errs}")
                except subprocess.TimeoutExpired:
                    self.running = True
                    logger.info(f"HLS Streamer started using {encoder}. Output: {self.output_dir}")
                    return
                    
            except Exception as e:
                logger.warning(f"Failed to start with {encoder}: {e}")
                if self.process:
                    self.process.kill()
                    self.process = None
        
        logger.error("All HLS encoders failed!")
        self.running = False

    def update_frame(self, frame: np.ndarray):
        if not self.running or self.process is None:
            return
            
        try:
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except BrokenPipeError:
            logger.error("FFmpeg process pipe broken. Restarting...")
            self.stop()
            self.start()
        except Exception as e:
            logger.error(f"Error writing frame to HLS: {e}")

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                self.process.kill()
            self.process = None
        logger.info("HLS Streamer stopped")
