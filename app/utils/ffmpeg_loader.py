import subprocess
import numpy as np
import json
import os
import time
import threading
import queue

class FFmpegVideoLoader:
    def __init__(self, source, gpu_id=0):
        self.source = source
        self.gpu_id = gpu_id
        self.process = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.frame_size = 0
        self.running = False
        self._probe()
        self._launch()

    def _probe(self):
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', self.source]
            # startupinfo to hide console window on Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTUPINFO.DWF_USESHOWWINDOW
            
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo)
            data = json.loads(res.stdout)
            stream = data['streams'][0]
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            n, d = map(int, stream['r_frame_rate'].split('/'))
            self.fps = n / d if d > 0 else 30.0
            self.frame_size = self.width * self.height * 3
        except Exception as e:
            raise RuntimeError(f"FFprobe failed: {e}")

    def _launch(self):
        cmd = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_device', str(self.gpu_id), '-hwaccel_output_format', 'cuda', '-i', self.source, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vf', 'hwdownload,format=bgr24', '-vsync', '0', '-v', 'error', 'pipe:1']
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTUPINFO.DWF_USESHOWWINDOW

        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            bufsize=10**7,
            startupinfo=startupinfo
        )
        self.running = True

    def read(self):
        if not self.process or not self.running:
            return False, None
        
        try:
            raw = self.process.stdout.read(self.frame_size)
            if not raw or len(raw) != self.frame_size:
                self.running = False
                return False, None
            
            return True, np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        except Exception:
            self.running = False
            return False, None

    def release(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def __del__(self):
        self.release()
