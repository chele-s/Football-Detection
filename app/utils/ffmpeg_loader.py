import subprocess
import numpy as np
import json
import os
import signal

class FFmpegVideoLoader:
    def __init__(self, source, gpu_id=0):
        self.source = source
        self.gpu_id = gpu_id
        self.process = None
        self.width = 0
        self.height = 0
        self.fps = 30.0
        self.frame_size = 0
        self._probe()
        self._launch()

    def _probe(self):
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate', '-of', 'json', self.source]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(res.stdout)
            stream = data['streams'][0]
            self.width = int(stream['width'])
            self.height = int(stream['height'])
            n, d = map(int, stream['r_frame_rate'].split('/'))
            self.fps = n / d if d > 0 else 30.0
            self.frame_size = self.width * self.height * 3
        except Exception:
            raise RuntimeError("Probe failed")

    def _launch(self):
        cmd = ['ffmpeg', '-hwaccel', 'cuda', '-hwaccel_device', str(self.gpu_id), '-hwaccel_output_format', 'cuda', '-i', self.source, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-vf', 'hwdownload,format=bgr24', '-vsync', '0', '-v', 'error', 'pipe:1']
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)

    def read(self):
        if not self.process: return False, None
        try:
            raw = self.process.stdout.read(self.frame_size)
            if len(raw) != self.frame_size:
                self.release()
                return False, None
            return True, np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        except Exception:
            self.release()
            return False, None

    def release(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def __del__(self):
        self.release()
