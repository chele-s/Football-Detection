import cv2
import threading
import time
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import logging

logger = logging.getLogger(__name__)

class MJPEGServer:
    def __init__(self, port=8080, hls_dir=None):
        self.port = port
        self.hls_dir = hls_dir
        self.frame = None
        self.lock = threading.Lock()
        self.server = None
        self.thread = None
        self.running = False
        
    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy() if frame is not None else None
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def start(self):
        if self.running:
            return
        
        self.running = True
        
        class StreamHandler(BaseHTTPRequestHandler):
            server_instance = self
            
            def log_message(self, format, *args):
                pass
            
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    try:
                        # Serve the player.html file
                        player_path = os.path.join(os.path.dirname(__file__), 'player.html')
                        if os.path.exists(player_path):
                            with open(player_path, 'rb') as f:
                                self.wfile.write(f.read())
                        else:
                            self.wfile.write(b"<h1>Player not found</h1>")
                    except Exception as e:
                        logger.error(f"Error serving player: {e}")
                elif self.path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Expires', '0')
                    self.end_headers()
                    
                    try:
                        while self.server_instance.running:
                            frame = self.server_instance.get_frame()
                            if frame is not None:
                                _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                                self.wfile.write(b'--frame\r\n')
                                self.send_header('Content-Type', 'image/jpeg')
                                self.send_header('Content-Length', len(jpeg))
                                self.end_headers()
                                self.wfile.write(jpeg.tobytes())
                                self.wfile.write(b'\r\n')
                            time.sleep(0.033)
                    except:
                        pass
                elif self.path.startswith('/hls/') and self.server_instance.hls_dir:
                    try:
                        filename = self.path.split('/')[-1]
                        file_path = os.path.join(self.server_instance.hls_dir, filename)
                        
                        if os.path.exists(file_path):
                            self.send_response(200)
                            if filename.endswith('.m3u8'):
                                self.send_header('Content-type', 'application/vnd.apple.mpegurl')
                            elif filename.endswith('.ts'):
                                self.send_header('Content-type', 'video/MP2T')
                            self.send_header('Access-Control-Allow-Origin', '*')
                            self.end_headers()
                            
                            with open(file_path, 'rb') as f:
                                self.wfile.write(f.read())
                        else:
                            self.send_error(404)
                    except Exception as e:
                        logger.error(f"Error serving HLS file: {e}")
                        self.send_error(500)
                else:
                    self.send_error(404)
        
        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True
        
        StreamHandler.server_instance = self
        self.server = ThreadedHTTPServer(('0.0.0.0', self.port), StreamHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"MJPEG server started on port {self.port}")
    
    def stop(self):
        if self.server:
            self.running = False
            self.server.shutdown()
            self.server.server_close()
            logger.info("MJPEG server stopped")
