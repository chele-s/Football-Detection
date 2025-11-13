import cv2
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import logging

logger = logging.getLogger(__name__)

class MJPEGServer:
    def __init__(self, port=8080):
        self.port = port
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
                if self.path == '/stream.mjpg':
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
