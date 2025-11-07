import subprocess
import requests
from typing import Optional


class RTMPClient:
    @staticmethod
    def test_connection(url: str, timeout: int = 5) -> bool:
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', url],
                capture_output=True,
                timeout=timeout
            )
            return result.returncode == 0
        except:
            return False
    
    @staticmethod
    def get_youtube_stream_url(youtube_url: str) -> Optional[str]:
        try:
            import yt_dlp
            
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'quiet': True,
                'no_warnings': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info['url']
        
        except ImportError:
            print("[ERROR] yt-dlp no instalado. Ejecuta: pip install yt-dlp")
            return None
        except Exception as e:
            print(f"[ERROR] No se pudo obtener URL de YouTube: {e}")
            return None
    
    @staticmethod
    def is_rtmp_url(url: str) -> bool:
        return url.startswith('rtmp://') or url.startswith('rtmps://')
    
    @staticmethod
    def is_youtube_url(url: str) -> bool:
        return 'youtube.com' in url or 'youtu.be' in url
