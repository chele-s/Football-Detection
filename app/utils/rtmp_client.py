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
            
            print("[INFO] YouTube bloqueado - SOLUCIÓN: Usa un video descargado o stream HLS")
            print("[INFO] Para descargar el video, ejecuta en Colab:")
            print(f"       !yt-dlp -f 'best[height<=720]' -o football.mp4 '{youtube_url}'")
            print("       Luego usa: /content/football.mp4 como Video URL")
            
            methods = [
                {
                    'name': 'Android TV',
                    'opts': {
                        'format': 'best[height<=720]',
                        'quiet': False,
                        'no_warnings': False,
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['android_embedded', 'android', 'tv_embedded'],
                            }
                        }
                    }
                },
                {
                    'name': 'iOS',
                    'opts': {
                        'format': 'best[height<=720]',
                        'quiet': False,
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['ios'],
                            }
                        }
                    }
                },
                {
                    'name': 'Web embed',
                    'opts': {
                        'format': 'best[height<=720]',
                        'quiet': False,
                        'extractor_args': {
                            'youtube': {
                                'player_client': ['web_embedded'],
                            }
                        }
                    }
                }
            ]
            
            for method in methods:
                try:
                    print(f"[INFO] Intentando método: {method['name']}...")
                    with yt_dlp.YoutubeDL(method['opts']) as ydl:
                        info = ydl.extract_info(youtube_url, download=False)
                        if 'url' in info:
                            print(f"[SUCCESS] URL extraída con método {method['name']}")
                            return info['url']
                        elif 'entries' in info and len(info['entries']) > 0:
                            return info['entries'][0]['url']
                except Exception as e:
                    print(f"[FAIL] Método {method['name']} falló: {str(e)[:100]}")
                    continue
            
            print("[ERROR] Todos los métodos fallaron")
            return None
        
        except ImportError:
            print("[ERROR] yt-dlp no instalado. Ejecuta: pip install yt-dlp")
            return None
        except Exception as e:
            print(f"[ERROR] Error crítico: {e}")
            return None
    
    @staticmethod
    def is_rtmp_url(url: str) -> bool:
        return url.startswith('rtmp://') or url.startswith('rtmps://')
    
    @staticmethod
    def is_youtube_url(url: str) -> bool:
        return 'youtube.com' in url or 'youtu.be' in url
