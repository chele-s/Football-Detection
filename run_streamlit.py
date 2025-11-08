import subprocess
import sys
import time
import requests
import json

def check_ngrok():
    try:
        return subprocess.run(['ngrok', 'version'], 
                            capture_output=True, 
                            timeout=2).returncode == 0
    except:
        return False

def get_ngrok_url():
    try:
        response = requests.get('http://localhost:4040/api/tunnels', timeout=2)
        tunnels = response.json()['tunnels']
        if tunnels:
            return tunnels[0]['public_url']
    except:
        pass
    return None

def start_ngrok():
    print("Starting ngrok tunnel...")
    try:
        ngrok_process = subprocess.Popen(
            ['ngrok', 'http', '8501'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(3)
        
        url = get_ngrok_url()
        if url:
            print(f"✓ ngrok tunnel active: {url}")
            return ngrok_process
        else:
            print("⚠ ngrok tunnel not available")
            return None
    except Exception as e:
        print(f"⚠ ngrok error: {e}")
        return None

def main():
    print("="*60)
    print("Football Detection - Streamlit Visualization")
    print("="*60)
    print()
    
    ngrok_process = None
    
    if check_ngrok():
        ngrok_process = start_ngrok()
    else:
        print("⚠ ngrok not found - running on localhost only")
    
    print()
    print("="*60)
    print("Starting Streamlit on http://localhost:8501")
    print("Press Ctrl+C to stop")
    print("="*60)
    print()
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            'app_streamlit.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if ngrok_process:
            ngrok_process.terminate()

if __name__ == "__main__":
    main()
