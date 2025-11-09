#!/usr/bin/env python3
"""
Script para exponer el MJPEG server via ngrok en Colab.
Ejecuta esto en UNA CELDA SEPARADA mientras run_mjpeg_stream.py está corriendo en otra.
"""
from pyngrok import ngrok
import time

def setup_tunnel(port=8554, authtoken=None):
    print("🔧 Configurando túnel ngrok...")
    
    if authtoken:
        print("🔑 Configurando authtoken...")
        ngrok.set_auth_token(authtoken)
    else:
        print("⚠️  No se proporcionó authtoken. El túnel puede ser limitado.")
    
    try:
        # Crear túnel
        print(f"📡 Creando túnel para puerto {port}...")
        tunnel = ngrok.connect(port, "http")
        
        print("\n" + "="*60)
        print("✅ ¡Túnel creado exitosamente!")
        print("="*60)
        print(f"🎥 Video Stream URL: {tunnel.public_url}/stream.mjpg")
        print("\n📺 Abre esta URL en:")
        print("  - VLC: Media → Open Network Stream")
        print("  - Navegador: Chrome, Firefox, etc.")
        print("  - ffplay: ffplay '<URL>'")
        print("="*60)
        
        # Mantener el túnel activo
        print("\n💡 El túnel permanecerá activo mientras esta celda esté ejecutándose.")
        print("Press Ctrl+C para cerrar el túnel.\n")
        
        try:
            while True:
                time.sleep(10)
        except KeyboardInterrupt:
            print("\n🛑 Cerrando túnel...")
            ngrok.disconnect(tunnel.public_url)
            print("✅ Túnel cerrado")
    
    except Exception as e:
        print(f"❌ Error al crear el túnel: {e}")
        print("\nPosibles soluciones:")
        print("1. Verifica que el puerto 8554 esté activo (run_mjpeg_stream.py corriendo)")
        print("2. Asegúrate de que tu authtoken sea correcto")
        print("3. Cierra otros túneles activos de ngrok")

if __name__ == "__main__":
    YOUR_AUTHTOKEN = None 
    
    if YOUR_AUTHTOKEN:
        setup_tunnel(authtoken=YOUR_AUTHTOKEN)
    else:
        print("⚠️  IMPORTANTE: Configura tu authtoken para un túnel estable")
        print("1. Ve a: https://dashboard.ngrok.com/get-started/your-authtoken")
        print("2. Copia tu authtoken")
        print("3. Reemplaza YOUR_AUTHTOKEN en este script\n")
        setup_tunnel()
