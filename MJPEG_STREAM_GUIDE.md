# 🎥 Guía MJPEG Stream - Football Detection

## 📋 Descripción

Esta guía te muestra cómo iniciar el stream de detección de fútbol con MJPEG server, sin usar Streamlit.

---

## 🚀 Uso en Google Colab

### **Paso 1: Preparación**

En una celda de Colab, clona el repo e instala dependencias:

```python
!git clone https://github.com/TU_USUARIO/Football-Detection.git
%cd Football-Detection
!pip install -q -r requirements.txt
```

---

### **Paso 2: Iniciar el Stream (Celda 1)**

En una celda, ejecuta:

```python
!python run_mjpeg_stream.py
```

Deberías ver:
```
🎥 MJPEG Stream Server - Football Detection
============================================================
[1/5] Loading configurations...
[2/5] Loading RF-DETR model...
[3/5] Initializing ball tracker...
[4/5] Starting MJPEG server on port 8554...
✅ MJPEG server started!
📺 Stream URL: http://localhost:8554/stream.mjpg
[5/5] Opening video: /content/football.mp4
============================================================
🚀 Starting processing loop...
[STREAM] Frame   30 | FPS:  18.5 | Inf:  26.3ms | Loop:  28.1ms
```

**IMPORTANTE:** Esta celda quedará ejecutándose continuamente. **NO LA DETENGAS.**

---

### **Paso 3: Exponer el Stream con ngrok (Celda 2)**

En **OTRA CELDA** (mientras la primera sigue corriendo), ejecuta:

```python
from pyngrok import ngrok

# Configura tu authtoken (obtenerlo de https://dashboard.ngrok.com/get-started/your-authtoken)
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")

# Crear túnel
tunnel = ngrok.connect(8554, "http")
print(f'✅ Túnel creado!')
print(f'🎥 Video Stream URL: {tunnel.public_url}/stream.mjpg')
```

**O usa el script incluido:**

```python
!python setup_ngrok_tunnel.py
```

---

### **Paso 4: Ver el Stream**

Copia la URL que apareció (algo como `https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app/stream.mjpg`) y ábrela en:

#### **Opción 1: VLC Media Player**
1. Abre VLC
2. Media → Open Network Stream
3. Pega la URL
4. Play

#### **Opción 2: Navegador**
1. Abre Chrome o Firefox
2. Pega la URL
3. El video debería aparecer automáticamente

#### **Opción 3: ffplay (Linux/Mac)**
```bash
ffplay "https://xxxx.ngrok-free.app/stream.mjpg"
```

---

## 📊 Output Esperado

```
[STREAM] Frame   30 | FPS:  18.5 | Inf:  26.3ms | Loop:  28.1ms
[STREAM] Frame   60 | FPS:  19.2 | Inf:  25.1ms | Loop:  27.3ms
[STREAM] Frame   90 | FPS:  18.8 | Inf:  26.7ms | Loop:  28.9ms
```

**Métricas:**
- **FPS**: Frames por segundo de procesamiento real
- **Inf**: Tiempo de inferencia del modelo RF-DETR
- **Loop**: Tiempo total del loop (inferencia + tracking + rendering)

---

## 🛑 Detener el Stream

1. En la celda donde está corriendo `run_mjpeg_stream.py`, presiona el botón **STOP** ⏹️
2. O presiona **Ctrl+C** en la terminal

---

## 🔧 Configuración

### Cambiar video de entrada

Edita `run_mjpeg_stream.py`, línea 68:

```python
video_path = '/content/football.mp4'  # Cambia esto
```

### Cambiar puerto del MJPEG server

Edita `run_mjpeg_stream.py`, línea 48:

```python
mjpeg_server = MJPEGServer(port=8554)  # Cambia el puerto aquí
```

Y luego en `setup_ngrok_tunnel.py` o en tu código de ngrok, usa el mismo puerto.

---

## ⚠️ Troubleshooting

### "Connection refused" en ngrok
- **Causa**: El MJPEG server no está corriendo
- **Solución**: Asegúrate de que `run_mjpeg_stream.py` esté ejecutándose primero

### "Video ended, restarting..."
- **Causa**: El video llegó al final
- **Comportamiento**: El script reinicia el video automáticamente (loop infinito)

### FPS muy bajo (<5 FPS)
- **Causa**: GPU no está siendo utilizada
- **Solución**: Verifica que `device: cuda` en `configs/model_config.yml`

### Túnel ngrok se cierra después de poco tiempo
- **Causa**: No configuraste el authtoken
- **Solución**: Obtén tu token de https://dashboard.ngrok.com y agrégalo al script

---

## 📦 Archivos Incluidos

- **`run_mjpeg_stream.py`**: Script principal que inicia el procesamiento y el server
- **`setup_ngrok_tunnel.py`**: Script auxiliar para crear el túnel ngrok
- **`app/utils/mjpeg_server.py`**: Implementación del MJPEG HTTP server

---

## 🎯 Performance Esperado

Con RF-DETR Medium + tracking + virtual camera en Google Colab (Tesla T4):

- **FPS**: 15-20 FPS
- **Inference time**: 25-35ms
- **Loop time**: 28-40ms
- **Latency del stream**: <100ms (depende de ngrok)

---

## 📝 Notas

- El MJPEG stream es compatible con la mayoría de navegadores y reproductores
- La calidad del stream es alta (JPEG quality=85)
- No hay límite de viewers simultáneos (el server es multi-threaded)
- El túnel ngrok free tiene límite de 40 conexiones/minuto

---

**¿Problemas? Abre un issue en GitHub.**
