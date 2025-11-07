# ⚽ Football Tracker Pipeline

Sistema de seguimiento inteligente de balón en tiempo real usando YOLOv8 + ByteTrack + Cámara Virtual.

## 🎯 Características

- **Detección IA**: YOLOv8 optimizado para GPU (CUDA/TensorRT)
- **Tracking robusto**: ByteTrack + Filtro de Kalman para manejar oclusiones
- **Cámara virtual inteligente**: One-Euro Filter + dead-zones + anticipación
- **Tiempo real**: <33ms por frame (30+ FPS en GPU moderna)
- **Dual mode**: Procesamiento por lotes (batch) y streaming en vivo
- **YouTube support**: Prueba con videos en vivo de YouTube
- **RTMP ready**: Push a servidores de streaming (Nginx-RTMP, Wowza, etc.)

## 📁 Estructura

```
football_tracker_pipeline/
├── app/
│   ├── camera/          # Cámara virtual (One-Euro Filter, dead-zones)
│   ├── inference/       # Detector YOLOv8
│   ├── tracking/        # Tracker (ByteTrack + Kalman)
│   ├── pipelines/       # Batch & Stream pipelines
│   └── utils/           # Video I/O, RTMP, configs
├── configs/             # YAML configurations
├── data/                # Videos de entrada/salida (gitignored)
├── models/              # Modelos .pt (gitignored)
├── scripts/             # Scripts de instalación y arranque
└── main.py              # CLI principal
```

## 🚀 Instalación

### Requisitos previos
- Python 3.8+
- CUDA 11.8+ (para GPU)
- FFmpeg

### Instalación rápida

```bash
# Linux/Mac
bash scripts/install_dependencies.sh

# Windows
pip install -r requirements.txt
```

### Verificar GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🎮 Uso

### Modo Stream (Tiempo Real)

#### Prueba con YouTube (Debug Mode)

```bash
python main.py stream --debug --input "https://www.youtube.com/watch?v=VIDEO_ID"
```

Este modo:
- Extrae el stream directo de YouTube
- Muestra ventana de preview en tiempo real
- No requiere servidor RTMP
- Perfecto para desarrollo

#### Stream a RTMP (Producción)

```bash
python main.py stream \
  --input "rtmp://source.com/live/input" \
  --output "rtmp://destination.com/live/output"
```

O usa el script helper:

```bash
bash scripts/start_stream_worker.sh rtmp://input rtmp://output
```

### Modo Batch (Archivos)

```bash
python main.py batch --input data/inputs/match.mp4 --output data/outputs/tracked.mp4
```

Genera:
- Video procesado con seguimiento
- JSON con trayectorias del balón (si está habilitado)

## ⚙️ Configuración

### configs/model_config.yml

```yaml
model:
  path: "models/yolov8_nano.pt"
  confidence: 0.25
  device: "cuda"
  half_precision: true
```

### configs/stream_config.yml

```yaml
stream:
  target_fps: 30
  bitrate: "4000k"
  preset: "ultrafast"
  debug_mode: false

camera:
  dead_zone: 0.10        # 10% no-move zone
  anticipation: 0.3      # 30% anticipation
  zoom_padding: 1.2      # 20% zoom out
```

## 🎨 Personalización

### Ajustar suavizado de cámara

```yaml
camera:
  smoothing_min_cutoff: 1.0   # ↓ más suave, ↑ más reactivo
  smoothing_beta: 0.007        # ↓ menos anticipación
```

### Ajustar tracking

```yaml
tracking:
  max_lost_frames: 10          # Frames antes de perder track
  min_confidence: 0.3          # Confianza mínima
```

## 📊 Optimización de Rendimiento

### Para máxima velocidad:

1. **Exportar a TensorRT**:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True)
```

2. **Usar modelo más pequeño**: `yolov8n.pt` en vez de `yolov8x.pt`

3. **Reducir resolución de entrada**: Detectar en 640x640, renderizar en 1920x1080

4. **Half precision**: `half_precision: true` en config

### Benchmarks esperados (RTX 3080):

- YOLOv8 Nano: ~5-8ms por frame
- Tracking: ~1ms
- Cámara virtual: <1ms
- **Total: ~10-15ms → 60+ FPS posible**

## 🧪 Testing

### Notebook 1: Benchmark de modelo

```bash
jupyter notebook notebooks/01_benchmark_model_speed.ipynb
```

### Notebook 2: Tuning de cámara

```bash
jupyter notebook notebooks/02_tune_camera_smoothing.ipynb
```

### Notebook 3: Test RTMP

```bash
jupyter notebook notebooks/03_test_rtmp_connection.ipynb
```

## 🐛 Troubleshooting

### "No se pudo conectar a YouTube"

```bash
pip install --upgrade yt-dlp
```

### "CUDA out of memory"

Reducir batch size o usar modelo más pequeño:
```yaml
model:
  path: "models/yolov8n.pt"  # nano en vez de large
```

### "FFmpeg no encontrado"

```bash
# Ubuntu
sudo apt install ffmpeg

# Mac
brew install ffmpeg

# Windows
# Descargar de: https://ffmpeg.org/download.html
```

## 📈 Roadmap

- [ ] Multi-objeto tracking (jugadores + balón)
- [ ] Auto-zoom adaptativo basado en velocidad
- [ ] WebRTC support (ultra-baja latencia)
- [ ] Dashboard de métricas en vivo
- [ ] Docker container para deploy fácil

## 📝 Licencia

MIT

## 👤 Autor

Desarrollado para el proyecto de Ziyad - Sistema de transmisión deportiva inteligente.
