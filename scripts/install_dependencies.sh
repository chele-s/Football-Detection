#!/bin/bash

echo "===== Football Tracker Pipeline - Instalador ====="
echo ""

if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 no está instalado"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: FFmpeg no está instalado"
    echo "Instala FFmpeg: sudo apt install ffmpeg (Ubuntu) o brew install ffmpeg (Mac)"
    exit 1
fi

echo "Instalando dependencias de Python..."
pip install -r requirements.txt

echo ""
echo "Verificando CUDA..."
python3 -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Creando directorios..."
mkdir -p data/inputs
mkdir -p data/outputs/batch_renders
mkdir -p data/outputs/tracker_paths
mkdir -p models

echo ""
echo "===== Instalación completada ====="
echo ""
echo "Siguiente paso:"
echo "1. Coloca tu modelo YOLOv8 en: models/yolov8_nano.pt"
echo "2. Coloca videos de prueba en: data/inputs/"
echo "3. Ejecuta: python main.py stream --debug --input URL_DE_YOUTUBE"
