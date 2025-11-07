Write-Host "===== Football Tracker Pipeline - Instalador =====" -ForegroundColor Green
Write-Host ""

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python no está instalado" -ForegroundColor Red
    exit 1
}

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    Write-Host "WARNING: FFmpeg no encontrado" -ForegroundColor Yellow
    Write-Host "Descarga FFmpeg de: https://ffmpeg.org/download.html"
    Write-Host "O instala con: winget install ffmpeg"
}

Write-Host "Instalando dependencias de Python..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "Verificando CUDA..." -ForegroundColor Cyan
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host ""
Write-Host "Creando directorios..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "data\inputs" | Out-Null
New-Item -ItemType Directory -Force -Path "data\outputs\batch_renders" | Out-Null
New-Item -ItemType Directory -Force -Path "data\outputs\tracker_paths" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null

Write-Host ""
Write-Host "===== Instalación completada =====" -ForegroundColor Green
Write-Host ""
Write-Host "Siguiente paso:"
Write-Host "1. Coloca tu modelo YOLOv8 en: models\yolov8_nano.pt"
Write-Host "2. Coloca videos de prueba en: data\inputs\"
Write-Host "3. Ejecuta: python main.py stream --debug --input URL_DE_YOUTUBE"
