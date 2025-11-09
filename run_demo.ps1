$ErrorActionPreference = "Stop"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Football Detection - Run Demo     " -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path ".venv")) {
    Write-Host "ERROR: Virtual environment not found" -ForegroundColor Red
    Write-Host "Run setup_demo.ps1 first" -ForegroundColor Yellow
    exit 1
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Checking configuration..." -ForegroundColor Cyan
if (-not (Test-Path "configs\stream_config.yml")) {
    Write-Host "ERROR: configs\stream_config.yml not found" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path "configs\model_config.yml")) {
    Write-Host "ERROR: configs\model_config.yml not found" -ForegroundColor Red
    exit 1
}

$streamConfig = Get-Content "configs\stream_config.yml" -Raw
if ($streamConfig -match 'input_url:\s*[''"]?([^''"\n]+)') {
    $inputUrl = $matches[1].Trim()
    Write-Host "Input: $inputUrl" -ForegroundColor Green
} else {
    Write-Host "WARNING: input_url not configured in stream_config.yml" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "Starting MJPEG Stream Server..." -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Stream will be available at:" -ForegroundColor Cyan
Write-Host "http://localhost:8554/stream.mjpg" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

python run_mjpeg_stream.py
