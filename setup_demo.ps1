$ErrorActionPreference = "Stop"

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "  Football Detection - Setup Demo   " -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    Write-Host "Install Python 3.9-3.11 from https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

$pythonVersion = & python --version 2>&1
Write-Host "Python: $pythonVersion" -ForegroundColor Green

if (Test-Path ".venv") {
    Write-Host "Virtual environment exists" -ForegroundColor Yellow
    $response = Read-Host "Recreate? (y/n)"
    if ($response -eq 'y') {
        Remove-Item -Recurse -Force .venv
        Write-Host "Removed existing venv" -ForegroundColor Yellow
    }
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
    Write-Host "Virtual environment created" -ForegroundColor Green
}

Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

Write-Host ""
Write-Host "Detecting CUDA..." -ForegroundColor Cyan
$cudaAvailable = $false
try {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        $cudaVersion = & nvidia-smi 2>$null | Select-String "CUDA Version: (\d+\.\d+)" | ForEach-Object { $_.Matches.Groups[1].Value }
        if ($cudaVersion) {
            Write-Host "NVIDIA GPU detected (CUDA $cudaVersion)" -ForegroundColor Green
            $cudaAvailable = $true
        }
    }
} catch {}

if (-not $cudaAvailable) {
    Write-Host "No CUDA detected - will use CPU" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Cyan

if ($cudaAvailable) {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
} else {
    pip install torch torchvision --quiet
}

Write-Host "Installing remaining packages..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
$torchCheck = python -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU'); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
$device = $torchCheck[0]
$gpuName = $torchCheck[1]
Write-Host "Device: $device" -ForegroundColor Green
if ($device -eq "CUDA") {
    Write-Host "GPU: $gpuName" -ForegroundColor Green
}

Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path "data\inputs" | Out-Null
New-Item -ItemType Directory -Force -Path "data\outputs" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null

Write-Host ""
Write-Host "=====================================" -ForegroundColor Green
Write-Host "     Setup Complete                  " -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Edit configs\stream_config.yml:" -ForegroundColor White
Write-Host "   - Set stream.input_url to your video path" -ForegroundColor White
Write-Host "2. Edit configs\model_config.yml:" -ForegroundColor White
Write-Host "   - Set model.device to 'cuda' or 'cpu'" -ForegroundColor White
Write-Host "   - Set model.path if using custom weights" -ForegroundColor White
Write-Host "3. Run: .\run_demo.ps1" -ForegroundColor White
Write-Host ""
