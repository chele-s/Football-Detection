# ============================================================================
# RF-DETR Football Detection - Windows Production Installation Script
# ============================================================================
# Run in PowerShell as Administrator:
#   Set-ExecutionPolicy Bypass -Scope Process -Force
#   .\setup_production_windows.ps1
# ============================================================================

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  RF-DETR Football Detection - Windows Production Setup" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1. Check Administrator Rights
# ============================================================================
Write-Info "Verifying administrator rights..."
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator"
    Write-Host "Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}
Write-Success "Administrator rights confirmed"

# ============================================================================
# 2. Check NVIDIA GPU
# ============================================================================
Write-Info "Checking for NVIDIA GPU..."
try {
    $nvidiaGPU = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($nvidiaGPU) {
        Write-Success "NVIDIA GPU detected: $($nvidiaGPU.Name)"
    } else {
        Write-Warning "No NVIDIA GPU detected. Pipeline will use CPU mode."
    }
} catch {
    Write-Warning "Could not detect GPU: $_"
}

# Check nvidia-smi
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>$null
    if ($nvidiaSmi) {
        $driverInfo = $nvidiaSmi.Split(',')
        Write-Success "NVIDIA Driver: $($driverInfo[0].Trim())"
        Write-Success "GPU Model: $($driverInfo[1].Trim())"
    }
} catch {
    Write-Warning "nvidia-smi not found. Install NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx"
}

# ============================================================================
# 3. Install Chocolatey Package Manager
# ============================================================================
Write-Info "Checking Chocolatey package manager..."
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Chocolatey..."
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    # Refresh environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Success "Chocolatey installed"
} else {
    Write-Success "Chocolatey already installed"
}

# ============================================================================
# 4. Install System Dependencies
# ============================================================================
Write-Info "Installing system dependencies..."

# Git
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Git..."
    choco install git -y --no-progress
} else {
    Write-Success "Git already installed"
}

# Python 3.10
Write-Info "Checking Python 3.10..."
$pythonInstalled = $false
$pythonCmd = $null

# Try to find Python 3.10
$pythonPaths = @(
    "C:\Python310\python.exe",
    "C:\Program Files\Python310\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)

foreach ($path in $pythonPaths) {
    if (Test-Path $path) {
        $version = & $path --version 2>&1
        if ($version -match "3\.10") {
            $pythonCmd = $path
            $pythonInstalled = $true
            Write-Success "Python 3.10 found at: $path"
            break
        }
    }
}

if (-not $pythonInstalled) {
    Write-Info "Installing Python 3.10..."
    choco install python310 -y --no-progress --params "/InstallDir:C:\Python310"
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    $pythonCmd = "C:\Python310\python.exe"
    if (-not (Test-Path $pythonCmd)) {
        $pythonCmd = "python"
    }
    
    Write-Success "Python 3.10 installed"
}

# Verify Python
$pyVersion = & $pythonCmd --version 2>&1
Write-Info "Python version: $pyVersion"

# FFmpeg
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Info "Installing FFmpeg..."
    choco install ffmpeg -y --no-progress
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    Write-Success "FFmpeg installed"
} else {
    Write-Success "FFmpeg already installed"
}

# Visual Studio Build Tools (needed for compiling Python packages)
Write-Info "Checking Visual Studio Build Tools..."
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $buildTools = & $vsWhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($buildTools) {
        Write-Success "Visual Studio Build Tools found"
    } else {
        Write-Warning "Visual Studio Build Tools not found"
        Write-Info "Installing Visual Studio Build Tools (this may take a while)..."
        choco install visualstudio2022buildtools -y --no-progress --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"
    }
} else {
    Write-Warning "Visual Studio Build Tools not found"
    Write-Info "Installing Visual Studio Build Tools (this may take a while)..."
    choco install visualstudio2022buildtools -y --no-progress --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"
}

# ============================================================================
# 5. Create Virtual Environment
# ============================================================================
Write-Info "Creating Python virtual environment..."

$venvPath = "rf-detr-venv-310"
if (Test-Path $venvPath) {
    Write-Warning "Virtual environment already exists at $venvPath"
    $response = Read-Host "Remove and recreate? (y/n)"
    if ($response -eq 'y') {
        Remove-Item -Recurse -Force $venvPath
        Write-Info "Removed old environment"
    } else {
        Write-Info "Using existing environment"
    }
}

if (-not (Test-Path $venvPath)) {
    & $pythonCmd -m venv $venvPath
    Write-Success "Virtual environment created"
}

# Activate virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
    Write-Success "Virtual environment activated"
} else {
    Write-Error "Could not find activation script at $activateScript"
    exit 1
}

# Verify we're in venv
$pythonInVenv = & python --version 2>&1
Write-Info "Using Python: $pythonInVenv"

# ============================================================================
# 6. Install Python Dependencies
# ============================================================================
Write-Info "Installing Python dependencies..."

python -m pip install --upgrade pip setuptools wheel

if (Test-Path "requirements.txt") {
    Write-Info "Installing from requirements.txt..."
    pip install -r requirements.txt
    
    Write-Success "Python dependencies installed"
} else {
    Write-Error "requirements.txt not found!"
    exit 1
}

# ============================================================================
# 7. Check CUDA Installation
# ============================================================================
Write-Info "Checking CUDA installation..."

$cudaPath = $env:CUDA_PATH
if ($cudaPath) {
    Write-Success "CUDA found at: $cudaPath"
    
    # Check nvcc
    $nvccPath = Join-Path $cudaPath "bin\nvcc.exe"
    if (Test-Path $nvccPath) {
        $cudaVersion = & $nvccPath --version | Select-String "release"
        Write-Success "CUDA Version: $cudaVersion"
    }
} else {
    Write-Warning "CUDA not found in environment variables"
    Write-Info "Download and install CUDA Toolkit 11.8+ from:"
    Write-Info "https://developer.nvidia.com/cuda-downloads"
    Write-Warning "Pipeline will attempt to use PyTorch's bundled CUDA"
}

# ============================================================================
# 8. Verify PyTorch CUDA Support
# ============================================================================
Write-Info "Verifying PyTorch CUDA support..."

$torchCheck = @"
import torch
import sys

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    
    # Test GPU tensor
    try:
        x = torch.randn(100, 100).cuda()
        print('GPU tensor creation: OK')
    except Exception as e:
        print(f'GPU tensor creation: FAILED - {e}')
        sys.exit(1)
else:
    print('WARNING: CUDA not available in PyTorch')
    print('Install CUDA-enabled PyTorch:')
    print('pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118')
"@

python -c $torchCheck

if ($LASTEXITCODE -ne 0) {
    Write-Warning "PyTorch CUDA check failed"
    $response = Read-Host "Reinstall PyTorch with CUDA support? (y/n)"
    if ($response -eq 'y') {
        Write-Info "Installing PyTorch with CUDA 11.8..."
        pip uninstall torch torchvision -y
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        Write-Success "PyTorch reinstalled"
    }
}

# ============================================================================
# 9. PyNvCodec Check (Windows typically uses CPU fallback)
# ============================================================================
Write-Info "Checking PyNvCodec availability..."

$pynvcodecCheck = @"
try:
    import PyNvCodec as nvc
    print('PyNvCodec: Available')
except ImportError:
    print('PyNvCodec: Not available (will use CPU video processing)')
    print('Note: PyNvCodec is difficult to build on Windows.')
    print('      GPU acceleration will use PyTorch CUDA for inference only.')
"@

python -c $pynvcodecCheck

Write-Warning "PyNvCodec is typically not available on Windows"
Write-Info "Pipeline will use CPU for video I/O and GPU for inference"

# ============================================================================
# 10. Create Directory Structure
# ============================================================================
Write-Info "Creating project directories..."

$directories = @("models", "data\inputs", "data\outputs", "clips", "logs")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

Write-Success "Directories created"

# ============================================================================
# 11. Create Windows Service Scripts
# ============================================================================
Write-Info "Creating service management scripts..."

# Start script (PowerShell)
$startScript = @'
# Start RF-DETR streaming pipeline
$ErrorActionPreference = "Stop"

Write-Host "Starting RF-DETR Football Detection Stream..." -ForegroundColor Cyan

# Activate virtual environment
$venvPath = "rf-detr-venv-310"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (-not (Test-Path $activateScript)) {
    Write-Host "ERROR: Virtual environment not found at $venvPath" -ForegroundColor Red
    exit 1
}

. $activateScript

# Start pipeline
$logFile = "logs\stream_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Write-Host "Log file: $logFile" -ForegroundColor Yellow
Write-Host "Stream URL: http://localhost:8554/stream.mjpg" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the stream" -ForegroundColor Yellow
Write-Host ""

python run_mjpeg_stream.py 2>&1 | Tee-Object -FilePath $logFile
'@

Set-Content -Path "start_stream.ps1" -Value $startScript
Write-Success "Created start_stream.ps1"

# Start script (Batch for convenience)
$startBatch = @'
@echo off
echo Starting RF-DETR Stream...
powershell -ExecutionPolicy Bypass -File start_stream.ps1
pause
'@

Set-Content -Path "start_stream.bat" -Value $startBatch
Write-Success "Created start_stream.bat"

# Stop script
$stopScript = @'
# Stop RF-DETR streaming pipeline
Write-Host "Stopping RF-DETR Stream..." -ForegroundColor Yellow

$process = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*mjpeg*" }
if ($process) {
    Stop-Process -Id $process.Id -Force
    Write-Host "Stream stopped" -ForegroundColor Green
} else {
    Write-Host "No active stream found" -ForegroundColor Yellow
}
'@

Set-Content -Path "stop_stream.ps1" -Value $stopScript
Write-Success "Created stop_stream.ps1"

# ============================================================================
# 12. Create Windows Diagnostics Script
# ============================================================================
Write-Info "Creating diagnostics script for Windows..."

$diagScript = @'
# RF-DETR Windows GPU Diagnostics
$ErrorActionPreference = "Continue"

function Write-Section {
    param($title)
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
}

Write-Section "RF-DETR Football Detection - Windows Diagnostics"

# GPU Check
Write-Section "NVIDIA GPU"
try {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpu) {
        Write-Host "[OK] GPU: $($gpu.Name)" -ForegroundColor Green
        Write-Host "[OK] Driver: $($gpu.DriverVersion)" -ForegroundColor Green
    } else {
        Write-Host "[WARN] No NVIDIA GPU detected" -ForegroundColor Yellow
    }
    
    $nvidiaSmi = & nvidia-smi --query-gpu=driver_version,memory.total,compute_cap --format=csv,noheader 2>$null
    if ($nvidiaSmi) {
        $info = $nvidiaSmi.Split(',')
        Write-Host "[OK] Driver: $($info[0].Trim())" -ForegroundColor Green
        Write-Host "[OK] VRAM: $($info[1].Trim())" -ForegroundColor Green
        Write-Host "[OK] Compute Capability: $($info[2].Trim())" -ForegroundColor Green
    }
} catch {
    Write-Host "[ERROR] Could not query GPU: $_" -ForegroundColor Red
}

# Python Check
Write-Section "Python Environment"
$pyVersion = python --version 2>&1
Write-Host "[OK] Python: $pyVersion" -ForegroundColor Green
Write-Host "[OK] Python Path: $(Get-Command python | Select-Object -ExpandProperty Source)" -ForegroundColor Green

# PyTorch Check
Write-Section "PyTorch & CUDA"
python -c @"
import torch
print(f'[OK] PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'[OK] CUDA Available: True')
    print(f'[OK] CUDA Version: {torch.version.cuda}')
    print(f'[OK] GPU: {torch.cuda.get_device_name(0)}')
else:
    print('[WARN] CUDA not available in PyTorch')
"@

# Dependencies Check
Write-Section "Dependencies"
$packages = @("torch", "torchvision", "opencv-python", "rfdetr", "supervision", "numpy", "scipy", "yt-dlp")
foreach ($pkg in $packages) {
    $check = pip show $pkg 2>$null
    if ($check) {
        $version = ($check | Select-String "Version:").ToString().Split(":")[1].Trim()
        Write-Host "[OK] $pkg`: $version" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $pkg" -ForegroundColor Red
    }
}

# PyNvCodec Check
Write-Section "PyNvCodec (GPU Video Acceleration)"
python -c @"
try:
    import PyNvCodec
    print('[OK] PyNvCodec: Available')
except ImportError:
    print('[INFO] PyNvCodec: Not available (expected on Windows)')
    print('[INFO] Will use CPU for video I/O')
"@

# FFmpeg Check
Write-Section "FFmpeg"
try {
    $ffmpegVersion = & ffmpeg -version 2>&1 | Select-Object -First 1
    Write-Host "[OK] $ffmpegVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] FFmpeg not found" -ForegroundColor Red
}

# Model Check
Write-Section "Model Weights"
if (Test-Path "models\best_rf-detr.pth") {
    $size = (Get-Item "models\best_rf-detr.pth").Length / 1MB
    Write-Host "[OK] Model found: models\best_rf-detr.pth ($([math]::Round($size, 1)) MB)" -ForegroundColor Green
} else {
    Write-Host "[MISSING] Model not found: models\best_rf-detr.pth" -ForegroundColor Red
    Write-Host "         Download your trained model and place it there" -ForegroundColor Yellow
}

# Config Check
Write-Section "Configuration Files"
$configs = @("configs\model_config.yml", "configs\stream_config.yml")
foreach ($cfg in $configs) {
    if (Test-Path $cfg) {
        Write-Host "[OK] $cfg" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $cfg" -ForegroundColor Red
    }
}

Write-Section "Summary"
Write-Host ""
Write-Host "System ready for deployment!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Configure: Edit configs\stream_config.yml" -ForegroundColor White
Write-Host "  2. Start: .\start_stream.bat or .\start_stream.ps1" -ForegroundColor White
Write-Host "  3. View: http://localhost:8554/stream.mjpg in VLC" -ForegroundColor White
Write-Host ""
'@

Set-Content -Path "diagnose_gpu_windows.ps1" -Value $diagScript
Write-Success "Created diagnose_gpu_windows.ps1"

# ============================================================================
# 13. Run Diagnostics
# ============================================================================
Write-Info "Running post-installation diagnostics..."
Write-Host ""

& .\diagnose_gpu_windows.ps1

# ============================================================================
# 14. Final Instructions
# ============================================================================
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host "  Installation Complete!" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Place your model weights:" -ForegroundColor White
Write-Host "   models\best_rf-detr.pth" -ForegroundColor Yellow
Write-Host ""
Write-Host "2. Configure your stream:" -ForegroundColor White
Write-Host "   Edit configs\stream_config.yml" -ForegroundColor Yellow
Write-Host ""
Write-Host "3. Start the pipeline:" -ForegroundColor White
Write-Host "   Double-click: start_stream.bat" -ForegroundColor Yellow
Write-Host "   Or run: .\start_stream.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "4. View the stream:" -ForegroundColor White
Write-Host "   Open in VLC: http://localhost:8554/stream.mjpg" -ForegroundColor Yellow
Write-Host ""
Write-Host "5. Record (in another PowerShell window):" -ForegroundColor White
Write-Host "   ffmpeg -i http://localhost:8554/stream.mjpg -c copy output.mp4" -ForegroundColor Yellow
Write-Host ""
Write-Host "Troubleshooting:" -ForegroundColor Cyan
Write-Host "   Run: .\diagnose_gpu_windows.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "============================================================================" -ForegroundColor Green
Write-Host ""
