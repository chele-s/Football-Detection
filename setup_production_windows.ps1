    # ============================================================================
# RF-DETR Football Detection - Windows Production Setup (Robust)
# ============================================================================
# Optimized for RTX 30/40/50 Series GPUs
# Run as Administrator
# ============================================================================

#Requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# --- Helper Functions ---
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[SUCCESS] $args" -ForegroundColor Green }
function Write-Warning { Write-Host "[WARNING] $args" -ForegroundColor Yellow }
function Write-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "  RF-DETR Football Detection - Windows Setup (Robust)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1. System Checks
# ============================================================================
Write-Info "Checking system requirements..."

# Check for NVIDIA GPU
try {
    $gpu = Get-WmiObject Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($gpu) {
        Write-Success "NVIDIA GPU detected: $($gpu.Name)"
    } else {
        Write-Warning "No NVIDIA GPU detected. Performance will be limited."
    }
} catch {
    Write-Warning "Could not query GPU information."
}

# ============================================================================
# 2. Install Core Tools (Chocolatey)
# ============================================================================
Write-Info "Checking package manager..."

if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Info "Installing Chocolatey..."
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    # Refresh env vars for this session
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
} else {
    Write-Success "Chocolatey is already installed."
}

# Install Git and Python if missing
$chocoPackages = @("git", "python310", "ffmpeg")
foreach ($pkg in $chocoPackages) {
    if (-not (Get-Command $pkg -ErrorAction SilentlyContinue)) {
        if ($pkg -eq "python310" -and (Get-Command python -ErrorAction SilentlyContinue)) {
            # Check if existing python is 3.10
            $ver = python --version 2>&1
            if ($ver -match "3.10") { continue }
        }
        Write-Info "Installing $pkg..."
        choco install $pkg -y --no-progress
    }
}

# Refresh Path again to be sure
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# ============================================================================
# 3. Visual Studio Build Tools (Critical for compiling some pip packages)
# ============================================================================
# We only install if we really can't find the compiler
Write-Info "Checking for C++ Build Tools..."
try {
    # Simple check: try to run cl.exe (MSVC compiler)
    # This is often not in path, so we check standard locations or registry
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $installed = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installed) {
            Write-Success "Visual Studio Build Tools detected."
        } else {
            throw "Not found"
        }
    } else {
        throw "Not found"
    }
} catch {
    Write-Warning "C++ Build Tools not found. Installing (this takes time)..."
    choco install visualstudio2022buildtools -y --no-progress --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive --norestart"
    Write-Info "Build tools installed. NOTE: You might need to restart your computer if builds fail later."
}

# ============================================================================
# 4. Python Virtual Environment
# ============================================================================
Write-Info "Setting up Python environment..."

$venvName = "rf-detr-venv-310"

# Find Python 3.10 executable specifically
$pyCmd = "python"
if (Get-Command py -ErrorAction SilentlyContinue) {
    # Use py launcher if available to select 3.10
    $pyCmd = "py -3.10"
}

if (-not (Test-Path $venvName)) {
    Invoke-Expression "$pyCmd -m venv $venvName"
    Write-Success "Created virtual environment: $venvName"
}

# Activate
$activateScript = ".\$venvName\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
} else {
    Write-Error "Could not find activation script. Setup failed."
    exit 1
}

# ============================================================================
# 5. Install Python Dependencies (Robust Method)
# ============================================================================
Write-Info "Installing Python libraries..."

python -m pip install --upgrade pip setuptools wheel

# 1. Install PyTorch with CUDA support EXPLICITLY first
# This prevents pip from picking up the CPU version from standard pypi
Write-Info "Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 2. Install other requirements
if (Test-Path "requirements.txt") {
    Write-Info "Installing requirements.txt..."
    # Exclude torch from requirements file installation to avoid overwriting
    # We use grep/select-string to filter it out temporarily
    Get-Content requirements.txt | Where-Object { $_ -notmatch "torch" } | Set-Content requirements_no_torch.temp
    pip install -r requirements_no_torch.temp
    Remove-Item requirements_no_torch.temp
}

# ============================================================================
# 6. Create Directories
# ============================================================================
$dirs = @("models", "data\inputs", "data\outputs", "clips", "logs")
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

# ============================================================================
# 7. Create Start Scripts
# ============================================================================
$startBat = @"
@echo off
call $venvName\Scripts\activate.bat
echo Starting Stream...
python run_mjpeg_stream.py
pause
"@
Set-Content -Path "start_stream.bat" -Value $startBat

Write-Success "Setup Complete!"
Write-Host ""
Write-Host "To start the application:" -ForegroundColor Cyan
Write-Host "  Double-click 'start_stream.bat'" -ForegroundColor White
Write-Host ""
Write-Host "IMPORTANT: Ensure you have placed 'best_rf-detr.pth' in the 'models' folder." -ForegroundColor Yellow
Write-Host ""
