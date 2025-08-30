# Genesis Trading Bot - Virtual Environment Activation Script
# For Windows PowerShell

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Get the directory of this script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Success "🚀 Genesis Trading Bot - Environment Setup"
Write-Host "========================================"

# Check Python version
function Test-PythonVersion {
    param($PythonCmd)
    
    $RequiredVersion = "3.11.8"
    
    try {
        $version = & $PythonCmd --version 2>&1
        if ($version -match "Python (\d+\.\d+\.\d+)") {
            $installedVersion = $matches[1]
            if ($installedVersion.StartsWith($RequiredVersion.Substring(0, 6))) {
                Write-Success "✓ Found Python $installedVersion"
                return $true
            }
        }
    } catch {
        return $false
    }
    return $false
}

# Find appropriate Python command
$PythonCmd = ""
if (Test-PythonVersion "python") {
    $PythonCmd = "python"
} elseif (Test-PythonVersion "python3") {
    $PythonCmd = "python3"
} elseif (Test-PythonVersion "python3.11") {
    $PythonCmd = "python3.11"
} elseif (Test-PythonVersion "py -3.11") {
    $PythonCmd = "py -3.11"
} else {
    Write-Error "❌ Error: Python 3.11.8 is required but not found"
    Write-Host "Please install Python 3.11.8 from python.org or using pyenv-win"
    Write-Host ""
    Write-Host "Download from: https://www.python.org/downloads/release/python-3118/"
    exit 1
}

# Virtual environment directory
$VenvDir = Join-Path $ProjectRoot ".venv"

# Create virtual environment if it doesn't exist
if (-not (Test-Path $VenvDir)) {
    Write-Warning "Creating virtual environment..."
    & $PythonCmd -m venv $VenvDir
    Write-Success "✓ Virtual environment created"
}

# Activate virtual environment
Write-Warning "Activating virtual environment..."
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"

if (Test-Path $ActivateScript) {
    & $ActivateScript
    Write-Success "✓ Virtual environment activated"
    Write-Host "  Path: $VenvDir"
} else {
    Write-Error "❌ Failed to find activation script"
    exit 1
}

# Upgrade pip to latest version
Write-Warning "Upgrading pip..."
& python -m pip install --quiet --upgrade pip

# Install pip-tools if not present
$pipToolsInstalled = & pip show pip-tools 2>$null
if (-not $pipToolsInstalled) {
    Write-Warning "Installing pip-tools..."
    & pip install --quiet pip-tools
}

# Determine which requirements to install based on TIER environment variable
$Tier = if ($env:TIER) { $env:TIER } else { "sniper" }
Write-Warning "Installing requirements for tier: $Tier"

# Install appropriate requirements
$RequirementsFile = switch ($Tier) {
    "sniper" { Join-Path $ProjectRoot "requirements\sniper.txt" }
    "hunter" { Join-Path $ProjectRoot "requirements\hunter.txt" }
    "strategist" { Join-Path $ProjectRoot "requirements\strategist.txt" }
    default {
        Write-Error "❌ Unknown tier: $Tier"
        exit 1
    }
}

if (Test-Path $RequirementsFile) {
    Write-Warning "Installing from $RequirementsFile..."
    & pip install -q -r $RequirementsFile
    Write-Success "✓ Dependencies installed"
} else {
    Write-Warning "⚠ Requirements file not found: $RequirementsFile"
}

# Install development dependencies if in dev mode
if (($env:DEV_MODE -eq "true") -or ($env:ENV -eq "development")) {
    $DevRequirements = Join-Path $ProjectRoot "requirements\dev.txt"
    if (Test-Path $DevRequirements) {
        Write-Warning "Installing development dependencies..."
        & pip install -q -r $DevRequirements
        Write-Success "✓ Development dependencies installed"
    }
}

# Set environment variables
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"
$env:GENESIS_ROOT = $ProjectRoot

# Display final status
Write-Host ""
Write-Success "✅ Environment Ready!"
Write-Host "========================================"
Write-Host "Python:      $(& python --version)"
Write-Host "Pip:         $(& pip --version | Select-String -Pattern '\d+\.\d+\.\d+' -AllMatches | ForEach-Object { $_.Matches[0].Value })"
Write-Host "Tier:        $Tier"
Write-Host "Project:     $ProjectRoot"
Write-Host "Virtual Env: $VenvDir"
Write-Host ""
Write-Host "To deactivate the environment, run: deactivate"