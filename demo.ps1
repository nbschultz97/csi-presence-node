#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Vantage Demo Quick-Start (Windows/PowerShell)
.DESCRIPTION
    One-command demo launcher. Checks dependencies, installs if needed,
    and starts the web dashboard in simulation mode.
.EXAMPLE
    .\demo.ps1                    # Demo mode (synthetic data)
    .\demo.ps1 -ThroughWall      # Through-wall demo scenario
    .\demo.ps1 -Live -Log .\data\csi_raw.log   # Live with log file
    .\demo.ps1 -Port 9090        # Custom port
#>
param(
    [switch]$ThroughWall,
    [switch]$Live,
    [string]$Log,
    [string]$Replay,
    [int]$Port = 8088,
    [switch]$SkipCheck
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root

Write-Host ""
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host "  🎯 VANTAGE — Through-Wall Presence Detection" -ForegroundColor Cyan
Write-Host "  ========================================" -ForegroundColor Cyan
Write-Host ""

# --- Dependency check ---
if (-not $SkipCheck) {
    Write-Host "  Checking dependencies..." -ForegroundColor DarkGray

    # Python
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        Write-Host "  ❌ Python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
        exit 1
    }
    $pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
    Write-Host "  ✅ Python $pyVer" -ForegroundColor Green

    # Check key packages
    $missing = @()
    foreach ($pkg in @("numpy", "scipy", "yaml", "watchdog")) {
        $check = python -c "import $pkg" 2>&1
        if ($LASTEXITCODE -ne 0) { $missing += $pkg }
    }

    if ($missing.Count -gt 0) {
        Write-Host "  ⚠️  Missing packages: $($missing -join ', ')" -ForegroundColor Yellow
        Write-Host "  Installing requirements..." -ForegroundColor Yellow
        python -m pip install -r requirements.txt --quiet
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ❌ pip install failed" -ForegroundColor Red
            exit 1
        }
        Write-Host "  ✅ Dependencies installed" -ForegroundColor Green
    } else {
        Write-Host "  ✅ All dependencies present" -ForegroundColor Green
    }
}

# --- Build command ---
$cmd = @("python", "run.py")

if ($Live) {
    $cmd += "--dashboard"
    if ($Log) { $cmd += "--log"; $cmd += $Log }
} elseif ($Replay) {
    $cmd += "--dashboard"
    $cmd += "--replay"; $cmd += $Replay
} else {
    $cmd += "--demo"
}

if ($ThroughWall) { $cmd += "--through-wall" }
$cmd += "--port"; $cmd += $Port.ToString()

Write-Host ""
Write-Host "  Mode: $(if ($Live) { 'LIVE' } elseif ($Replay) { 'REPLAY' } else { 'SIMULATION' })" -ForegroundColor White
if ($ThroughWall) { Write-Host "  Profile: Through-Wall" -ForegroundColor Yellow }
Write-Host "  Dashboard: http://localhost:$Port" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

# Launch
& $cmd[0] $cmd[1..($cmd.Length-1)]

Pop-Location
