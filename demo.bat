@echo off
REM Vantage Demo Quick-Start (Windows)
REM Usage: demo.bat                    - Simulation mode
REM        demo.bat --through-wall     - Through-wall demo
REM        demo.bat --live --log FILE  - Live mode

echo.
echo   ========================================
echo   VANTAGE - Through-Wall Presence Detection
echo   ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   ERROR: Python not found. Install Python 3.10+ from python.org
    exit /b 1
)

REM Install deps if needed
python -c "import numpy, scipy, yaml, watchdog" >nul 2>&1
if %errorlevel% neq 0 (
    echo   Installing dependencies...
    python -m pip install -r requirements.txt --quiet
)

if "%1"=="--live" (
    echo   Mode: LIVE
    python run.py --dashboard %2 %3 %4 %5
) else (
    echo   Mode: SIMULATION
    echo   Dashboard: http://localhost:8088
    echo.
    python run.py --demo %*
)
