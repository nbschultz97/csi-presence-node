@echo off
REM Vantage Through-Wall Demo — Windows Quick Launch
REM Usage: demo_windows.bat [--through-wall]
REM
REM Opens the web dashboard in simulation mode.
REM Add --through-wall for attenuated signal scenarios.

cd /d "%~dp0\.."

echo.
echo   ========================================
echo     VANTAGE — Through-Wall Presence Demo
echo   ========================================
echo.

if "%1"=="--through-wall" (
    echo   Mode: Through-Wall Simulation
    echo   Profile: Lower thresholds, attenuated signals
    python run.py --demo --through-wall
) else (
    echo   Mode: Standard Simulation
    echo   Tip: Use --through-wall for through-wall demo
    python run.py --demo
)
