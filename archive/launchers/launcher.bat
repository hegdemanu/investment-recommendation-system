@echo off
REM Investment Recommendation System Launcher
REM For Windows systems

echo Starting Investment Recommendation System...
echo ============================================

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in the PATH. Please install Python and try again.
    goto end
)

REM Run the full analysis
python run_investment_system.py full-analysis

echo ============================================
echo Process complete. Press any key to exit.

:end
pause 