@echo off
REM Simple launcher script for Windows

echo === Investment Recommendation System ===

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Try to use the conda Python first
set PYTHON_PATH=.conda\bin\python.exe

REM If conda Python doesn't exist, try the system Python
if not exist "%PYTHON_PATH%" (
    set PYTHON_PATH=python
)

REM Check if run_investment_system.py exists
if not exist "run_investment_system.py" (
    echo Error: run_investment_system.py not found in the current directory.
    echo Press any key to exit.
    pause
    exit /b 1
)

REM Run the system
echo Starting the system...
"%PYTHON_PATH%" run_investment_system.py dashboard

echo.
echo Press any key to exit.
pause 