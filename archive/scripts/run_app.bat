@echo off
:: Investment Recommendation System Launcher for Windows
:: Just double-click this file to run the entire application

title Investment Recommendation System

echo.
echo ===================================
echo    INVESTMENT RECOMMENDATION SYSTEM
echo ===================================
echo.
echo Starting application...
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.6+ and try again
    goto end
)

:: Run the Python launcher
python run_app.py

:end
echo.
echo Press any key to exit...
pause >nul 