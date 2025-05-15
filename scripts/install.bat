REM Author: Felipe Artur Macedo Lima
REM Date: 05-14-2025
REM Description: This script installs the COVID-19, TB, and Pneumonia X-ray Analysis Tool on Windows.
REM Usage: Run this script in the command prompt to set up the environment and install dependencies.
REM This script assumes that Python 3 and pip are already installed on the system.

@echo off
:: COVID-19, TB, and Pneumonia X-ray Analysis Tool - Windows Installation Script

echo ===================================================================
echo     Installing COVID-19, TB, and Pneumonia X-ray Analysis Tool      
echo ===================================================================

:: Navigate to project root directory
cd %~dp0..

echo [1/4] Creating Python virtual environment...
py -3.9 -m venv venv
if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment. Please make sure Python 3 is installed.
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo [3/4] Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo Failed to install required packages.
    exit /b 1
)

echo [4/4] Creating necessary directories...
if not exist dataset\TRAIN mkdir dataset\TRAIN
if not exist dataset\VAL mkdir dataset\VAL
if not exist dataset\TEST mkdir dataset\TEST
if not exist models mkdir models
if not exist results mkdir results

echo ===================================================================
echo                     Installation Complete!                           
echo ===================================================================
echo.
echo To use the application:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate.bat
echo 2. Run the main application:
echo    python main.py
echo.
echo NOTE: Before training the model, you need to organize your dataset with the following structure:
echo   dataset\TRAIN\[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]
echo   dataset\VAL\[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]
echo   dataset\TEST\[COVID, NORMAL, PNEUMONIA, TUBERCULOSIS]
echo ===================================================================
pause
```