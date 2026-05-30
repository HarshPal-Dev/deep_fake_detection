@echo off
echo ============================================
echo  DeepFake Detector - Setup Script
echo ============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo.
    echo Please install Python 3.10 or 3.11 first:
    echo   https://www.python.org/downloads/
    echo   OR install Anaconda: https://www.anaconda.com/download
    echo.
    echo After installing Python, run this script again.
    pause
    exit /b 1
)

echo [OK] Python found.
python --version
echo.

:: Create virtual environment
echo Creating virtual environment...
python -m venv .venv
echo.

:: Activate and install packages
echo Installing packages (this may take a few minutes)...
call .venv\Scripts\activate.bat

pip install --upgrade pip --quiet
pip install streamlit>=1.35.0 --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
pip install timm>=0.9.0 Pillow>=10.0.0 numpy>=1.24.0 --quiet

echo.
echo ============================================
echo  Setup complete!
echo  Run: run_app.bat
echo ============================================
pause
