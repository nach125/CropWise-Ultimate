@echo off
cd /d "%~dp0"
echo CropWise Ultimate setup
python --version
if errorlevel 1 (
  echo Python not found. Install Python 3.10+ and select Add Python to PATH.
  pause
  exit /b 1
)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
pause
