@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the Python 3.12 environment using the README quick start first.
  pause
  exit /b 1
)
if not exist "models\ku-mobilenet.keras" (
  ".venv\Scripts\python.exe" scripts\download_models.py
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -m bsl webcam
if errorlevel 1 pause
