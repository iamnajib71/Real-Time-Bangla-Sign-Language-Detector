@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
if not exist "models\ku-mobilenet.keras" (
  ".venv\Scripts\python.exe" scripts\download_models.py
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m bsl doctor
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m bsl gui
if errorlevel 1 goto failed
exit /b 0
:failed
echo The app did not start successfully. Check the error above or run setup.bat.
pause
exit /b 1
