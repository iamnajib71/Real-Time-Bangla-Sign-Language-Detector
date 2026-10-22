@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 (
    echo Python 3.12 is required. Install it, then run setup.bat again.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt -r requirements-motion.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --no-deps -e .
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\download_models.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\download_motion_models.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\download_sentence_model.py
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m bsl doctor --motion
if errorlevel 1 goto failed
echo Setup complete. Run run-app.bat or run-tests.bat.
exit /b 0
:failed
echo Setup did not finish. Read the error above before continuing.
pause
exit /b 1
