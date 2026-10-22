@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run setup.bat first.
  pause
  exit /b 1
)
set QT_QPA_PLATFORM=offscreen
set TF_NUM_INTEROP_THREADS=1
set TF_NUM_INTRAOP_THREADS=2
".venv\Scripts\python.exe" -m pytest -q
set "bsl_test_exit=%errorlevel%"
pause
exit /b %bsl_test_exit%
