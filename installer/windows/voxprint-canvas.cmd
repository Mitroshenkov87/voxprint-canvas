@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0venv\Scripts\voxprint-canvas.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
  if errorlevel 1 exit /b 1
)
"%~dp0venv\Scripts\voxprint-canvas.exe" %*
