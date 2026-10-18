@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0venv\Scripts\voxprint-canvas.exe" goto run
rem PowerShell 7 (pwsh) first; Windows PowerShell 5.1 is the fallback.
set "VOXPRINT_PS=powershell"
where pwsh.exe >nul 2>nul && set "VOXPRINT_PS=pwsh"
%VOXPRINT_PS% -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 exit /b 1
:run
"%~dp0venv\Scripts\voxprint-canvas.exe" %*
