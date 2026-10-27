@echo off
rem Double-click to start the viewer. The first run creates a private Python environment (a few minutes).
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up for the first time...
    python -m venv .venv || (echo Python 3.9+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
)
".venv\Scripts\python.exe" viewer.py %*
if errorlevel 1 pause
