@echo off
echo =======================================================
echo  Smart Multi-Risk AI Platform - Localhost Launcher
echo =======================================================
echo.
if exist .venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] Using system Python environment...
)
echo.
echo [INFO] Launching default browser at http://localhost:5000 ...
start http://localhost:5000
echo [INFO] Starting Flask server on http://localhost:5000 (Press Ctrl+C to stop)...
echo.
python app.py
pause
