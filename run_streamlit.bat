@echo off
REM Nokia 5G Fronthaul Optimization - Streamlit Application Launcher
REM This script launches the Streamlit application

echo ================================================
echo Nokia 5G Fronthaul Optimization
echo Streamlit Application
echo ================================================
echo.
echo Starting Streamlit application...
echo.
echo Once the application starts:
echo - The browser will open automatically
echo - Default URL: http://localhost:8501
echo - Press Ctrl+C to stop the server
echo.
echo ================================================
echo.

cd /d "%~dp0"
streamlit run app.py --server.headless false

pause
