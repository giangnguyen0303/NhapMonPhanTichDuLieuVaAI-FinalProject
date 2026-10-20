@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=C:\Users\tannh\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   IMDb Analytics ^& AI - Khoi chay Streamlit
echo ============================================
echo.

"%PY%" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo [LOI] App khong chay duoc. Hay kiem tra da cai chua:
    echo   "%PY%" -m pip install -r requirements.txt
    echo.
    pause
)
