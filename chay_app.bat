@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
title IMDb Analytics ^& AI

echo ============================================
echo   IMDb Analytics ^& AI - Khoi chay Streamlit
echo ============================================

set "FOUND="
where python >nul 2>nul
if errorlevel 1 goto try_py
python -c "import streamlit" >nul 2>nul
if errorlevel 1 goto try_py
set "FOUND=python"
goto run

:try_py
where py >nul 2>nul
if errorlevel 1 goto missing
py -3 -c "import streamlit" >nul 2>nul
if errorlevel 1 goto missing
set "FOUND=py"
goto run

:run
echo.
if "%FOUND%"=="py" (
    py -3 -m streamlit run app.py
) else (
    python -m streamlit run app.py
)
if errorlevel 1 (
    echo.
    echo   [LOI] App dung voi loi tren. Hay doc o tren de biet nguyen nhan.
    pause
)
goto end

:missing
echo.
echo   Ban chua co Python/Streamlit tren may.
echo     - Chua cai Python: tai https://www.python.org/downloads/
echo     - Da cai Python nhung thieu thu vien: chon Y ben duoi de cai.
echo.
choice /C YN /N /M "   Cai dat thu vien tu requirements.txt bay gio? [Y/N] "
if errorlevel 2 goto end
where python >nul 2>nul
if errorlevel 1 goto nopython
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   [LOI] Cai dat that bai. Hay chay thu:  pip install -r requirements.txt
    pause
    goto end
)
echo.
python -m streamlit run app.py
if errorlevel 1 pause
goto end

:nopython
echo.
echo   Khong tim thay Python. Hay cai Python truoc roi chay lai file nay.
pause
goto end

:end
