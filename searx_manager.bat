@echo off
echo ========================================
echo     SEARX SMART MANAGER
echo ========================================
echo.

:menu
echo 1. Start Searx (smart)
echo 2. Show Searx status
echo 3. Stop all instances
echo 4. Free port 8080
echo 5. Full system test
echo 6. Exit
echo.
set /p choice="Choose an option (1-6): "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto status
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto freeport
if "%choice%"=="5" goto test
if "%choice%"=="6" goto end
echo Invalid option, please try again.
goto menu

:start
echo.
echo ========================================
echo    STARTING SEARX SMARTLY
echo ========================================
python searx_smart_start.py start
echo.
pause
goto menu

:status
echo.
echo ========================================
echo        CURRENT SEARX STATUS
echo ========================================
python searx_smart_start.py status
echo.
pause
goto menu

:stop
echo.
echo ========================================
echo      STOPPING ALL INSTANCES
echo ========================================
python searx_smart_start.py stop
echo.
pause
goto menu

:freeport
echo.
echo ========================================
echo        FREEING PORT 8080
echo ========================================
call free_port_8080.bat
echo.
pause
goto menu

:test
echo.
echo ========================================
echo         FULL SYSTEM TEST
echo ========================================
echo.
echo 1. Checking Docker...
docker --version
if errorlevel 1 (
    echo ERROR: Docker is not installed or accessible
    pause
    goto menu
)
echo Docker OK
echo.

echo 2. Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or accessible
    pause
    goto menu
)
echo Python OK
echo.

echo 3. Testing dependencies...
python -c "import requests, psutil; print('Dependencies OK')"
if errorlevel 1 (
    echo ERROR: Missing dependencies
    echo Automatic installation...
    pip install requests psutil beautifulsoup4 selenium pillow
)
echo.

echo 4. Testing port manager...
python -c "from port_manager import PortManager; pm = PortManager(); print(f'Port 8080 available: {pm.is_port_available(8080)}')"
echo.

echo 5. Starting full test...
python searx_smart_start.py start
echo.
echo Test complete!
pause
goto menu

:end
echo.
echo Thank you for using the Searx manager!
echo Goodbye!
pause
exit /b 0
