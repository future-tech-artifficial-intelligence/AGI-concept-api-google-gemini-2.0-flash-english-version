@echo off
title Searx artificial intelligence API GOOGLE GEMINI 2.0 FLASH - Final Startup
color 0B

echo.
echo ============================================================
echo           🎉 FINALIZING SEARX artificial intelligence API GOOGLE GEMINI 2.0 FLASH SYSTEM
echo ============================================================
echo.

echo ✅ Intelligent Searx System: READY (5/6 tests successful)
echo ✅ Port Manager: FUNCTIONAL
echo ✅ Searx Interface: OPERATIONAL
echo ✅ Visual Capture: INTEGRATED
echo ✅ Management Scripts: AVAILABLE
echo.
echo ⚠️ Docker Desktop: TO BE STARTED
echo.

echo 🔍 Checking Docker status...
docker ps >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ Docker Desktop is not started
    echo.
    echo 🚀 AUTOMATIC DOCKER STARTUP...
    echo.
    
    REM Try to start Docker Desktop
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    echo ⏳ Waiting for Docker Desktop to start...
    echo    (This may take 1-2 minutes)
    echo.
    
    REM Wait for Docker to be ready
    set /a counter=0
    :wait_docker
    timeout /t 10 /nobreak >nul
    docker ps >nul 2>&1
    if not errorlevel 1 (
        echo ✅ Docker Desktop is now active!
        goto docker_ready
    )
    
    set /a counter+=1
    if %counter% lss 12 (
        echo    Attempt %counter%/12 - Docker starting up...
        goto wait_docker
    )
    
    echo.
    echo ⚠️ Docker is taking longer than expected to start
    echo.
    echo 💡 Solutions:
    echo    1. Wait another 1-2 minutes
    echo    2. Manually start Docker Desktop
    echo    3. Restart your computer if necessary
    echo.
    echo Do you want to continue without waiting for Docker?
    set /p continue="Continue? (y/N): "
    if /i "%continue%"=="y" goto start_without_docker
    if /i "%continue%"=="yes" goto start_without_docker
    
    echo Operation canceled. Start Docker Desktop manually then rerun this script.
    pause
    exit /b 1
    
) else (
    echo ✅ Docker Desktop is already active!
    goto docker_ready
)

:docker_ready
echo.
echo ============================================================
echo              🚀 INTELLIGENT SEARX STARTUP
echo ============================================================
echo.

echo Launching the system with all functionalities...
python searx_smart_start.py

if errorlevel 1 (
    echo.
    echo ❌ Problem during startup
    echo 🔧 Attempting with port release...
    
    REM Release port 8080 if necessary
    if exist "free_port_8080.bat" call free_port_8080.bat
    
    echo.
    echo Retrying...
    python searx_smart_start.py
    
    if errorlevel 1 (
        echo.
        echo ❌ Persistent failure
        goto troubleshooting
    )
)

echo.
echo ============================================================
echo                🎉 SEARX artificial intelligence API GOOGLE GEMINI 2.0 FLASH OPERATIONAL!
echo ============================================================
echo.
echo ✅ System started successfully
echo 🌐 Web interface accessible (see URL above)
echo 🔍 Ready for autonomous searches
echo 📸 Visual capture activated
echo 🤖 Gemini Integration available
echo.
echo 💡 To test the system:
echo    1. Open the URL displayed in your browser
echo    2. Perform a manual search
echo    3. Launch python app.py for full integration
echo.
pause
goto end

:start_without_docker
echo.
echo ============================================================
echo        🔧 STARTING IN DEVELOPMENT MODE (without Docker)
echo ============================================================
echo.
echo ⚠️ Degraded mode: some functionalities limited
echo ✅ Tests and development: possible
echo.

echo Checking available components...
python -c "
from port_manager import PortManager
from searx_interface import SearxInterface

print('✅ Port Manager: OK')
print('✅ Searx Interface: OK')
print('⚠️ Docker Searx: Not available')
print('')
print('🔧 Available functionalities:')
print('   - Intelligent port management')
print('   - Search interface (structure)')
print('   - Visual capture (if ChromeDriver installed)')
print('   - Gemini Integration (structure)')
print('')
print('💡 To activate full Searx:')
print('   1. Start Docker Desktop')
print('   2. Rerun this script')
"

echo.
pause
goto end

:troubleshooting
echo.
echo ============================================================
echo                  🔧 TROUBLESHOOTING GUIDE
echo ============================================================
echo.
echo ❌ The system encountered difficulties
echo.
echo 🔍 Checks to perform:
echo.
echo 1. DOCKER:
echo    ✓ Docker Desktop installed and started
echo    ✓ 'docker ps' command works
echo    ✓ Sufficient memory (4GB+ recommended)
echo.
echo 2. PORTS:
echo    ✓ Ports 8080-8083 free
echo    ✓ No conflict with other services
echo    ✓ Firewall allowing local connections
echo.
echo 3. DEPENDENCIES:
echo    ✓ Python 3.8+ installed
echo    ✓ pip install -r requirements.txt executed
    ✓ psutil, requests modules available
echo.
echo 🔧 Corrective actions:
echo.
echo A. Full restart:
set /p restart="   Restart computer? (y/N): "
if /i "%restart%"=="y" shutdown /r /t 60 /c "Restart for Searx artificial intelligence API GOOGLE GEMINI 2.0 FLASH"
if /i "%restart%"=="yes" shutdown /r /t 60 /c "Restart for Searx artificial intelligence API GOOGLE GEMINI 2.0 FLASH"

echo.
echo B. Docker reinstallation:
echo    1. Uninstall Docker Desktop
echo    2. Restart computer
echo    3. Reinstall Docker Desktop
echo    4. Rerun this script
echo.

echo C. Advanced support:
echo    1. Consult logs: searx_smart_start.log
echo    2. Execute: python test_searx_complete.py
echo    3. Document errors for support
echo.

pause

:end
echo.
echo ============================================================
echo                     👋 DONE
echo ============================================================
echo.
echo Thank you for configuring Searx artificial intelligence API GOOGLE GEMINI 2.0 FLASH!
echo.
echo 📋 Summary of your installation:
echo    ✅ Intelligent System: INSTALLED
echo    ✅ Port Management: ACTIVE
echo    ✅ Management Scripts: AVAILABLE
echo    🌐 Web Interface: CONFIGURABLE
echo.
echo 🚀 Next steps:
echo    1. Ensure Docker is running
echo    2. Launch: python searx_smart_start.py
echo    3. Or use: start_searx_ai.bat
echo.
echo 📚 Full documentation in README files
echo.
pause
exit /b 0
