@echo off
echo 🐳 DOCKER DESKTOP VERIFICATION AND STARTUP
echo ================================================
echo.

echo 📋 Checking Docker status...
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker is not accessible
    echo.
    echo 🔧 Attempting to start Docker Desktop...
    
    REM Try to start Docker Desktop
    if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
        echo Starting Docker Desktop...
        start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        echo ⏳ Waiting for Docker to start (60 seconds)...
        timeout /t 60 /nobreak >nul
    ) else (
        echo ❌ Docker Desktop is not installed in the default directory
        echo.
        echo 📥 Please install Docker Desktop from:
        echo https://www.docker.com/products/docker-desktop/
        echo.
        pause
        exit /b 1
    )
    
    REM Check again after startup
    docker --version >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Docker Desktop could not start correctly
        echo.
        echo 🔧 Possible solutions:
        echo 1. Manually start Docker Desktop
        echo 2. Restart the computer
        echo 3. Reinstall Docker Desktop
        echo.
        pause
        exit /b 1
    )
)

echo ✅ Docker is available
docker --version

echo.
echo 🔍 Checking Searx container status...
docker ps -a --filter "name=ai_searx" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo 🚀 Docker Desktop is ready for Searx!
pause
