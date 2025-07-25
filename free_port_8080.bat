@echo off
echo 🔧 FREEING UP PORT 8080 FOR SEARX
echo ====================================
echo.

echo 🔍 Searching for processes using port 8080...
netstat -ano | findstr :8080

echo.
echo 📋 Detected processes:
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
    if not "%%a"=="0" (
        echo PID: %%a
        tasklist /fi "PID eq %%a" 2>nul | findstr /v "INFO:"
    )
)

echo.
echo ⚠️ Do you want to stop these processes to free up port 8080? (Y/N)
set /p choice=

if /i "%choice%"=="Y" (
    echo.
    echo 🛑 Stopping processes using port 8080...
    
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
        if not "%%a"=="0" (
            echo Stopping process PID: %%a
            taskkill /PID %%a /F >nul 2>&1
            if %ERRORLEVEL% EQU 0 (
                echo ✅ Process %%a stopped
            ) else (
                echo ❌ Could not stop process %%a
            )
        )
    )
    
    echo.
    echo ⏳ Verifying that the port is freed...
    timeout /t 3 /nobreak >nul
    
    netstat -ano | findstr :8080 >nul
    if %ERRORLEVEL% EQU 0 (
        echo ⚠️ Port 8080 is still occupied
        echo You may need to restart your computer
    ) else (
        echo ✅ Port 8080 freed successfully!
    )
) else (
    echo ❌ Operation cancelled
)

echo.
pause
