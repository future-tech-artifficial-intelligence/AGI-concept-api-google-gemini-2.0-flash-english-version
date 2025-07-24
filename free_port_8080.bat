@echo off
echo 🔧 LIBÉRATION DU PORT 8080 POUR SEARX
echo ====================================
echo.

echo 🔍 Recherche des processus utilisant le port 8080...
netstat -ano | findstr :8080

echo.
echo 📋 Processus détectés:
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
    if not "%%a"=="0" (
        echo PID: %%a
        tasklist /fi "PID eq %%a" 2>nul | findstr /v "INFO:"
    )
)

echo.
echo ⚠️ Voulez-vous arrêter ces processus pour libérer le port 8080 ? (O/N)
set /p choice=

if /i "%choice%"=="O" (
    echo.
    echo 🛑 Arrêt des processus utilisant le port 8080...
    
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
        if not "%%a"=="0" (
            echo Arrêt du processus PID: %%a
            taskkill /PID %%a /F >nul 2>&1
            if %ERRORLEVEL% EQU 0 (
                echo ✅ Processus %%a arrêté
            ) else (
                echo ❌ Impossible d'arrêter le processus %%a
            )
        )
    )
    
    echo.
    echo ⏳ Vérification que le port est libéré...
    timeout /t 3 /nobreak >nul
    
    netstat -ano | findstr :8080 >nul
    if %ERRORLEVEL% EQU 0 (
        echo ⚠️ Le port 8080 est toujours occupé
        echo Vous devrez peut-être redémarrer votre ordinateur
    ) else (
        echo ✅ Port 8080 libéré avec succès !
    )
) else (
    echo ❌ Opération annulée
)

echo.
pause
