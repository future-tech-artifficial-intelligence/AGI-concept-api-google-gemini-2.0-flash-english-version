@echo off
echo 🐳 VÉRIFICATION ET DÉMARRAGE DE DOCKER DESKTOP
echo ================================================
echo.

echo 📋 Vérification du statut de Docker...
docker --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker n'est pas accessible
    echo.
    echo 🔧 Tentative de démarrage de Docker Desktop...
    
    REM Essayer de démarrer Docker Desktop
    if exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" (
        echo Démarrage de Docker Desktop...
        start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
        echo ⏳ Attente du démarrage de Docker (60 secondes)...
        timeout /t 60 /nobreak >nul
    ) else (
        echo ❌ Docker Desktop n'est pas installé dans le répertoire par défaut
        echo.
        echo 📥 Veuillez installer Docker Desktop depuis :
        echo https://www.docker.com/products/docker-desktop/
        echo.
        pause
        exit /b 1
    )
    
    REM Vérifier à nouveau après le démarrage
    docker --version >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Docker Desktop n'a pas pu démarrer correctement
        echo.
        echo 🔧 Solutions possibles :
        echo 1. Démarrer manuellement Docker Desktop
        echo 2. Redémarrer l'ordinateur
        echo 3. Réinstaller Docker Desktop
        echo.
        pause
        exit /b 1
    )
)

echo ✅ Docker est disponible
docker --version

echo.
echo 🔍 Vérification du statut des conteneurs Searx...
docker ps -a --filter "name=ai_searx" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo 🚀 Docker Desktop est prêt pour Searx !
pause
