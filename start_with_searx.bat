@echo off
echo 🔍 SYSTEME DE RECHERCHE SEARX POUR L'IA
echo ========================================
echo.

echo 📦 Étape 1: Installation des dépendances Python...
python install_searx_deps.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors de l'installation des dépendances
    pause
    exit /b 1
)
echo.

echo 🐳 Étape 2: Vérification de Docker...
call check_docker.bat
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker n'est pas prêt
    echo Veuillez démarrer Docker Desktop manuellement puis relancer ce script
    pause
    exit /b 1
)
echo.

echo 🚀 Étape 3: Démarrage du système Searx...
docker-compose -f docker-compose.searx.yml up -d
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors du démarrage de Searx
    echo Vérifiez que Docker Desktop est bien démarré
    pause
    exit /b 1
)
echo.

echo ⏳ Attente du démarrage de Searx (45 secondes)...
timeout /t 45 /nobreak >nul
echo.

echo 🧪 Étape 4: Tests du système...
python test_searx_system.py
echo.

echo 🎯 Étape 5: Démarrage de l'application IA...
echo Interface IA: http://localhost:4004
echo Interface Searx: http://localhost:8080
echo.
echo 💡 Conseil: Testez en demandant à l'IA: "Recherche des informations sur Python"
echo.

python app.py

pause
