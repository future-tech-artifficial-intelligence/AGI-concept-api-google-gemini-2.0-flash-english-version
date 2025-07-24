@echo off
echo 🎯 SYSTEME SEARX AVEC CAPTURE VISUELLE POUR L'IA
echo ===============================================
echo.

echo 📦 Étape 1: Installation des dépendances Python de base...
python install_searx_deps.py
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Erreur lors de l'installation des dépendances de base
    pause
    exit /b 1
)
echo.

echo 📸 Étape 2: Installation des dépendances de capture visuelle...
python install_searx_visual_deps.py
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ Capture visuelle non disponible, mais le système peut fonctionner sans
    echo Continuer quand même ? (O/N)
    set /p choice=
    if /i "%choice%" NEQ "O" (
        echo Installation annulée
        pause
        exit /b 1
    )
)
echo.

echo 🐳 Étape 3: Vérification de Docker...
call check_docker.bat
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Docker n'est pas prêt
    echo Veuillez démarrer Docker Desktop manuellement puis relancer ce script
    pause
    exit /b 1
)
echo.

echo 🚀 Étape 4: Démarrage du système Searx...
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

echo 🧪 Étape 5: Tests du système de base...
python test_searx_system.py
echo.

echo 📸 Étape 6: Tests du système de capture visuelle...
python test_searx_visual_system.py
echo.

echo 🎯 Étape 7: Démarrage de l'application IA avec vision...
echo Interface IA avec vision: http://localhost:4004
echo Interface Searx: http://localhost:8080
echo.
echo 💡 Testez avec ces phrases:
echo "Recherche des informations visuelles sur Python"
echo "Montre-moi des résultats sur l'intelligence artificielle" 
echo "Capture et analyse les résultats de recherche sur..."
echo.

python app.py

pause
