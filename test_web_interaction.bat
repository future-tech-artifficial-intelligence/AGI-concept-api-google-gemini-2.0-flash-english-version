@echo off
echo =====================================================
echo Test des Capacites d'Interaction Web de l'API Gemini
echo =====================================================
echo.

REM Vérifier si Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou pas dans le PATH
    echo Veuillez installer Python et réessayer
    pause
    exit /b 1
)

REM Vérifier si les dépendances sont installées
echo 📦 Vérification des dépendances...
python -c "import selenium" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Selenium non installé, installation en cours...
    pip install selenium
)

python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Requests non installé, installation en cours...
    pip install requests
)

echo ✅ Dépendances vérifiées
echo.

REM Lancer le test
echo 🚀 Lancement du test d'interaction web...
echo.
python run_web_interaction_test.py

echo.
echo ✨ Test terminé!
echo 📁 Consultez le dossier 'test_results_web_interaction' pour les rapports
echo.
pause
