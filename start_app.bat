@echo off
title Application Searx AI
color 0E

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║          🚀 DÉMARRAGE AUTOMATIQUE - APPLICATION AI          ║
echo ║              Searx + Gemini + Analyse Visuelle              ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

echo 🔍 Vérification des prérequis...

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python non disponible
    pause
    exit /b 1
)

REM Vérifier les fichiers
if not exist "app.py" (
    echo ❌ Fichier app.py manquant
    pause
    exit /b 1
)

echo ✅ Prérequis OK
echo.
echo 🚀 Lancement de l'application avec Searx intégré...
echo.

REM Lancer l'application avec démarrage automatique de Searx
python run_with_searx.py

echo.
echo 👋 Application fermée
pause
