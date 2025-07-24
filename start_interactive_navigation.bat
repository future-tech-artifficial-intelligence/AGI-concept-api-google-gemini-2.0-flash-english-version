@echo off
REM Script de lancement rapide pour le système de navigation interactive Gemini
REM Version Windows Batch

title Lanceur Système Navigation Interactive Gemini

echo ===============================================================================
echo                SYSTEME DE NAVIGATION INTERACTIVE GEMINI
echo                          Lanceur Windows
echo ===============================================================================

REM Vérification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python n'est pas installé ou non accessible
    echo 💡 Installez Python depuis https://python.org
    pause
    exit /b 1
)

REM Vérification du fichier lanceur
if not exist "quick_launcher.py" (
    echo ❌ Fichier quick_launcher.py non trouvé
    echo 🔧 Assurez-vous d'être dans le bon répertoire
    pause
    exit /b 1
)

REM Affichage des informations système
echo 🐍 Python détecté
python --version
echo 📁 Répertoire: %CD%
echo.

REM Lancement du lanceur Python
echo 🚀 Démarrage du lanceur interactif...
echo.
python quick_launcher.py

REM Gestion de l'erreur
if errorlevel 1 (
    echo.
    echo ❌ Erreur lors du lancement
    echo 📋 Codes d'erreur courants:
    echo    1 - Problème de configuration
    echo    2 - Dépendances manquantes
    echo    3 - Erreur de script
    echo.
    echo 💡 Solutions possibles:
    echo    - Exécutez l'installation (option 1 du menu)
    echo    - Vérifiez votre configuration
    echo    - Consultez les logs d'erreur
)

echo.
echo 🏁 Session terminée
pause
