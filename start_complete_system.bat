@echo off
title Searx AI - Demarrage Complet avec App
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║               🚀 SEARX AI - DÉMARRAGE COMPLET               ║
echo  ║                    Application + Searx                      ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

REM Vérifications préliminaires
echo 🔍 Vérifications préliminaires...
echo.

REM Vérifier Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERREUR: Python non disponible
    echo 💡 Installez Python depuis: https://python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python detecte

REM Vérifier les dépendances critiques
python -c "import requests, psutil; print('✅ Dependances principales OK')" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Installation des dependances...
    pip install requests psutil beautifulsoup4 selenium pillow lxml
    if errorlevel 1 (
        echo ❌ Echec installation dependances
        pause
        exit /b 1
    )
)

REM Vérifier les fichiers du système
if not exist "port_manager.py" (
    echo ❌ ERREUR: Système Searx incomplet
    echo 💡 Assurez-vous que tous les fichiers sont présents
    pause
    exit /b 1
)
echo ✅ Fichiers systeme OK

echo.
echo ════════════════════════════════════════════════════════════════
echo                    🚀 DÉMARRAGE AUTOMATIQUE
echo ════════════════════════════════════════════════════════════════
echo.

echo 📋 Le système va maintenant:
echo    1. 🐳 Vérifier et démarrer Docker si nécessaire
echo    2. 🔧 Initialiser le gestionnaire de ports intelligent
echo    3. 🔍 Démarrer Searx automatiquement
echo    4. 📸 Activer la capture visuelle
echo    5. 🌐 Lancer l'application Flask complète
echo.

echo ⚡ TOUT SERA AUTOMATIQUE - Aucune intervention requise!
echo.
echo Voulez-vous continuer?
set /p confirm="Demarrer le systeme complet? (O/n): "
if /i "%confirm%"=="n" exit /b 0
if /i "%confirm%"=="non" exit /b 0

echo.
echo ════════════════════════════════════════════════════════════════
echo               🎯 LANCEMENT DE L'APPLICATION COMPLÈTE
echo ════════════════════════════════════════════════════════════════
echo.
echo 🚀 Démarrage automatique en cours...
echo.
echo 📝 Le système va maintenant:
echo    1. Vérifier et démarrer Docker automatiquement
echo    2. Initialiser Searx avec gestion intelligente des ports
echo    3. Lancer l'application Flask avec toutes les fonctionnalités
echo.
echo 📝 LOGS EN TEMPS RÉEL:
echo ────────────────────────────────────────────────────────────────

REM Lancer le démarreur unifié qui gère tout automatiquement
python start_app_with_searx.py

echo.
echo ════════════════════════════════════════════════════════════════
echo                        📊 RÉSULTAT FINAL
echo ════════════════════════════════════════════════════════════════

if errorlevel 1 (
    echo.
    echo ❌ L'application a rencontré un problème
    echo.
    echo 🔧 SOLUTIONS DE DÉPANNAGE:
    echo.
    echo 1. DOCKER:
    echo    - Démarrez Docker Desktop manuellement
    echo    - Attendez qu'il soit complètement initialisé
    echo    - Relancez ce script
    echo.
    echo 2. PORTS:
    echo    - Libérez le port 8080: free_port_8080.bat
    echo    - Ou le système le fera automatiquement
    echo.
    echo 3. DIAGNOSTIC:
    echo    - Lancez: python test_searx_complete.py
    echo    - Consultez les logs pour plus de détails
    echo.
    echo 💡 Dans la plupart des cas, relancer suffit!
    echo.
) else (
    echo.
    echo ✅ APPLICATION DÉMARRÉE AVEC SUCCÈS!
    echo.
    echo 🎉 SYSTÈME SEARX AI COMPLÈTEMENT OPÉRATIONNEL
    echo.
    echo 🌐 Votre application est maintenant accessible
    echo 🔍 Searx intégré pour recherches autonomes
    echo 📸 Analyse visuelle activée
    echo 🤖 IA Gemini avec capacités web
    echo.
    echo 💡 L'application continue de fonctionner...
    echo    Fermez cette fenêtre pour arrêter le système
    echo.
)

echo ════════════════════════════════════════════════════════════════
echo.
pause
exit /b 0
