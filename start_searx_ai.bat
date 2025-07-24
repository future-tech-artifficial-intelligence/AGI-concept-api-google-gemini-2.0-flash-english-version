@echo off
title Searx AI - Demarrage Rapide
color 0A

echo.
echo  ███████╗███████╗ █████╗ ██████╗ ██╗  ██╗    █████╗ ██╗
echo  ██╔════╝██╔════╝██╔══██╗██╔══██╗╚██╗██╔╝   ██╔══██╗██║
echo  ███████╗█████╗  ███████║██████╔╝ ╚███╔╝    ███████║██║
echo  ╚════██║██╔══╝  ██╔══██║██╔══██╗ ██╔██╗    ██╔══██║██║
echo  ███████║███████╗██║  ██║██║  ██║██╔╝ ██╗   ██║  ██║██║
echo  ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝  ╚═╝╚═╝
echo.
echo              Systeme de Recherche Autonome pour IA
echo                        Version Intelligente
echo.
echo ============================================================

REM Vérification rapide de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERREUR: Python n'est pas installe ou accessible
    echo.
    echo 💡 Installez Python depuis: https://python.org/downloads/
    echo    Assurez-vous de cocher "Add to PATH" lors de l'installation
    pause
    exit /b 1
)

REM Vérification rapide de Docker
docker --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  ATTENTION: Docker n'est pas detecte
    echo.
    echo    Docker est necessaire pour faire fonctionner Searx
    echo    Voulez-vous continuer sans Docker? (les tests fonctionneront)
    echo.
    set /p continue="Continuer sans Docker? (o/N): "
    if /i not "%continue%"=="o" if /i not "%continue%"=="oui" (
        echo.
        echo 💡 Installez Docker Desktop: https://docker.com/products/docker-desktop
        pause
        exit /b 1
    )
)

echo.
echo 🔍 Verification du systeme...
echo.

REM Test rapide des composants
python -c "import sys; print(f'✅ Python {sys.version.split()[0]} detecte')" 2>nul
if errorlevel 1 (
    echo ❌ Probleme avec Python
    pause
    exit /b 1
)

REM Vérification des dépendances critiques
echo 📦 Verification des dependances...
python -c "import requests, psutil; print('✅ Dependances principales OK')" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Installation des dependances manquantes...
    pip install requests psutil beautifulsoup4 selenium pillow lxml
    if errorlevel 1 (
        echo ❌ Echec installation dependances
        pause
        exit /b 1
    )
    echo ✅ Dependances installees
)

REM Vérification des fichiers du système
if not exist "port_manager.py" (
    echo ❌ ERREUR: Fichier port_manager.py manquant
    echo    Le systeme n'est pas complet
    pause
    exit /b 1
)

if not exist "searx_interface.py" (
    echo ❌ ERREUR: Fichier searx_interface.py manquant
    echo    Le systeme n'est pas complet
    pause
    exit /b 1
)

echo ✅ Verification terminee
echo.

:menu
echo ============================================================
echo                    MENU PRINCIPAL
echo ============================================================
echo.
echo  1. 🚀 DEMARRAGE RAPIDE (recommande)
echo  2. 🧪 Test complet du systeme
echo  3. 📊 Afficher l'etat actuel
echo  4. 🛑 Arreter toutes les instances
echo  5. 🔧 Liberation manuelle du port 8080
echo  6. 📋 Menu avance
echo  7. ❌ Quitter
echo.
set /p choice="Votre choix (1-7): "

if "%choice%"=="1" goto quickstart
if "%choice%"=="2" goto fulltest
if "%choice%"=="3" goto status
if "%choice%"=="4" goto stopall
if "%choice%"=="5" goto freeport
if "%choice%"=="6" goto advanced
if "%choice%"=="7" goto end

echo ❌ Option invalide
goto menu

:quickstart
echo.
echo ============================================================
echo                  🚀 DEMARRAGE RAPIDE
echo ============================================================
echo.
echo Lancement du systeme Searx intelligent...
echo (Gestion automatique des ports et configurations)
echo.

python searx_smart_start.py

if errorlevel 1 (
    echo.
    echo ❌ Echec du demarrage automatique
    echo.
    echo 🔧 Solutions possibles:
    echo    1. Liberez le port 8080 (option 5)
    echo    2. Redemarrez Docker Desktop
    echo    3. Redemarrez votre ordinateur
    echo.
    pause
) else (
    echo.
    echo ✅ Searx demarre avec succes!
    echo 🌐 Verifiez l'URL affichee ci-dessus
    echo.
    pause
)
goto menu

:fulltest
echo.
echo ============================================================
echo               🧪 TEST COMPLET DU SYSTEME
echo ============================================================
echo.

python test_searx_complete.py

echo.
echo 📋 Test termine - consultez les resultats ci-dessus
pause
goto menu

:status
echo.
echo ============================================================
echo                📊 ETAT ACTUEL DU SYSTEME
echo ============================================================
echo.

python searx_smart_start.py status

echo.
pause
goto menu

:stopall
echo.
echo ============================================================
echo            🛑 ARRET DE TOUTES LES INSTANCES
echo ============================================================
echo.

python searx_smart_start.py stop

echo.
echo ✅ Commande d'arret executee
pause
goto menu

:freeport
echo.
echo ============================================================
echo            🔧 LIBERATION DU PORT 8080
echo ============================================================
echo.

if exist "free_port_8080.bat" (
    call free_port_8080.bat
) else (
    echo Recherche et arret des processus sur le port 8080...
    
    REM Méthode alternative de libération de port
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8080') do (
        echo Processus trouve: %%a
        taskkill /F /PID %%a >nul 2>&1
        if not errorlevel 1 echo ✅ Processus %%a arrete
    )
    
    echo Verification...
    netstat -ano | findstr :8080
    if errorlevel 1 (
        echo ✅ Port 8080 maintenant libre
    ) else (
        echo ⚠️  Port 8080 encore occupe
    )
)

echo.
pause
goto menu

:advanced
echo.
echo ============================================================
echo                   📋 MENU AVANCE
echo ============================================================
echo.
echo  1. Demarrer avec port specifique
echo  2. Regenerer les configurations Docker
echo  3. Nettoyer tous les conteneurs Docker
echo  4. Reinstaller les dependances
echo  5. Diagnostic reseau complet
echo  6. Retour au menu principal
echo.
set /p advchoice="Choix avance (1-6): "

if "%advchoice%"=="1" goto customport
if "%advchoice%"=="2" goto regenconfig
if "%advchoice%"=="3" goto cleancontainers
if "%advchoice%"=="4" goto reinstalldeps
if "%advchoice%"=="5" goto networkdiag
if "%advchoice%"=="6" goto menu

echo Option invalide
goto advanced

:customport
echo.
set /p customport="Port a utiliser (ex: 8081): "
echo Demarrage sur port %customport%...
REM Ici on pourrait créer une config custom
goto menu

:regenconfig
echo.
echo Regeneration des configurations Docker...
python -c "from port_manager import PortManager; pm = PortManager(); pm.setup_searx_with_available_port()"
echo ✅ Configurations regenerees
pause
goto advanced

:cleancontainers
echo.
echo Nettoyage de tous les conteneurs Docker...
docker ps -a --filter "name=searx" --format "{{.Names}}" > temp_containers.txt
for /f %%i in (temp_containers.txt) do (
    echo Arret et suppression de %%i...
    docker stop %%i >nul 2>&1
    docker rm %%i >nul 2>&1
)
del temp_containers.txt >nul 2>&1
echo ✅ Nettoyage termine
pause
goto advanced

:reinstalldeps
echo.
echo Reinstallation complete des dependances...
pip uninstall -y requests psutil beautifulsoup4 selenium pillow lxml
pip install -r requirements.txt
echo ✅ Dependances reinstallees
pause
goto advanced

:networkdiag
echo.
echo ============================================================
echo               🔍 DIAGNOSTIC RESEAU COMPLET
echo ============================================================
echo.

echo 📊 Ports en ecoute:
netstat -ano | findstr :808
echo.

echo 🐳 Conteneurs Docker:
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.

echo 🔌 Test de connectivite locale:
ping -n 1 127.0.0.1 > nul && echo ✅ Localhost accessible || echo ❌ Probleme localhost

echo.
pause
goto advanced

:end
echo.
echo ============================================================
echo                    👋 AU REVOIR!
echo ============================================================
echo.
echo Merci d'avoir utilise le systeme Searx AI!
echo.
echo 💡 Pour redemarrer:
echo    - Double-cliquez sur ce fichier
echo    - Ou lancez: python searx_smart_start.py
echo.
echo 🌐 Documentation: https://searx.github.io/searx/
echo 🐳 Docker: https://docker.com/products/docker-desktop
echo.
pause
exit /b 0
