@echo off
echo ========================================
echo     GESTIONNAIRE INTELLIGENT SEARX
echo ========================================
echo.

:menu
echo 1. Demarrer Searx (intelligent)
echo 2. Afficher l'etat de Searx
echo 3. Arreter toutes les instances
echo 4. Liberer le port 8080
echo 5. Test complet du systeme
echo 6. Quitter
echo.
set /p choice="Choisissez une option (1-6): "

if "%choice%"=="1" goto start
if "%choice%"=="2" goto status
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto freeport
if "%choice%"=="5" goto test
if "%choice%"=="6" goto end
echo Option invalide, veuillez reessayer.
goto menu

:start
echo.
echo ========================================
echo    DEMARRAGE INTELLIGENT DE SEARX
echo ========================================
python searx_smart_start.py start
echo.
pause
goto menu

:status
echo.
echo ========================================
echo        ETAT ACTUEL DE SEARX
echo ========================================
python searx_smart_start.py status
echo.
pause
goto menu

:stop
echo.
echo ========================================
echo      ARRET DE TOUTES LES INSTANCES
echo ========================================
python searx_smart_start.py stop
echo.
pause
goto menu

:freeport
echo.
echo ========================================
echo        LIBERATION DU PORT 8080
echo ========================================
call free_port_8080.bat
echo.
pause
goto menu

:test
echo.
echo ========================================
echo         TEST COMPLET DU SYSTEME
echo ========================================
echo.
echo 1. Verification de Docker...
docker --version
if errorlevel 1 (
    echo ERREUR: Docker n'est pas installe ou accessible
    pause
    goto menu
)
echo Docker OK
echo.

echo 2. Verification de Python...
python --version
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou accessible
    pause
    goto menu
)
echo Python OK
echo.

echo 3. Test des dependances...
python -c "import requests, psutil; print('Dependencies OK')"
if errorlevel 1 (
    echo ERREUR: Dependances manquantes
    echo Installation automatique...
    pip install requests psutil beautifulsoup4 selenium pillow
)
echo.

echo 4. Test du gestionnaire de ports...
python -c "from port_manager import PortManager; pm = PortManager(); print(f'Port 8080 available: {pm.is_port_available(8080)}')"
echo.

echo 5. Demarrage du test complet...
python searx_smart_start.py start
echo.
echo Test complete!
pause
goto menu

:end
echo.
echo Merci d'avoir utilise le gestionnaire Searx!
echo Au revoir!
pause
exit /b 0
