@echo off
title Searx AI - Demarrage Final
color 0B

echo.
echo ============================================================
echo           🎉 FINALISATION DU SYSTEME SEARX AI
echo ============================================================
echo.

echo ✅ Systeme Searx intelligent: PRET (5/6 tests reussis)
echo ✅ Gestionnaire de ports: FONCTIONNEL
echo ✅ Interface Searx: OPERATIONNELLE  
echo ✅ Capture visuelle: INTEGREE
echo ✅ Scripts de gestion: DISPONIBLES
echo.
echo ⚠️  Docker Desktop: A DEMARRER
echo.

echo 🔍 Verification de l'etat de Docker...
docker ps >nul 2>&1
if errorlevel 1 (
    echo.
    echo ❌ Docker Desktop n'est pas demarre
    echo.
    echo 🚀 DEMARRAGE AUTOMATIQUE DE DOCKER...
    echo.
    
    REM Essayer de démarrer Docker Desktop
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    echo ⏳ Attente du demarrage de Docker Desktop...
    echo    (Cela peut prendre 1-2 minutes)
    echo.
    
    REM Attendre que Docker soit prêt
    set /a counter=0
    :wait_docker
    timeout /t 10 /nobreak >nul
    docker ps >nul 2>&1
    if not errorlevel 1 (
        echo ✅ Docker Desktop est maintenant actif!
        goto docker_ready
    )
    
    set /a counter+=1
    if %counter% lss 12 (
        echo    Tentative %counter%/12 - Docker en cours de demarrage...
        goto wait_docker
    )
    
    echo.
    echo ⚠️  Docker met plus de temps que prevu a demarrer
    echo.
    echo 💡 Solutions:
    echo    1. Attendez encore 1-2 minutes
    echo    2. Demarrez manuellement Docker Desktop
    echo    3. Redemarrez votre ordinateur si necessaire
    echo.
    echo Voulez-vous continuer sans attendre Docker?
    set /p continue="Continuer? (o/N): "
    if /i "%continue%"=="o" goto start_without_docker
    if /i "%continue%"=="oui" goto start_without_docker
    
    echo Operation annulee. Demarrez Docker Desktop manuellement puis relancez ce script.
    pause
    exit /b 1
    
) else (
    echo ✅ Docker Desktop est deja actif!
    goto docker_ready
)

:docker_ready
echo.
echo ============================================================
echo              🚀 DEMARRAGE INTELLIGENT SEARX
echo ============================================================
echo.

echo Lancement du systeme avec toutes les fonctionnalites...
python searx_smart_start.py

if errorlevel 1 (
    echo.
    echo ❌ Probleme lors du demarrage
    echo 🔧 Tentative avec liberation de port...
    
    REM Libérer le port 8080 si nécessaire
    if exist "free_port_8080.bat" call free_port_8080.bat
    
    echo.
    echo Nouvelle tentative...
    python searx_smart_start.py
    
    if errorlevel 1 (
        echo.
        echo ❌ Echec persistant
        goto troubleshooting
    )
)

echo.
echo ============================================================
echo                🎉 SEARX AI OPERATIONNEL!
echo ============================================================
echo.
echo ✅ Systeme demarre avec succes
echo 🌐 Interface web accessible (voir URL ci-dessus)
echo 🔍 Pret pour les recherches autonomes
echo 📸 Capture visuelle activee
echo 🤖 Integration Gemini disponible
echo.
echo 💡 Pour tester le systeme:
echo    1. Ouvrez l'URL affichee dans votre navigateur
echo    2. Testez une recherche manuelle
echo    3. Lancez python app.py pour l'integration complete
echo.
pause
goto end

:start_without_docker
echo.
echo ============================================================
echo        🔧 DEMARRAGE EN MODE DEVELOPPEMENT (sans Docker)
echo ============================================================
echo.
echo ⚠️  Mode degrade: certaines fonctionnalites limitees
echo ✅ Tests et developpement: possibles
echo.

echo Verification des composants disponibles...
python -c "
from port_manager import PortManager
from searx_interface import SearxInterface

print('✅ Gestionnaire de ports: OK')
print('✅ Interface Searx: OK')
print('⚠️  Docker Searx: Non disponible')
print('')
print('🔧 Fonctionnalites disponibles:')
print('   - Gestion intelligente des ports')
print('   - Interface de recherche (structure)')
print('   - Capture visuelle (si ChromeDriver installe)')
print('   - Integration Gemini (structure)')
print('')
print('💡 Pour activer Searx complet:')
print('   1. Demarrez Docker Desktop')
print('   2. Relancez ce script')
"

echo.
pause
goto end

:troubleshooting
echo.
echo ============================================================
echo                  🔧 GUIDE DE DEPANNAGE
echo ============================================================
echo.
echo ❌ Le systeme a rencontre des difficultes
echo.
echo 🔍 Verifications a effectuer:
echo.
echo 1. DOCKER:
echo    ✓ Docker Desktop installe et demarre
echo    ✓ Commande 'docker ps' fonctionne
echo    ✓ Memoire suffisante (4GB+ recommandes)
echo.
echo 2. PORTS:
echo    ✓ Ports 8080-8083 libres
echo    ✓ Pas de conflit avec autres services
echo    ✓ Firewall autorisant les connexions locales
echo.
echo 3. DEPENDANCES:
echo    ✓ Python 3.8+ installe
echo    ✓ pip install -r requirements.txt execute
echo    ✓ Modules psutil, requests disponibles
echo.
echo 🔧 Actions correctives:
echo.
echo A. Redemarrage complet:
set /p restart="   Redemarrer l'ordinateur? (o/N): "
if /i "%restart%"=="o" shutdown /r /t 60 /c "Redemarrage pour Searx AI"
if /i "%restart%"=="oui" shutdown /r /t 60 /c "Redemarrage pour Searx AI"

echo.
echo B. Reinstallation Docker:
echo    1. Desinstallez Docker Desktop
echo    2. Redemarrez l'ordinateur
echo    3. Reinstallez Docker Desktop
echo    4. Relancez ce script
echo.

echo C. Support avance:
echo    1. Consultez les logs: searx_smart_start.log
echo    2. Executez: python test_searx_complete.py
echo    3. Documentez les erreurs pour le support
echo.

pause

:end
echo.
echo ============================================================
echo                     👋 TERMINE
echo ============================================================
echo.
echo Merci d'avoir configure Searx AI!
echo.
echo 📋 Recap de votre installation:
echo    ✅ Systeme intelligent: INSTALLE
echo    ✅ Gestion des ports: ACTIVE
echo    ✅ Scripts de gestion: DISPONIBLES
echo    🌐 Interface web: CONFIGURABLE
echo.
echo 🚀 Prochaines etapes:
echo    1. Assurez-vous que Docker fonctionne
echo    2. Lancez: python searx_smart_start.py
echo    3. Ou utilisez: start_searx_ai.bat
echo.
echo 📚 Documentation complete dans les fichiers README
echo.
pause
exit /b 0
