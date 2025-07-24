@echo off
echo 🚀 Demarrage de GeminiChat avec ngrok
echo.

echo 📱 Etape 1: Demarrage de Flask sur le port 5000...
start "Flask Server" cmd /k "python app.py"

echo ⏳ Attente que Flask demarre...
timeout /t 5 /nobreak >nul

echo 🌐 Etape 2: Demarrage du tunnel ngrok...
python ngrok_quick.py

pause
