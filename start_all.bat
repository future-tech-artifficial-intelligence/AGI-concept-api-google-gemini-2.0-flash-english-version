@echo off
echo 🚀 Starting GeminiChat website with ngrok
echo.

echo 📱 Step 1: Starting Flask on port 5000...
start "Flask Server" cmd /k "python app.py"

echo ⏳ Waiting for Flask to start...
timeout /t 5 /nobreak >nul

echo 🌐 Step 2: Starting ngrok tunnel...
python ngrok_quick.py

pause
