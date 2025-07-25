# 🚀 STARTUP INSTRUCTIONS

## Step 1: Start Docker Desktop

1.  **Open the Start menu** and search for "Docker Desktop"
2.  **Right-click** on Docker Desktop → **"Run as administrator"**
3.  **Wait** for Docker Desktop to launch completely (🐳 icon in the taskbar)

## Step 2: Verify Docker

Run this command to verify:
```cmd
check_docker.bat
```

## Step 3: Automatic Startup of the Searx System

Once Docker is ready, launch:
```cmd
start_with_searx.bat
```

## Step 4: Manual Test if Necessary

If automatic startup fails:
```cmd
# Check Docker
python searx_manager.py

# Full test
python test_searx_system.py

# Manual app startup
python app.py
```

## 🎯 Functionality Test

Once the application has started (http://localhost:4004), test with:
-   "Search for recent information about artificial intelligence API GOOGLE GEMINI 2.0 FLASH"
-   "Find news about Python"
-   "Look for programming tutorials"

The artificial intelligence API GOOGLE GEMINI 2.0 FLASH should automatically use Searx for these queries!

---
**💡 Note**: If Docker Desktop is not installed, download it from:
https://www.docker.com/products/docker-desktop/
