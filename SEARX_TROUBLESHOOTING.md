# 🛠️ Searx Troubleshooting Guide

## Problem: Docker not accessible

### Symptoms
```
unable to get image 'searxng/searxng:latest': error during connect:
Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.49/...":
open //./pipe/dockerDesktopLinuxEngine: The specified file was not found.
```

### Solutions

#### 1. Verify Docker Desktop
```cmd
# Check if Docker Desktop is installed
check_docker.bat

# If not installed, download from:
# https://www.docker.com/products/docker-desktop/
```

#### 2. Start Docker Desktop manually
1. Search for "Docker Desktop" in the Start menu
2. Right-click → "Run as administrator"
3. Wait for full startup (Docker icon in the taskbar)

#### 3. Restart Docker services
```cmd
# In an administrator terminal
net stop com.docker.service
net start com.docker.service
```

#### 4. Full verification
```cmd
# Complete Docker test
docker --version
docker info
docker ps
```

## Problem: Port 8080 occupied

### Symptoms
```
Error response from daemon: driver failed programming external connectivity
on endpoint ai_searx: Bind for 0.0.0.0:8080 failed: port is already allocated
```

### Solutions

#### 1. Identify the process
```cmd
netstat -ano | findstr :8080
```

#### 2. Stop the process
```cmd
# Replace PID with the found number
taskkill /PID <PID> /F
```

#### 3. Change the port (optional)
Modify in `docker-compose.searx.yml`:
```yaml
ports:
  - "8081:8080"  # Use port 8081 instead
```

## Problem: No search results

### Possible causes
1. Internet connectivity
2. Blocked search engines
3. Incorrect Searx configuration

### Solutions

#### 1. Connectivity test
```cmd
ping google.com
curl http://localhost:8080/stats
```

#### 2. Check logs
```cmd
docker logs ai_searx
```

#### 3. Restart Searx
```cmd
docker-compose -f docker-compose.searx.yml restart
```

## Problem: Python application cannot find Searx

### Solutions

#### 1. Verify integration
```python
python -c "from searx_interface import get_searx_interface; print('OK')"
```

#### 2. Manual test
```python
python searx_interface.py
```

#### 3. Check Python logs
Look in the console for messages:
- `✅ Searx interface integrated`
- `⚠️ Searx interface not available`

## 🆘 Quick diagnostic commands

```cmd
# Full status
docker ps -a | findstr searx
curl http://localhost:8080/
python test_searx_system.py

# Full cleanup
docker-compose -f docker-compose.searx.yml down --volumes
docker system prune -f

# Full restart
docker-compose -f docker-compose.searx.yml up -d --force-recreate
```

##  Support

If no solution works:

1.  **Check prerequisites:**
    -   Windows 10/11 with WSL2 enabled
    -   Docker Desktop 4.0+ installed
    -   4GB free RAM minimum

2.  **Collect information:**
    ```cmd
    docker --version
    docker info > docker-info.txt
    docker logs ai_searx > searx-logs.txt
    ```

3.  **Alternative solutions:**
    -   Use the existing web scraping system
    -   Temporarily disable Searx in `gemini_api_adapter.py`
