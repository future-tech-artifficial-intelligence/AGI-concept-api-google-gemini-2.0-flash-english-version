# 🚀 Quick Start Guide - Searx System

## Get Started in 5 Minutes

### 📋 Prerequisites
- [ ] Docker Desktop installed and running
- [ ] Python 3.8+ available
- [ ] Port 8080 free

### 🎯 Quick Steps

1.  **Automatic startup** (recommended)
    ```cmd
    start_with_searx.bat
    ```

2.  **Verification**
    -   ✅ artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interface: http://localhost:4004
    -   ✅ Searx Interface: http://localhost:8080

### 🧪 Quick Test

Type into the artificial intelligence API GOOGLE GEMINI 2.0 FLASH interface:
```
"Search for information on Python"
```

The artificial intelligence API GOOGLE GEMINI 2.0 FLASH should automatically use Searx for the search!

## ⚡ Useful Commands

```cmd
# System test
python test_searx_system.py

# Restart Searx
docker-compose -f docker-compose.searx.yml restart

# Full shutdown
docker-compose -f docker-compose.searx.yml down
```

## 🆘 Frequent Problems

| Problem            | Solution               |
|--------------------|------------------------|
| Docker not started | Launch Docker Desktop  |
| Port 8080 occupied | `netstat -ano \| findstr :8080` |
| No results         | Check internet + logs  |

---
**💡 Tip**: Use `start_with_searx.bat` for a fully automated startup!
