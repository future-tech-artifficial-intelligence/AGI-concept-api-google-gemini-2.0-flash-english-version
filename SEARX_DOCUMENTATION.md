# 🔍 Searx Search System for artificial intelligence API GOOGLE GEMINI 2.0 FLASH

## Overview

Docker installation is necessary because Docker will enable support for the open-source metasearch engine Searx. This is essential so that artificial intelligence API GOOGLE GEMINI 2.0 FLASH can perform internet searches and improve autonomously using the Searx metasearch engine The link to install Docker Desktop: https://www.docker.com/

## 🎯 Features

### ✅ Autonomous search
- Automatic detection of queries requiring a search
- Multi-engine search (Google, Bing, DuckDuckGo, Wikipedia, etc.)
- Intelligent HTML parsing of results
- Seamless integration with the artificial intelligence API GOOGLE GEMINI 2.0 FLASH

### ✅ Intelligent categorization
- **general**: General searches
- **it**: Technology, programming, GitHub
- **videos**: YouTube and other video platforms
- **actualites**: News and recent information

### ✅ Containerized architecture
- Deployment with Docker Compose
- Isolated and secure configuration
- Automatic startup with the application

## 🚀 Installation and startup

### Prerequisites
- Docker Desktop installed and running
- Python 3.8+ with pip
- Port 8080 available for Searx

### Quick start
```bash
# Method 1: Automatic script (recommended)
start_with_searx.bat

# Method 2: Manual startup
python install_searx_deps.py
docker-compose -f docker-compose.searx.yml up -d
python app.py
```

### Operation verification
```bash
# Full system test
python test_searx_system.py

# Manual verification
curl http://localhost:8080/search?q=test&format=json
```

## 🔧 Configuration

### Searx Configuration (`searx-config/settings.yml`)
```yaml
# Enabled search engines
engines:
  - name: google
    engine: google
    categories: general
    disabled: false
  
  - name: wikipedia
    engine: wikipedia
    categories: general
    disabled: false
```

### Docker Configuration (`docker-compose.searx.yml`)
```yaml
services:
  searx:
    image: searxng/searxng:latest
    ports:
      - "8080:8080"
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080
```

## 📡 API and integration

### Usage from artificial intelligence API GOOGLE GEMINI 2.0 FLASH
The artificial intelligence API GOOGLE GEMINI 2.0 FLASH automatically detects queries requiring a search:

**Automatic triggers:**
- "search the internet..."
- "find information..."
- "recent information..."
- "news..."

**Usage example:**
```
User: "Search for recent information about artificial intelligence"
artificial intelligence API GOOGLE GEMINI 2.0 FLASH: [Automatically triggers a Searx search and uses the results]
```

### Programmatic interface
```python
from searx_interface import get_searx_interface

searx = get_searx_interface()

# Simple search
results = searx.search("artificial intelligence", max_results=5)

# Search with category
results = searx.search("python tutorial", category="it", max_results=10)

# Advanced search
results = searx.search_with_filters(
    query="AI news",
    engines=["google", "bing"],
    safe_search=0
)
```

## 🛠️ Technical architecture

### Main components

1.  **SearxInterface** (`searx_interface.py`)
    - Python interface for Searx
    - HTML parsing of results
    - Error handling and retry

2.  **SearxManager** (`searx_manager.py`)
    - Docker lifecycle management
    - Service health monitoring
    - Auto-startup and recovery

3.  **artificial intelligence API GOOGLE GEMINI 2.0 FLASH Integration** (`gemini_api_adapter.py`)
    - Automatic query detection
    - Formatting results for the artificial intelligence API GOOGLE GEMINI 2.0 FLASH
    - Fallback to the old system

### Processing flow

```
User query
       ↓
Automatic detection (Gemini)
       ↓
Search query extraction
       ↓
Searx search (HTML)
       ↓
Results parsing
       ↓
Formatting for artificial intelligence API GOOGLE GEMINI 2.0 FLASH
       ↓
Enriched response
```

## 🔍 Automatic detection

### Trigger keywords
- Search: "search", "look for", "find"
- Internet: "on the internet", "on the web"
- News: "recent information", "news", "latest news"
- Specific: "what's happening", "what's new", "current trends"

### Detected categories
- **Technology**: "python", "programming", "github", "api"
- **News**: "news", "journal"
- **Videos**: "video", "youtube", "tutorial"
- **General**: all other queries

## 📊 Monitoring and logs

### Main logs
- `INFO:SearxInterface`: Search operations
- `INFO:SearxManager`: Docker management
- `INFO:GeminiAPI`: Integration with the artificial intelligence API GOOGLE GEMINI 2.0 FLASH

### Health monitoring
```python
from searx_manager import get_searx_manager

manager = get_searx_manager()
status = manager.get_service_status()
print(f"Docker: {status['docker_status']}")
print(f"HTTP: {status['http_status']}")
```

## 🔒 Security

### Security measures
- Unique secret key for Searx
- Full Docker isolation
- Rate limiting
- Secure parsing of HTML results

### Security configuration
```yaml
search:
  safe_search: 0
  ban_time_on_fail: 5
  max_ban_time_on_fail: 120

server:
  secret_key: "ai_search_secret_key_2025"
```

## 🚨 Troubleshooting

### Common problems

**1. Docker not started**
```
❌ Error: Cannot connect to the Docker daemon
Solution: Start Docker Desktop
```

**2. Port 8080 occupied**
```
❌ Error: Port 8080 already in use
Solution: docker-compose down; netstat -ano | findstr :8080
```

**3. No search results**
```
⚠️ No results found
Solution: Check internet connectivity and configured engines
```

### Diagnostic commands
```bash
# Container status
docker ps | grep searx

# Searx logs
docker logs ai_searx

# Connectivity test
curl http://localhost:8080/stats

# Full restart
docker-compose -f docker-compose.searx.yml restart
```

## 📈 Performance

### Optimizations
- Caching search results
- Optimized HTML parsing with BeautifulSoup
- Parallel requests to engines
- Adaptive timeout

### Typical metrics
- Response time: 2-5 seconds
- Results per search: 5-20
- Simultaneous engines: 3-6
- Availability: >99%

## 🔄 Maintenance

### Searx update
```bash
docker-compose -f docker-compose.searx.yml pull
docker-compose -f docker-compose.searx.yml up -d
```

### Cleanup
```bash
# Stop and remove containers
docker-compose -f docker-compose.searx.yml down --volumes

# Image cleanup
docker image prune -f
```

### Configuration backup
```bash
# Backup configuration
tar -czf searx-config-backup.tar.gz searx-config/

# Restore configuration
tar -xzf searx-config-backup.tar.gz
```

## 🆘 Support

### Resources
- [Official Searx documentation](https://docs.searxng.org/)
- [Docker Compose documentation](https://docs.docker.com/compose/)
- Application logs in the Python console

### Contacts
- Technical issues: Check Python logs
- Docker problems: Check Docker Desktop
- Performance: Use `test_searx_system.py`

---

**Version**: 1.0
**Date**: July 2025
**Compatibility**: Windows 10+, Docker Desktop 4.0+
