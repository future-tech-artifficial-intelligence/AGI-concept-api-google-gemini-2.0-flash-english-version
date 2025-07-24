# 🔍 Système de Recherche Searx pour l'IA

## Vue d'ensemble

Le système de recherche Searx intégré permet à l'API Gemini d'effectuer des recherches autonomes en temps réel en utilisant le parsing HTML au lieu du web scraping traditionnel. Cette approche offre des performances supérieures et une meilleure fiabilité.

## 🎯 Fonctionnalités

### ✅ Recherche autonome
- Détection automatique des requêtes nécessitant une recherche
- Recherche multi-moteurs (Google, Bing, DuckDuckGo, Wikipedia, etc.)
- Parsing HTML intelligent des résultats
- Intégration transparente avec l'API Gemini

### ✅ Catégorisation intelligente
- **general**: Recherches générales
- **it**: Technologie, programmation, GitHub
- **videos**: YouTube et autres plateformes vidéo
- **actualites**: Actualités et informations récentes

### ✅ Architecture conteneurisée
- Déploiement avec Docker Compose
- Configuration isolée et sécurisée
- Démarrage automatique avec l'application

## 🚀 Installation et démarrage

### Prérequis
- Docker Desktop installé et démarré
- Python 3.8+ avec pip
- Port 8080 disponible pour Searx

### Démarrage rapide
```bash
# Méthode 1: Script automatique (recommandé)
start_with_searx.bat

# Méthode 2: Démarrage manuel
python install_searx_deps.py
docker-compose -f docker-compose.searx.yml up -d
python app.py
```

### Vérification du fonctionnement
```bash
# Test complet du système
python test_searx_system.py

# Vérification manuelle
curl http://localhost:8080/search?q=test&format=json
```

## 🔧 Configuration

### Configuration Searx (`searx-config/settings.yml`)
```yaml
# Moteurs de recherche activés
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

### Configuration Docker (`docker-compose.searx.yml`)
```yaml
services:
  searx:
    image: searxng/searxng:latest
    ports:
      - "8080:8080"
    environment:
      - SEARXNG_BASE_URL=http://localhost:8080
```

## 📡 API et intégration

### Utilisation depuis l'IA
L'IA détecte automatiquement les requêtes nécessitant une recherche :

**Déclencheurs automatiques :**
- "recherche sur internet..."
- "trouve des informations..."
- "informations récentes..."
- "actualités..."

**Exemple d'utilisation :**
```
Utilisateur: "Recherche des informations récentes sur l'intelligence artificielle"
IA: [Déclenche automatiquement une recherche Searx et utilise les résultats]
```

### Interface programmatique
```python
from searx_interface import get_searx_interface

searx = get_searx_interface()

# Recherche simple
results = searx.search("intelligence artificielle", max_results=5)

# Recherche avec catégorie
results = searx.search("tutoriel python", category="it", max_results=10)

# Recherche avancée
results = searx.search_with_filters(
    query="actualités IA",
    engines=["google", "bing"],
    safe_search=0
)
```

## 🛠️ Architecture technique

### Composants principaux

1. **SearxInterface** (`searx_interface.py`)
   - Interface Python pour Searx
   - Parsing HTML des résultats
   - Gestion des erreurs et retry

2. **SearxManager** (`searx_manager.py`)
   - Gestion du cycle de vie Docker
   - Surveillance de la santé du service
   - Auto-démarrage et récupération

3. **Intégration Gemini** (`gemini_api_adapter.py`)
   - Détection automatique des requêtes
   - Formatage des résultats pour l'IA
   - Fallback vers l'ancien système

### Flux de traitement

```
Requête utilisateur
       ↓
Détection automatique (Gemini)
       ↓
Extraction de la requête de recherche
       ↓
Recherche Searx (HTML)
       ↓
Parsing des résultats
       ↓
Formatage pour l'IA
       ↓
Réponse enrichie
```

## 🔍 Détection automatique

### Mots-clés déclencheurs
- Recherche: "recherche", "cherche", "trouve"
- Internet: "sur internet", "sur le web"
- Actualités: "informations récentes", "actualités", "dernières nouvelles"
- Spécifique: "que se passe-t-il", "quoi de neuf", "tendances actuelles"

### Catégories détectées
- **Technologie**: "python", "programmation", "github", "api"
- **Actualités**: "actualité", "news", "journal"
- **Vidéos**: "vidéo", "youtube", "tutoriel"
- **Général**: toutes les autres requêtes

## 📊 Surveillance et logs

### Logs principaux
- `INFO:SearxInterface`: Opérations de recherche
- `INFO:SearxManager`: Gestion Docker
- `INFO:GeminiAPI`: Intégration avec l'IA

### Monitoring de santé
```python
from searx_manager import get_searx_manager

manager = get_searx_manager()
status = manager.get_service_status()
print(f"Docker: {status['docker_status']}")
print(f"HTTP: {status['http_status']}")
```

## 🔒 Sécurité

### Mesures de sécurité
- Clé secrète unique pour Searx
- Isolation Docker complète
- Limitation du rate limiting
- Parsing sécurisé des résultats HTML

### Configuration de sécurité
```yaml
search:
  safe_search: 0
  ban_time_on_fail: 5
  max_ban_time_on_fail: 120

server:
  secret_key: "ai_search_secret_key_2025"
```

## 🚨 Dépannage

### Problèmes courants

**1. Docker non démarré**
```
❌ Erreur: Cannot connect to the Docker daemon
Solution: Démarrer Docker Desktop
```

**2. Port 8080 occupé**
```
❌ Erreur: Port 8080 already in use
Solution: docker-compose down; netstat -ano | findstr :8080
```

**3. Pas de résultats de recherche**
```
⚠️ Aucun résultat trouvé
Solution: Vérifier la connectivité internet et les moteurs configurés
```

### Commands de diagnostic
```bash
# Status des conteneurs
docker ps | grep searx

# Logs Searx
docker logs ai_searx

# Test de connectivité
curl http://localhost:8080/stats

# Redémarrage complet
docker-compose -f docker-compose.searx.yml restart
```

## 📈 Performance

### Optimisations
- Cache des résultats de recherche
- Parsing HTML optimisé avec BeautifulSoup
- Requêtes parallèles aux moteurs
- Timeout adaptatif

### Métriques typiques
- Temps de réponse: 2-5 secondes
- Résultats par recherche: 5-20
- Moteurs simultanés: 3-6
- Disponibilité: >99%

## 🔄 Maintenance

### Mise à jour Searx
```bash
docker-compose -f docker-compose.searx.yml pull
docker-compose -f docker-compose.searx.yml up -d
```

### Nettoyage
```bash
# Arrêt et suppression des conteneurs
docker-compose -f docker-compose.searx.yml down --volumes

# Nettoyage des images
docker image prune -f
```

### Sauvegarde de la configuration
```bash
# Sauvegarder la configuration
tar -czf searx-config-backup.tar.gz searx-config/

# Restaurer la configuration
tar -xzf searx-config-backup.tar.gz
```

## 🆘 Support

### Ressources
- [Documentation Searx officielle](https://docs.searxng.org/)
- [Docker Compose documentation](https://docs.docker.com/compose/)
- Logs applicatifs dans la console Python

### Contacts
- Issues techniques: Vérifier les logs Python
- Problèmes Docker: Vérifier Docker Desktop
- Performance: Utiliser `test_searx_system.py`

---

**Version**: 1.0  
**Date**: Juillet 2025  
**Compatibilité**: Windows 10+, Docker Desktop 4.0+
