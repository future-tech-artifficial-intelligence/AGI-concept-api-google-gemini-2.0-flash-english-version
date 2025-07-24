# 🛠️ Guide de dépannage Searx

## Problème: Docker non accessible

### Symptômes
```
unable to get image 'searxng/searxng:latest': error during connect: 
Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.49/...": 
open //./pipe/dockerDesktopLinuxEngine: Le fichier spéciﬁé est introuvable.
```

### Solutions

#### 1. Vérifier Docker Desktop
```cmd
# Vérifier si Docker Desktop est installé
check_docker.bat

# Si non installé, télécharger depuis:
# https://www.docker.com/products/docker-desktop/
```

#### 2. Démarrer Docker Desktop manuellement
1. Chercher "Docker Desktop" dans le menu Démarrer
2. Cliquer droit → "Exécuter en tant qu'administrateur"
3. Attendre le démarrage complet (icône Docker dans la barre des tâches)

#### 3. Redémarrer les services Docker
```cmd
# Dans un terminal administrateur
net stop com.docker.service
net start com.docker.service
```

#### 4. Vérification complète
```cmd
# Test complet de Docker
docker --version
docker info
docker ps
```

## Problème: Port 8080 occupé

### Symptômes
```
Error response from daemon: driver failed programming external connectivity 
on endpoint ai_searx: Bind for 0.0.0.0:8080 failed: port is already allocated
```

### Solutions

#### 1. Identifier le processus
```cmd
netstat -ano | findstr :8080
```

#### 2. Arrêter le processus
```cmd
# Remplacer PID par le numéro trouvé
taskkill /PID <PID> /F
```

#### 3. Changer le port (optionnel)
Modifier dans `docker-compose.searx.yml`:
```yaml
ports:
  - "8081:8080"  # Utiliser le port 8081 à la place
```

## Problème: Pas de résultats de recherche

### Causes possibles
1. Connectivité Internet
2. Moteurs de recherche bloqués
3. Configuration Searx incorrecte

### Solutions

#### 1. Test de connectivité
```cmd
ping google.com
curl http://localhost:8080/stats
```

#### 2. Vérifier les logs
```cmd
docker logs ai_searx
```

#### 3. Redémarrer Searx
```cmd
docker-compose -f docker-compose.searx.yml restart
```

## Problème: Application Python ne trouve pas Searx

### Solutions

#### 1. Vérifier l'intégration
```python
python -c "from searx_interface import get_searx_interface; print('OK')"
```

#### 2. Test manuel
```python
python searx_interface.py
```

#### 3. Vérifier les logs Python
Chercher dans la console les messages:
- `✅ Interface Searx intégrée`
- `⚠️ Interface Searx non disponible`

## 🆘 Commandes de diagnostic rapide

```cmd
# Status complet
docker ps -a | findstr searx
curl http://localhost:8080/ 
python test_searx_system.py

# Nettoyage complet
docker-compose -f docker-compose.searx.yml down --volumes
docker system prune -f

# Redémarrage complet
docker-compose -f docker-compose.searx.yml up -d --force-recreate
```

## 📞 Support

Si aucune solution ne fonctionne:

1. **Vérifier les prérequis:**
   - Windows 10/11 avec WSL2 activé
   - Docker Desktop 4.0+ installé
   - 4GB RAM libre minimum

2. **Collecter les informations:**
   ```cmd
   docker --version
   docker info > docker-info.txt
   docker logs ai_searx > searx-logs.txt
   ```

3. **Solutions alternatives:**
   - Utiliser le système de web scraping existant
   - Désactiver temporairement Searx dans `gemini_api_adapter.py`
