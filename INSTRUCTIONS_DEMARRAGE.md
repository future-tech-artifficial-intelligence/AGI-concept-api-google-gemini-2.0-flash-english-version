# 🚀 INSTRUCTIONS DE DÉMARRAGE

## Étape 1: Démarrer Docker Desktop

1. **Ouvrir le menu Démarrer** et chercher "Docker Desktop"
2. **Cliquer droit** sur Docker Desktop → **"Exécuter en tant qu'administrateur"**
3. **Attendre** que Docker Desktop se lance complètement (icône 🐳 dans la barre des tâches)

## Étape 2: Vérifier Docker

Exécutez cette commande pour vérifier:
```cmd
check_docker.bat
```

## Étape 3: Démarrage automatique du système Searx

Une fois Docker prêt, lancez:
```cmd
start_with_searx.bat
```

## Étape 4: Test manuel si nécessaire

Si le démarrage automatique échoue:
```cmd
# Vérifier Docker
python searx_manager.py

# Test complet
python test_searx_system.py

# Démarrage manuel de l'app
python app.py
```

## 🎯 Test de fonctionnement

Une fois l'application démarrée (http://localhost:4004), testez avec:
- "Recherche des informations récentes sur l'intelligence artificielle"
- "Trouve des actualités sur Python"
- "Cherche des tutoriels de programmation"

L'IA devrait automatiquement utiliser Searx pour ces requêtes !

---
**💡 Note**: Si Docker Desktop n'est pas installé, téléchargez-le depuis:
https://www.docker.com/products/docker-desktop/
