# 🚀 Guide de démarrage rapide - Système Searx

## Démarrage en 5 minutes

### 📋 Prérequis
- [ ] Docker Desktop installé et démarré
- [ ] Python 3.8+ disponible
- [ ] Port 8080 libre

### 🎯 Étapes rapides

1. **Démarrage automatique** (recommandé)
   ```cmd
   start_with_searx.bat
   ```

2. **Vérification**
   - ✅ Interface IA: http://localhost:4004
   - ✅ Interface Searx: http://localhost:8080

### 🧪 Test rapide

Tapez dans l'interface IA :
```
"Recherche des informations sur Python"
```

L'IA devrait automatiquement utiliser Searx pour la recherche !

## ⚡ Commandes utiles

```cmd
# Test du système
python test_searx_system.py

# Redémarrage Searx
docker-compose -f docker-compose.searx.yml restart

# Arrêt complet
docker-compose -f docker-compose.searx.yml down
```

## 🆘 Problèmes fréquents

| Problème | Solution |
|----------|----------|
| Docker non démarré | Lancer Docker Desktop |
| Port 8080 occupé | `netstat -ano \| findstr :8080` |
| Pas de résultats | Vérifier internet + logs |

---
**💡 Astuce**: Utilisez `start_with_searx.bat` pour un démarrage entièrement automatisé !
