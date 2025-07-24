# 🤖 Guide d'Installation Termux - Intelligence Artificielle

## 📱 Installation sur Android avec Termux

### Prérequis

1. **Installer Termux** depuis F-Droid (recommandé) ou Google Play Store
2. **Espace disque**: Au moins 2GB libres
3. **RAM**: 2GB minimum, 4GB recommandé
4. **Connexion Internet**: Requise pour l'installation

### 🔧 Installation Automatique

```bash
# 1. Mettre à jour Termux
pkg update && pkg upgrade

# 2. Installer Python et Git
pkg install python python-pip git curl

# 3. Cloner le projet
git clone [URL_DU_PROJET] ai-project
cd ai-project

# 4. Lancer l'installation automatique
python termux_launcher.py
```

### 🛠️ Installation Manuelle

Si l'installation automatique échoue:

```bash
# 1. Packages système requis
pkg install python python-pip git curl wget clang pkg-config libjpeg-turbo libpng zlib openssl

# 2. Configurer l'accès au stockage
termux-setup-storage

# 3. Mettre à jour pip
python -m pip install --upgrade pip wheel

# 4. Installer les dépendances Python
python -m pip install -r requirements-termux.txt --no-cache-dir

# 5. Lancer l'application
python app.py
```

### 📁 Structure de Fichiers Termux

```
/data/data/com.termux/files/home/
├── ai-project/                 # Dossier du projet
│   ├── app.py                 # Application principale
│   ├── termux_launcher.py     # Lanceur Termux
│   ├── requirements-termux.txt # Dépendances optimisées
│   └── ...
└── storage/
    └── shared/
        └── AI_Data/           # Données de l'application
            ├── conversations/ # Historique des conversations
            ├── uploads/       # Fichiers uploadés
            ├── cache/         # Cache temporaire
            └── logs/          # Logs de l'application
```

### 🚀 Lancement de l'Application

```bash
# Option 1: Lanceur Termux (recommandé)
python termux_launcher.py

# Option 2: Lancement direct
python app.py

# Option 3: En arrière-plan
nohup python app.py &
```

### 🌐 Accès à l'Application

Une fois lancée, l'application sera accessible:

- **Local**: `http://localhost:5000`
- **Réseau**: `http://[IP_LOCAL]:5000`

Pour connaître votre IP locale:
```bash
ip route get 1 | awk '{print $7}'
```

### 🔧 Optimisations Termux

#### Gestion de la Mémoire
```bash
# Limiter l'utilisation mémoire
export MALLOC_ARENA_MAX=2
export PYTHONDONTWRITEBYTECODE=1
```

#### Performance
```bash
# Utiliser le cache pip local
python -m pip install --user package_name

# Installation sans cache (économise l'espace)
python -m pip install --no-cache-dir package_name
```

### ⚡ Scripts Utiles

#### Script de Démarrage Rapide
```bash
#!/data/data/com.termux/files/usr/bin/bash
# start-ai.sh

cd ~/ai-project
export PYTHONUNBUFFERED=1
export MALLOC_ARENA_MAX=2
python termux_launcher.py
```

#### Script d'Arrêt
```bash
#!/data/data/com.termux/files/usr/bin/bash
# stop-ai.sh

pkill -f "python.*app.py"
echo "Application arrêtée"
```

### 🐛 Dépannage

#### Problèmes Courants

1. **Erreur de permission**:
   ```bash
   termux-setup-storage
   chmod +x termux_launcher.py
   ```

2. **Module non trouvé**:
   ```bash
   python -m pip install module_name --no-cache-dir
   ```

3. **Mémoire insuffisante**:
   ```bash
   # Fermer les autres applications
   # Redémarrer Termux
   export MALLOC_ARENA_MAX=1
   ```

4. **Compilation longue**:
   ```bash
   # Certains modules prennent du temps sur ARM
   # Patience requise pour scipy, matplotlib
   ```

#### Vérification de l'Installation
```bash
# Vérifier Python
python --version

# Vérifier les modules
python -c "import flask, requests, numpy; print('Modules OK')"

# Vérifier l'espace disque
df -h $PREFIX
```

### 📱 Conseils d'Utilisation

1. **Batterie**: L'application peut consommer de la batterie en continu
2. **Réseau**: Utiliser le WiFi pour de meilleures performances
3. **Stockage**: Nettoyer régulièrement le cache et les logs
4. **Mise à jour**: Mettre à jour Termux et les packages régulièrement

#### Automatisation du Lancement
Créer un widget Termux ou utiliser Tasker pour lancer automatiquement:
```bash
am start -n com.termux/com.termux.HomeActivity -e com.termux.ipc.arguments "~/ai-project/start-ai.sh"
```

### 🔒 Sécurité

- L'application est accessible uniquement sur le réseau local par défaut
- Pour un accès externe, utiliser un tunnel sécurisé (ngrok, etc.)
- Éviter d'exposer directement sur Internet sans authentification

### 📊 Monitoring

```bash
# Surveiller l'utilisation des ressources
top -p $(pgrep python)

# Voir les logs en temps réel
tail -f storage/shared/AI_Data/logs/app.log
```

### 🆘 Support

En cas de problème:
1. Vérifier les logs: `storage/shared/AI_Data/logs/`
2. Redémarrer Termux complètement
3. Réinstaller les dépendances problématiques
4. Consulter la documentation Termux officielle

---

**Note**: L'installation sur Termux peut prendre 30-60 minutes selon la connexion Internet et la puissance de l'appareil Android.
