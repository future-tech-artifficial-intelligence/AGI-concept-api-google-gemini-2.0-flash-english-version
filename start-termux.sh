#!/data/data/com.termux/files/usr/bin/bash
# Script de démarrage rapide pour Termux

# Configuration des couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "🤖 INTELLIGENCE ARTIFICIELLE - TERMUX LAUNCHER"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

# Vérifier si nous sommes dans Termux
if [ -z "$TERMUX_VERSION" ]; then
    echo -e "${RED}❌ Ce script doit être exécuté dans Termux${NC}"
    exit 1
fi

echo -e "${BLUE}📱 Termux Version: $TERMUX_VERSION${NC}"

# Fonction pour vérifier si un package est installé
check_package() {
    if pkg list-installed 2>/dev/null | grep -q "^$1/"; then
        echo -e "${GREEN}✅ $1 installé${NC}"
        return 0
    else
        echo -e "${YELLOW}📦 Installation de $1...${NC}"
        pkg install -y "$1"
        return $?
    fi
}

# Vérifier et installer les dépendances système
echo -e "${BLUE}🔧 Vérification des dépendances système...${NC}"

REQUIRED_PACKAGES="python python-pip git curl"
for package in $REQUIRED_PACKAGES; do
    check_package "$package"
done

# Configurer l'accès au stockage si nécessaire
if [ ! -d "$HOME/storage" ]; then
    echo -e "${YELLOW}🔑 Configuration de l'accès au stockage...${NC}"
    termux-setup-storage
fi

# Créer les répertoires de données
DATA_DIR="$HOME/storage/shared/AI_Data"
mkdir -p "$DATA_DIR"/{conversations,uploads,cache,logs}
echo -e "${GREEN}📁 Répertoires de données créés dans $DATA_DIR${NC}"

# Variables d'environnement pour optimiser Termux
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MALLOC_ARENA_MAX=2

# Vérifier si l'application existe
if [ ! -f "app.py" ]; then
    echo -e "${RED}❌ app.py non trouvé dans le répertoire courant${NC}"
    echo -e "${YELLOW}💡 Assurez-vous d'être dans le répertoire du projet${NC}"
    exit 1
fi

# Installer les dépendances Python si nécessaire
if [ -f "requirements-termux.txt" ]; then
    echo -e "${BLUE}🐍 Installation des dépendances Python...${NC}"
    python -m pip install --upgrade pip wheel
    python -m pip install -r requirements-termux.txt --no-cache-dir
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Dépendances Python installées${NC}"
    else
        echo -e "${YELLOW}⚠️  Certaines dépendances ont échoué, mais continuons...${NC}"
    fi
fi

# Obtenir l'adresse IP locale
get_local_ip() {
    python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(('8.8.8.8', 80))
    ip = s.getsockname()[0]
    s.close()
    print(ip)
except:
    print('localhost')
"
}

LOCAL_IP=$(get_local_ip)

echo -e "${GREEN}"
echo "✅ CONFIGURATION TERMINÉE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${NC}"

echo -e "${CYAN}🚀 Lancement de l'application...${NC}"

# Lancer l'application
if [ -f "termux_launcher.py" ]; then
    python termux_launcher.py
else
    python app.py
fi

# Informations de connexion
echo -e "${GREEN}"
echo "🌐 APPLICATION LANCÉE !"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📱 Accès local:  http://localhost:5000"
echo "🌐 Accès réseau: http://$LOCAL_IP:5000"
echo ""
echo "💡 CONSEILS:"
echo "   • Utilisez Ctrl+C pour arrêter l'application"
echo "   • Accédez via un navigateur web"
echo "   • Pour l'accès externe, utilisez l'IP réseau"
echo -e "${NC}"
