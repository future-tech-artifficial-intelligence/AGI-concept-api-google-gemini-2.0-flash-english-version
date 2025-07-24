#!/data/data/com.termux/files/usr/bin/python
# -*- coding: utf-8 -*-
"""
Lanceur principal pour Termux/Android
Configure et lance l'application AI sur Android
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Configuration des logs pour Termux
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('TermuxLauncher')

def check_termux_environment():
    """Vérifie que nous sommes bien dans Termux"""
    if 'TERMUX_VERSION' not in os.environ:
        logger.error("❌ Ce script doit être exécuté dans Termux")
        return False
    
    logger.info(f"🤖 Termux version: {os.environ.get('TERMUX_VERSION', 'Unknown')}")
    return True

def setup_termux_environment():
    """Configure l'environnement Termux initial"""
    logger.info("🔧 Configuration de l'environnement Termux...")
    
    # Variables d'environnement importantes pour Termux
    env_vars = {
        'PYTHONUNBUFFERED': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'FLASK_ENV': 'production',
        'MALLOC_ARENA_MAX': '2',  # Limiter l'utilisation mémoire
    }
    
    for var, value in env_vars.items():
        if var not in os.environ:
            os.environ[var] = value
            logger.info(f"✅ Variable {var} définie: {value}")

def install_termux_dependencies():
    """Installe les dépendances Termux manquantes"""
    logger.info("📦 Vérification des dépendances Termux...")
    
    # Packages système requis
    required_packages = [
        'python',
        'python-pip', 
        'git',
        'curl',
        'wget',
        'clang',
        'pkg-config',
        'libjpeg-turbo',
        'libpng'
    ]
    
    for package in required_packages:
        try:
            result = subprocess.run(['pkg', 'list-installed', package], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.info(f"📦 Installation de {package}...")
                subprocess.run(['pkg', 'install', '-y', package], check=True)
            else:
                logger.info(f"✅ {package} déjà installé")
        except subprocess.CalledProcessError:
            logger.warning(f"⚠️  Impossible d'installer {package}")

def create_storage_directories():
    """Crée les répertoires de stockage nécessaires"""
    logger.info("📁 Création des répertoires de stockage...")
    
    # Demander l'accès au stockage si nécessaire
    storage_path = Path.home() / 'storage'
    if not storage_path.exists():
        logger.info("🔑 Configuration de l'accès au stockage...")
        try:
            subprocess.run(['termux-setup-storage'], check=True, timeout=30)
            logger.info("✅ Accès au stockage configuré")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️  Timeout lors de la configuration du stockage")
        except subprocess.CalledProcessError:
            logger.warning("⚠️  Impossible de configurer l'accès au stockage")
    
    # Créer les répertoires de données
    data_dirs = [
        Path.home() / 'storage' / 'shared' / 'AI_Data',
        Path.home() / 'storage' / 'shared' / 'AI_Data' / 'conversations',
        Path.home() / 'storage' / 'shared' / 'AI_Data' / 'uploads',
        Path.home() / 'storage' / 'shared' / 'AI_Data' / 'cache',
        Path.home() / 'storage' / 'shared' / 'AI_Data' / 'logs'
    ]
    
    for directory in data_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Répertoire créé: {directory}")

def check_python_modules():
    """Vérifie et installe les modules Python requis"""
    logger.info("🐍 Vérification des modules Python...")
    
    try:
        # Importer l'auto-installer
        from auto_installer import AutoInstaller
        installer = AutoInstaller()
        
        # Générer le rapport des modules manquants
        missing_report = installer.generate_missing_modules_report()
        
        if "❌" in missing_report:
            logger.info("🔧 Installation automatique des modules manquants...")
            installer.install_missing_modules()
        else:
            logger.info("✅ Tous les modules sont disponibles")
            
    except ImportError:
        logger.warning("⚠️  Auto-installer non disponible, installation manuelle requise")

def get_network_info():
    """Affiche les informations réseau pour Termux"""
    try:
        import socket
        hostname = socket.gethostname()
        
        # Obtenir l'adresse IP locale
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        logger.info(f"🌐 Hostname: {hostname}")
        logger.info(f"🌐 IP locale: {local_ip}")
        
        return local_ip
    except Exception as e:
        logger.warning(f"⚠️  Impossible d'obtenir les infos réseau: {e}")
        return "localhost"

def main():
    """Point d'entrée principal"""
    print("""
🤖 LANCEUR TERMUX - INTELLIGENCE ARTIFICIELLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    # 1. Vérifier l'environnement Termux
    if not check_termux_environment():
        sys.exit(1)
    
    # 2. Configurer l'environnement
    setup_termux_environment()
    
    # 3. Installer les dépendances système
    install_termux_dependencies()
    
    # 4. Créer les répertoires de stockage
    create_storage_directories()
    
    # 5. Vérifier les modules Python
    check_python_modules()
    
    # 6. Obtenir les informations réseau
    local_ip = get_network_info()
    
    # 7. Lancer l'application principale
    logger.info("🚀 Lancement de l'application...")
    
    try:
        # Importer et lancer l'application
        import app
        
        # Informations de lancement
        port = 5000
        print(f"""
✅ APPLICATION LANCÉE AVEC SUCCÈS!

🌐 Accès local: http://localhost:{port}
🌐 Accès réseau: http://{local_ip}:{port}

💡 CONSEILS TERMUX:
   • Utilisez 'Ctrl+C' pour arrêter l'application
   • L'application fonctionne en arrière-plan
   • Accédez via un navigateur sur votre appareil
   
📱 Pour accéder depuis un autre appareil:
   • Assurez-vous d'être sur le même réseau WiFi
   • Utilisez l'adresse: http://{local_ip}:{port}
        """)
        
    except ImportError as e:
        logger.error(f"❌ Erreur d'import de l'application: {e}")
        logger.error("Vérifiez que tous les fichiers sont présents")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur lors du lancement: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
