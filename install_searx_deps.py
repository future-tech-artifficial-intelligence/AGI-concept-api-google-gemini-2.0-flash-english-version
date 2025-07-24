#!/usr/bin/env python3
"""
Installation des dépendances pour le système Searx
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('searx_deps_installer')

def install_package(package_name):
    """Installe un package Python avec pip"""
    try:
        logger.info(f"Installation de {package_name}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package_name
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"✅ {package_name} installé avec succès")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Erreur lors de l'installation de {package_name}: {e.stderr}")
        return False

def main():
    """Installe toutes les dépendances nécessaires pour Searx"""
    logger.info("🔧 Installation des dépendances pour le système Searx")
    
    dependencies = [
        'beautifulsoup4',  # Pour le parsing HTML
        'lxml',           # Parser XML/HTML plus rapide
        'requests',       # Client HTTP (normalement déjà installé)
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for package in dependencies:
        if install_package(package):
            success_count += 1
    
    logger.info(f"📊 Installation terminée: {success_count}/{total_count} packages installés")
    
    if success_count == total_count:
        logger.info("✅ Toutes les dépendances ont été installées avec succès")
        return True
    else:
        logger.warning("⚠️ Certaines dépendances n'ont pas pu être installées")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
