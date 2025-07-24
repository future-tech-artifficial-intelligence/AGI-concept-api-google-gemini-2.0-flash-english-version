#!/usr/bin/env python3
"""
Installation des dépendances pour le système de capture visuelle Searx
"""

import subprocess
import sys
import logging
import os
import platform
import requests
import zipfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('searx_visual_deps_installer')

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

def check_chrome_installed():
    """Vérifie si Chrome est installé"""
    try:
        if platform.system() == "Windows":
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            ]
            return any(os.path.exists(path) for path in chrome_paths)
        else:
            result = subprocess.run(['which', 'google-chrome'], capture_output=True, text=True)
            return result.returncode == 0
    except:
        return False

def check_edge_installed():
    """Vérifie si Edge est installé"""
    try:
        if platform.system() == "Windows":
            edge_paths = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
            ]
            return any(os.path.exists(path) for path in edge_paths)
        else:
            result = subprocess.run(['which', 'microsoft-edge'], capture_output=True, text=True)
            return result.returncode == 0
    except:
        return False

def download_chromedriver():
    """Télécharge ChromeDriver automatiquement"""
    try:
        logger.info("Téléchargement de ChromeDriver...")
        
        # Déterminer la version de Chrome
        if platform.system() == "Windows":
            chrome_path = None
            for path in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]:
                if os.path.exists(path):
                    chrome_path = path
                    break
            
            if not chrome_path:
                logger.warning("Chrome non trouvé pour déterminer la version")
                return False
        
        # Utiliser webdriver-manager pour une installation automatique
        logger.info("Utilisation de webdriver-manager pour l'installation automatique")
        return True
        
    except Exception as e:
        logger.error(f"Erreur téléchargement ChromeDriver: {e}")
        return False

def main():
    """Installe toutes les dépendances nécessaires pour la capture visuelle"""
    logger.info("🔧 Installation des dépendances pour la capture visuelle Searx")
    
    # Dépendances Python
    dependencies = [
        'selenium',              # WebDriver pour automation navigateur
        'webdriver-manager',     # Gestion automatique des drivers
        'Pillow',               # Traitement d'images
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    # Installation des packages Python
    for package in dependencies:
        if install_package(package):
            success_count += 1
    
    logger.info(f"📊 Installation Python terminée: {success_count}/{total_count} packages installés")
    
    # Vérification des navigateurs
    logger.info("🌐 Vérification des navigateurs disponibles...")
    
    chrome_available = check_chrome_installed()
    edge_available = check_edge_installed()
    
    if chrome_available:
        logger.info("✅ Google Chrome détecté")
    else:
        logger.warning("⚠️ Google Chrome non détecté")
    
    if edge_available:
        logger.info("✅ Microsoft Edge détecté")
    else:
        logger.warning("⚠️ Microsoft Edge non détecté")
    
    if not chrome_available and not edge_available:
        logger.error("❌ Aucun navigateur compatible détecté")
        logger.error("Veuillez installer Google Chrome ou Microsoft Edge")
        return False
    
    # Test d'importation
    logger.info("🧪 Test des imports...")
    
    try:
        import selenium
        from selenium import webdriver
        logger.info("✅ Selenium importé avec succès")
        
        import PIL
        logger.info("✅ Pillow importé avec succès")
        
        from webdriver_manager.chrome import ChromeDriverManager
        logger.info("✅ WebDriver Manager importé avec succès")
        
    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {e}")
        return False
    
    # Test rapide WebDriver
    logger.info("🔧 Test rapide du WebDriver...")
    
    try:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from webdriver_manager.chrome import ChromeDriverManager
        
        if chrome_available:
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=options
            )
            driver.get("about:blank")
            driver.quit()
            logger.info("✅ Test WebDriver Chrome réussi")
            
    except Exception as e:
        logger.warning(f"⚠️ Test WebDriver Chrome échoué: {e}")
        
        # Essayer Edge comme alternative
        try:
            from selenium.webdriver.edge.options import Options as EdgeOptions
            from webdriver_manager.microsoft import EdgeChromiumDriverManager
            
            if edge_available:
                options = EdgeOptions()
                options.add_argument('--headless')
                
                driver = webdriver.Edge(
                    service=webdriver.edge.service.Service(EdgeChromiumDriverManager().install()),
                    options=options
                )
                driver.get("about:blank")
                driver.quit()
                logger.info("✅ Test WebDriver Edge réussi")
                
        except Exception as e2:
            logger.error(f"❌ Test WebDriver Edge aussi échoué: {e2}")
            logger.error("La capture visuelle pourrait ne pas fonctionner")
            return False
    
    logger.info("🎉 Installation terminée avec succès !")
    logger.info("Le système de capture visuelle Searx est prêt à l'emploi")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("\n💡 Solutions possibles:")
        logger.error("1. Installer Google Chrome: https://www.google.com/chrome/")
        logger.error("2. Installer Microsoft Edge: https://www.microsoft.com/edge")
        logger.error("3. Vérifier les permissions d'installation")
        logger.error("4. Réessayer avec des droits administrateur")
    
    sys.exit(0 if success else 1)
