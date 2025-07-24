#!/usr/bin/env python3
"""
Système d'installation automatique des dépendances manquantes
Analyse les modules requis et les installe automatiquement
Supporte maintenant Termux/Android
"""

import subprocess
import sys
import importlib
import logging
import os
from typing import List, Dict, Tuple

# Import du détecteur de plateforme si disponible
try:
    from platform_detector import get_platform_detector
    PLATFORM_DETECTION_AVAILABLE = True
except ImportError:
    PLATFORM_DETECTION_AVAILABLE = False

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AutoInstaller')

class AutoInstaller:
    """Gestionnaire d'installation automatique des dépendances"""
    
    def __init__(self):
        # Détecter la plateforme si disponible
        if PLATFORM_DETECTION_AVAILABLE:
            self.detector = get_platform_detector()
            self.is_termux = self.detector.platform_info.get('is_termux', False)
            self.platform_type = self.detector.platform_info.get('platform_type', 'unknown')
        else:
            self.is_termux = self._detect_termux_fallback()
            self.platform_type = 'termux' if self.is_termux else 'standard'
        
        # Modules adaptés selon la plateforme
        if self.is_termux:
            self.required_modules = self._get_termux_modules()
        else:
            self.required_modules = self._get_standard_modules()
        
        self.optional_modules = {
            # Modules optionnels qui peuvent améliorer les performances
            'memory_profiler': 'memory-profiler>=0.61.0',
            'checksumdir': 'checksumdir>=1.2.0',
            'readability': 'readability-lxml>=0.8.1',
        }
    
    def _detect_termux_fallback(self) -> bool:
        """Détection de Termux en fallback si platform_detector n'est pas disponible"""
        termux_indicators = [
            'TERMUX_VERSION' in os.environ,
            'PREFIX' in os.environ and '/data/data/com.termux' in os.environ.get('PREFIX', ''),
            os.path.exists('/data/data/com.termux')
        ]
        return any(termux_indicators)
    
    def _get_termux_modules(self) -> Dict[str, str]:
        """Modules compatibles avec Termux"""
        return {
            # Modules de base - compatibles Termux
            'requests': 'requests>=2.31.0',
            'flask': 'flask>=2.3.0',
            'flask_compress': 'flask-compress>=1.14',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
            'pillow': 'pillow>=10.0.0',
            'beautifulsoup4': 'beautifulsoup4>=4.12.0',
            'lxml': 'lxml>=4.9.0',
            'aiohttp': 'aiohttp>=3.8.0',
            'networkx': 'networkx>=3.0',
            
            # Alternatives pour Termux
            'cv2': 'opencv-python-headless>=4.8.0',  # Au lieu d'opencv-python
            'matplotlib': 'matplotlib>=3.7.0',
            'scipy': 'scipy>=1.10.0',
            'textblob': 'textblob>=0.17.1',
            'nltk': 'nltk>=3.8.1',
            'psutil': 'psutil>=5.9.6',
            'tenacity': 'tenacity>=8.2.3',
            'py7zr': 'py7zr>=0.20.8',
            'xlsxwriter': 'xlsxwriter>=3.1.9',
            'feedparser': 'feedparser>=6.0.10',
        }
    
    def _get_standard_modules(self) -> Dict[str, str]:
        """Modules pour plateformes standard (non-Termux)"""
        return {
            # Modules pour le web scraping
            'aiohttp': 'aiohttp>=3.9.5',
            'networkx': 'networkx>=3.2.1',
            
            # Modules pour l'analyse d'images et OCR
            'cv2': 'opencv-python>=4.10.0.84',
            'pytesseract': 'pytesseract>=0.3.10',
            
            # Modules pour l'analyse de données
            'matplotlib': 'matplotlib>=3.8.2',
            'scipy': 'scipy>=1.11.4',
            
            # Modules pour le traitement audio
            'librosa': 'librosa>=0.10.2',
            'soundfile': 'soundfile>=0.12.1',
            
            # Modules pour le scraping web
            'selenium': 'selenium>=4.15.2',
            'beautifulsoup4': 'beautifulsoup4>=4.12.3',
            'lxml': 'lxml>=5.2.2',
            
            # Modules pour les bases de données
            'pymongo': 'pymongo>=4.6.0',
            'redis': 'redis>=5.0.1',
            
            # Modules pour l'analyse de texte
            'textblob': 'textblob>=0.17.1',
            'nltk': 'nltk>=3.8.1',
            
            # Modules système
            'psutil': 'psutil>=5.9.6',
            'tenacity': 'tenacity>=8.2.3',
            
            # Modules de compression
            'py7zr': 'py7zr>=0.20.8',
            
            # Formats de fichiers
            'xlsxwriter': 'xlsxwriter>=3.1.9',
            'feedparser': 'feedparser>=6.0.10',
        }
    
    def check_module_availability(self, module_name: str) -> bool:
        """Vérifie si un module est disponible"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    def install_package(self, package_spec: str) -> bool:
        """Installe un package via pip avec optimisations pour Termux"""
        try:
            logger.info(f"📦 Installation de {package_spec}...")
            
            # Arguments pip adaptés à la plateforme
            pip_args = [sys.executable, '-m', 'pip', 'install']
            
            if self.is_termux:
                # Optimisations spécifiques à Termux
                pip_args.extend(['--no-cache-dir', '--timeout', '600'])
            
            pip_args.append(package_spec)
            
            timeout = 600 if self.is_termux else 300  # Plus de temps pour Termux
            
            result = subprocess.run(
                pip_args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {package_spec} installé avec succès")
                return True
            else:
                logger.error(f"❌ Erreur lors de l'installation de {package_spec}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout lors de l'installation de {package_spec}")
            return False
        except Exception as e:
            logger.error(f"💥 Exception lors de l'installation de {package_spec}: {str(e)}")
            return False
    
    def upgrade_pip(self) -> bool:
        """Met à jour pip vers la dernière version"""
        try:
            logger.info("🔄 Mise à jour de pip...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"⚠️ Impossible de mettre à jour pip: {str(e)}")
            return False
    
    def install_missing_modules(self, include_optional: bool = False) -> Dict[str, bool]:
        """Installe tous les modules manquants"""
        results = {}
        
        # Mettre à jour pip d'abord
        self.upgrade_pip()
        
        # Modules requis
        missing_required = []
        for module_name, package_spec in self.required_modules.items():
            if not self.check_module_availability(module_name):
                missing_required.append((module_name, package_spec))
        
        if missing_required:
            logger.info(f"🔍 {len(missing_required)} modules requis manquants détectés")
            
            for module_name, package_spec in missing_required:
                success = self.install_package(package_spec)
                results[module_name] = success
        else:
            logger.info("✅ Tous les modules requis sont déjà installés")
        
        # Modules optionnels si demandé
        if include_optional:
            missing_optional = []
            for module_name, package_spec in self.optional_modules.items():
                if not self.check_module_availability(module_name):
                    missing_optional.append((module_name, package_spec))
            
            if missing_optional:
                logger.info(f"🔍 {len(missing_optional)} modules optionnels manquants détectés")
                
                for module_name, package_spec in missing_optional:
                    success = self.install_package(package_spec)
                    results[f"{module_name} (optionnel)"] = success
        
        return results
    
    def check_and_install_all(self, include_optional: bool = False) -> bool:
        """Vérifie et installe tous les modules manquants"""
        try:
            logger.info("🚀 Démarrage de la vérification des dépendances...")
            
            results = self.install_missing_modules(include_optional)
            
            if results:
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                logger.info(f"📊 Résultats: {successful}/{total} modules installés avec succès")
                
                if successful == total:
                    logger.info("🎉 Toutes les dépendances ont été installées avec succès!")
                    return True
                else:
                    failed_modules = [name for name, success in results.items() if not success]
                    logger.warning(f"⚠️ Modules non installés: {', '.join(failed_modules)}")
                    return False
            else:
                logger.info("✅ Aucune installation nécessaire")
                return True
                
        except Exception as e:
            logger.error(f"💥 Erreur lors de la vérification des dépendances: {str(e)}")
            return False
    
    def generate_missing_modules_report(self) -> str:
        """Génère un rapport des modules manquants"""
        missing = []
        
        for module_name in self.required_modules:
            if not self.check_module_availability(module_name):
                missing.append(f"❌ {module_name} (requis)")
        
        for module_name in self.optional_modules:
            if not self.check_module_availability(module_name):
                missing.append(f"⚠️ {module_name} (optionnel)")
        
        if missing:
            return f"Modules manquants:\n" + "\n".join(missing)
        else:
            return "✅ Tous les modules sont installés"

def run_auto_installer():
    """Point d'entrée principal pour l'installation automatique"""
    installer = AutoInstaller()
    
    # Afficher un rapport des modules manquants
    print("\n" + "="*50)
    print("🔧 VÉRIFICATION DES DÉPENDANCES")
    print("="*50)
    print(installer.generate_missing_modules_report())
    print("="*50 + "\n")
    
    # Installer les modules manquants
    success = installer.check_and_install_all(include_optional=False)
    
    if success:
        print("\n🎉 Installation automatique terminée avec succès!")
        print("🚀 Vous pouvez maintenant utiliser toutes les fonctionnalités du projet.")
    else:
        print("\n⚠️ Certains modules n'ont pas pu être installés automatiquement.")
        print("📝 Vous pouvez les installer manuellement avec: pip install <nom_du_module>")
    
    return success

if __name__ == "__main__":
    run_auto_installer()
