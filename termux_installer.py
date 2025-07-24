#!/usr/bin/env python3
"""
Installateur adaptatif pour Termux et autres plateformes
Gère l'installation intelligente des dépendances selon l'environnement
"""

import subprocess
import sys
import logging
import os
from typing import List, Dict, Tuple, Optional
from platform_detector import get_platform_detector

logger = logging.getLogger('TermuxInstaller')

class TermuxInstaller:
    """Installateur intelligent pour Termux et Android"""
    
    def __init__(self):
        self.detector = get_platform_detector()
        self.platform_info = self.detector.platform_info
        self.config = self.detector.config
        
        # Modules adaptés pour Termux
        self.termux_compatible_modules = {
            # Modules de base - toujours compatibles
            'requests': 'requests>=2.31.0',
            'flask': 'flask>=2.3.0',
            'flask_compress': 'flask-compress>=1.14',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
            'pillow': 'pillow>=10.0.0',
            'beautifulsoup4': 'beautifulsoup4>=4.12.0',
            'lxml': 'lxml>=4.9.0',
            
            # Modules avec alternatives pour Termux
            'aiohttp': 'aiohttp>=3.8.0',
            'networkx': 'networkx>=3.0',
            'matplotlib': 'matplotlib>=3.7.0',  # Peut nécessiter des packages système
            'scipy': 'scipy>=1.10.0',  # Compilation longue sur Termux
            
            # Modules problématiques sur Termux (alternatives)
            'opencv-python': None,  # Utiliser opencv-python-headless
            'librosa': None,  # Audio limité sur Android
            'soundfile': None,  # Audio limité sur Android
            'selenium': None,  # Pas de navigateur complet sur Termux
            'pytesseract': None,  # Tesseract complexe à installer
        }
        
        # Alternatives pour les modules problématiques
        self.termux_alternatives = {
            'cv2': 'opencv-python-headless>=4.8.0',
            'selenium': 'requests-html>=0.10.0',  # Alternative légère
            'librosa': 'scipy>=1.10.0',  # Pour certaines fonctions audio basiques
            'pytesseract': 'easyocr>=1.7.0',  # OCR alternatif
        }
        
        # Packages système requis pour Termux
        self.termux_system_packages = [
            'python',
            'python-pip',
            'git',
            'curl',
            'wget',
            'clang',
            'pkg-config',
            'libjpeg-turbo',
            'libpng',
            'zlib',
            'openssl'
        ]
    
    def setup_termux_environment(self) -> bool:
        """Configure l'environnement Termux initial"""
        if not self.platform_info['is_termux']:
            logger.info("Non-Termux environment detected, skipping Termux setup")
            return True
        
        logger.info("🤖 Configuration de l'environnement Termux...")
        
        # 1. Mettre à jour les packages
        if not self._update_termux_packages():
            return False
        
        # 2. Installer les packages système requis
        if not self._install_termux_system_packages():
            return False
        
        # 3. Configurer pip
        if not self._configure_pip_for_termux():
            return False
        
        # 4. Créer les répertoires de données
        self._create_data_directories()
        
        logger.info("✅ Environnement Termux configuré avec succès")
        return True
    
    def _update_termux_packages(self) -> bool:
        """Met à jour les packages Termux"""
        try:
            logger.info("📦 Mise à jour des packages Termux...")
            result = subprocess.run(['pkg', 'update', '-y'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning(f"Avertissement lors de la mise à jour: {result.stderr}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour: {e}")
            return False
    
    def _install_termux_system_packages(self) -> bool:
        """Installe les packages système Termux requis"""
        try:
            logger.info("🔧 Installation des packages système...")
            for package in self.termux_system_packages:
                logger.info(f"Installation de {package}...")
                result = subprocess.run(['pkg', 'install', '-y', package], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    logger.warning(f"Package {package} déjà installé ou erreur: {result.stderr}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'installation des packages système: {e}")
            return False
    
    def _configure_pip_for_termux(self) -> bool:
        """Configure pip pour Termux"""
        try:
            logger.info("🐍 Configuration de pip pour Termux...")
            
            # Mettre à jour pip
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                          capture_output=True, text=True)
            
            # Installer wheel pour éviter les compilations
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'wheel'], 
                          capture_output=True, text=True)
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la configuration de pip: {e}")
            return False
    
    def _create_data_directories(self):
        """Crée les répertoires de données nécessaires"""
        data_path = self.detector.get_data_path()
        directories = [
            data_path,
            os.path.join(data_path, 'conversations'),
            os.path.join(data_path, 'uploads'),
            os.path.join(data_path, 'cache'),
            os.path.join(data_path, 'logs')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"📁 Répertoire créé: {directory}")
    
    def install_python_dependencies(self, modules: List[str]) -> Tuple[List[str], List[str]]:
        """Installe les dépendances Python adaptées à la plateforme"""
        successful = []
        failed = []
        
        for module in modules:
            if self._install_single_module(module):
                successful.append(module)
            else:
                failed.append(module)
                # Essayer une alternative si disponible
                alternative = self._get_module_alternative(module)
                if alternative and self._install_single_module(alternative):
                    successful.append(f"{module} (via {alternative})")
                    failed.remove(module)
        
        return successful, failed
    
    def _install_single_module(self, module: str) -> bool:
        """Installe un module Python unique"""
        try:
            # Déterminer le nom du package à installer
            if module in self.termux_compatible_modules:
                package_spec = self.termux_compatible_modules[module]
                if package_spec is None:
                    logger.info(f"⚠️  Module {module} non compatible avec Termux")
                    return False
            else:
                package_spec = module
            
            logger.info(f"📦 Installation de {package_spec}...")
            
            # Options spéciales pour Termux
            pip_args = [sys.executable, '-m', 'pip', 'install']
            
            if self.platform_info['is_termux']:
                pip_args.extend(['--no-cache-dir', '--timeout', '300'])
            
            pip_args.append(package_spec)
            
            result = subprocess.run(pip_args, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                logger.info(f"✅ {module} installé avec succès")
                return True
            else:
                logger.warning(f"❌ Échec de l'installation de {module}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout lors de l'installation de {module}")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'installation de {module}: {e}")
            return False
    
    def _get_module_alternative(self, module: str) -> Optional[str]:
        """Retourne une alternative pour un module si disponible"""
        return self.termux_alternatives.get(module)
    
    def generate_termux_requirements(self, base_requirements: List[str]) -> str:
        """Génère un fichier requirements.txt adapté à Termux"""
        termux_requirements = []
        
        for req in base_requirements:
            module_name = req.split('>=')[0].split('==')[0].strip()
            
            if module_name in self.termux_compatible_modules:
                package_spec = self.termux_compatible_modules[module_name]
                if package_spec:
                    termux_requirements.append(package_spec)
            elif module_name in self.termux_alternatives:
                termux_requirements.append(self.termux_alternatives[module_name])
            else:
                termux_requirements.append(req)
        
        return '\n'.join(termux_requirements)
    
    def create_termux_launcher(self, app_file: str = 'app.py') -> str:
        """Crée un script de lancement adapté à Termux"""
        launcher_content = f"""#!/data/data/com.termux/files/usr/bin/python
# -*- coding: utf-8 -*-
'''
Lanceur Termux pour l'application AI
Optimisé pour l'environnement Android/Termux
'''

import os
import sys
import logging
from platform_detector import get_platform_detector

# Configuration des logs pour Termux
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('TermuxLauncher')

def main():
    detector = get_platform_detector()
    
    print(detector.get_platform_summary())
    
    if detector.platform_info['is_termux']:
        logger.info("🤖 Lancement en mode Termux...")
        
        # Configuration spécifique à Termux
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['FLASK_ENV'] = 'production'
        
        # Limiter l'utilisation mémoire
        if 'MALLOC_ARENA_MAX' not in os.environ:
            os.environ['MALLOC_ARENA_MAX'] = '2'
    
    # Lancer l'application principale
    try:
        import {app_file.replace('.py', '')}
        logger.info("✅ Application lancée avec succès")
    except ImportError as e:
        logger.error(f"❌ Erreur d'import: {{e}}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur lors du lancement: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        
        return launcher_content
    
    def get_installation_summary(self) -> str:
        """Retourne un résumé de l'installation pour Termux"""
        if not self.platform_info['is_termux']:
            return "❌ Pas un environnement Termux"
        
        summary = """
🤖 GUIDE D'INSTALLATION TERMUX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 1. PRÉPARATION TERMUX:
   pkg update && pkg upgrade
   pkg install python python-pip git

📦 2. PACKAGES SYSTÈME REQUIS:
   """ + ' '.join(self.termux_system_packages) + """

🐍 3. MODULES PYTHON COMPATIBLES:
   ✅ Modules supportés: """ + str(len([m for m in self.termux_compatible_modules.values() if m])) + """
   ⚠️  Modules avec alternatives: """ + str(len(self.termux_alternatives)) + """
   ❌ Modules non compatibles: """ + str(len([m for m in self.termux_compatible_modules.values() if not m])) + """

💡 4. OPTIMISATIONS TERMUX:
   • Utilisation de --no-cache-dir pour pip
   • Timeout étendu pour les installations
   • Limitation de l'utilisation mémoire
   • Chemins adaptés à l'architecture Android

🚀 5. LANCEMENT:
   python termux_launcher.py
"""
        
        return summary

def setup_termux_compatibility():
    """Point d'entrée principal pour configurer la compatibilité Termux"""
    installer = TermuxInstaller()
    
    if installer.platform_info['is_termux']:
        print("🤖 Environnement Termux détecté!")
        print(installer.get_installation_summary())
        
        # Configuration automatique
        if installer.setup_termux_environment():
            print("✅ Configuration Termux terminée avec succès")
            return True
        else:
            print("❌ Erreur lors de la configuration Termux")
            return False
    else:
        print("ℹ️  Environnement non-Termux détecté, aucune configuration spéciale requise")
        return True

if __name__ == "__main__":
    setup_termux_compatibility()
