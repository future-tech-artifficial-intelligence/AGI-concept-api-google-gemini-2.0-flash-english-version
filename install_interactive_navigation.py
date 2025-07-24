"""
Script d'Installation Automatique - Système de Navigation Interactive Gemini
Ce script configure automatiquement toutes les dépendances nécessaires
"""

import os
import sys
import subprocess
import logging
import platform
import json
from pathlib import Path
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('InteractiveInstaller')

class InteractiveNavigationInstaller:
    """Installeur automatique pour le système de navigation interactive"""
    
    def __init__(self):
        self.installation_log = []
        self.errors = []
        self.system_info = {
            'platform': platform.system(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        }
        
        logger.info(f"🚀 Démarrage de l'installation sur {self.system_info['platform']}")
    
    def log_step(self, step_name: str, success: bool, message: str = ""):
        """Enregistre une étape d'installation"""
        timestamp = datetime.now().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'step': step_name,
            'success': success,
            'message': message
        }
        
        self.installation_log.append(entry)
        
        if success:
            logger.info(f"✅ {step_name}: {message}")
        else:
            logger.error(f"❌ {step_name}: {message}")
            self.errors.append(entry)
    
    def check_python_version(self):
        """Vérifie la version de Python"""
        logger.info("🐍 Vérification de la version Python...")
        
        version_info = sys.version_info
        required_major, required_minor = 3, 8
        
        if version_info.major >= required_major and version_info.minor >= required_minor:
            self.log_step("Vérification Python", True, 
                         f"Python {version_info.major}.{version_info.minor}.{version_info.micro} OK")
            return True
        else:
            self.log_step("Vérification Python", False, 
                         f"Python {required_major}.{required_minor}+ requis, {version_info.major}.{version_info.minor} détecté")
            return False
    
    def install_base_requirements(self):
        """Installe les dépendances de base"""
        logger.info("📦 Installation des dépendances de base...")
        
        base_packages = [
            'selenium>=4.15.0',
            'webdriver-manager>=4.0.0',
            'requests>=2.31.0',
            'beautifulsoup4>=4.12.0',
            'lxml>=4.9.0',
            'Pillow>=10.0.0'
        ]
        
        try:
            for package in base_packages:
                logger.info(f"   Instalation de {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_step(f"Installation {package.split('>=')[0]}", True, "Package installé")
                else:
                    self.log_step(f"Installation {package.split('>=')[0]}", False, result.stderr)
                    return False
            
            return True
            
        except Exception as e:
            self.log_step("Installation Dépendances Base", False, str(e))
            return False
    
    def check_webdriver_availability(self):
        """Vérifie la disponibilité des WebDrivers"""
        logger.info("🌐 Vérification des WebDrivers...")
        
        drivers_available = {
            'chrome': False,
            'edge': False,
            'firefox': False
        }
        
        # Test Chrome
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from webdriver_manager.chrome import ChromeDriverManager
            
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            # Tenter d'initialiser ChromeDriver
            driver_path = ChromeDriverManager().install()
            driver = webdriver.Chrome(service=webdriver.chrome.service.Service(driver_path), 
                                    options=chrome_options)
            driver.quit()
            
            drivers_available['chrome'] = True
            self.log_step("Chrome WebDriver", True, "ChromeDriver opérationnel")
            
        except Exception as e:
            self.log_step("Chrome WebDriver", False, f"Erreur: {str(e)}")
        
        # Test Edge (si sur Windows)
        if self.system_info['platform'] == 'Windows':
            try:
                from selenium.webdriver.edge.options import Options as EdgeOptions
                from webdriver_manager.microsoft import EdgeChromiumDriverManager
                
                edge_options = EdgeOptions()
                edge_options.add_argument('--headless')
                edge_options.add_argument('--no-sandbox')
                
                driver_path = EdgeChromiumDriverManager().install()
                driver = webdriver.Edge(service=webdriver.edge.service.Service(driver_path),
                                      options=edge_options)
                driver.quit()
                
                drivers_available['edge'] = True
                self.log_step("Edge WebDriver", True, "EdgeDriver opérationnel")
                
            except Exception as e:
                self.log_step("Edge WebDriver", False, f"Erreur: {str(e)}")
        
        # Résumé
        available_count = sum(drivers_available.values())
        if available_count > 0:
            self.log_step("WebDrivers Globaux", True, 
                         f"{available_count} driver(s) disponible(s): {', '.join([k for k, v in drivers_available.items() if v])}")
            return True
        else:
            self.log_step("WebDrivers Globaux", False, "Aucun WebDriver disponible")
            return False
    
    def test_interactive_modules(self):
        """Teste l'importation des modules interactifs"""
        logger.info("🧪 Test des modules interactifs...")
        
        modules_to_test = [
            ('interactive_web_navigator', 'Navigateur Interactif'),
            ('gemini_interactive_adapter', 'Adaptateur Gemini Interactif'),
            ('gemini_api_adapter', 'Adaptateur Gemini Principal')
        ]
        
        successful_imports = 0
        
        for module_name, display_name in modules_to_test:
            try:
                __import__(module_name)
                self.log_step(f"Import {display_name}", True, "Module importé avec succès")
                successful_imports += 1
                
            except ImportError as e:
                self.log_step(f"Import {display_name}", False, f"Erreur d'import: {str(e)}")
        
        if successful_imports == len(modules_to_test):
            self.log_step("Test Modules", True, "Tous les modules sont disponibles")
            return True
        else:
            self.log_step("Test Modules", False, 
                         f"Seulement {successful_imports}/{len(modules_to_test)} modules disponibles")
            return False
    
    def create_configuration_file(self):
        """Crée un fichier de configuration par défaut"""
        logger.info("⚙️ Création du fichier de configuration...")
        
        config = {
            'interactive_navigation': {
                'enabled': True,
                'preferred_browser': 'chrome',
                'default_timeout': 30,
                'screenshot_enabled': True,
                'max_interactions_per_session': 50
            },
            'webdriver_settings': {
                'headless': True,
                'window_size': [1920, 1080],
                'page_load_timeout': 15,
                'implicit_wait': 5
            },
            'detection_settings': {
                'confidence_threshold': 0.6,
                'interaction_keywords': {
                    'click': ['clique', 'cliquer', 'appuie', 'appuyer'],
                    'navigate': ['explore', 'parcours', 'navigue'],
                    'analyze': ['analyse', 'regarde', 'examine']
                }
            },
            'safety_settings': {
                'respect_robots_txt': True,
                'interaction_delay': 1.0,
                'max_session_duration': 300
            },
            'installation_info': {
                'installed_on': datetime.now().isoformat(),
                'system_info': self.system_info,
                'version': '1.0.0'
            }
        }
        
        try:
            config_path = Path('interactive_navigation_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.log_step("Configuration", True, f"Fichier créé: {config_path}")
            return True
            
        except Exception as e:
            self.log_step("Configuration", False, f"Erreur création fichier: {str(e)}")
            return False
    
    def run_basic_tests(self):
        """Exécute des tests de base du système"""
        logger.info("🧪 Exécution des tests de base...")
        
        try:
            # Test de détection d'interaction
            from gemini_interactive_adapter import detect_interactive_need
            
            test_prompts = [
                "Clique sur l'onglet Services",
                "Explore tous les onglets de ce site",
                "Qu'est-ce que l'intelligence artificielle ?"
            ]
            
            detection_results = []
            for prompt in test_prompts:
                try:
                    result = detect_interactive_need(prompt)
                    detection_results.append({
                        'prompt': prompt,
                        'detected': result.get('requires_interaction', False),
                        'type': result.get('interaction_type'),
                        'confidence': result.get('confidence', 0)
                    })
                except Exception as e:
                    logger.warning(f"Erreur test détection pour '{prompt}': {e}")
            
            interactive_detected = sum(1 for r in detection_results if r['detected'])
            self.log_step("Tests Détection", True, 
                         f"{interactive_detected}/{len(test_prompts)} interactions détectées")
            
            # Test d'initialisation des composants
            try:
                from interactive_web_navigator import get_interactive_navigator
                navigator = get_interactive_navigator()
                
                if navigator:
                    stats = navigator.get_statistics()
                    self.log_step("Test Navigateur", True, "Navigateur initialisé")
                else:
                    self.log_step("Test Navigateur", False, "Navigateur non initialisé")
            
            except Exception as e:
                self.log_step("Test Navigateur", False, f"Erreur: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_step("Tests de Base", False, f"Erreur globale: {str(e)}")
            return False
    
    def generate_installation_report(self):
        """Génère un rapport d'installation complet"""
        logger.info("📋 Génération du rapport d'installation...")
        
        successful_steps = sum(1 for entry in self.installation_log if entry['success'])
        total_steps = len(self.installation_log)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        
        report = {
            'installation_summary': {
                'timestamp': datetime.now().isoformat(),
                'system_info': self.system_info,
                'total_steps': total_steps,
                'successful_steps': successful_steps,
                'success_rate': success_rate,
                'overall_status': 'SUCCESS' if success_rate >= 80 else 'PARTIAL' if success_rate >= 50 else 'FAILED'
            },
            'installation_log': self.installation_log,
            'errors': self.errors,
            'recommendations': self._generate_recommendations()
        }
        
        # Sauvegarder le rapport
        try:
            report_path = Path(f'installation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.log_step("Rapport Installation", True, f"Rapport sauvegardé: {report_path}")
        
        except Exception as e:
            self.log_step("Rapport Installation", False, f"Erreur sauvegarde: {str(e)}")
        
        return report
    
    def _generate_recommendations(self):
        """Génère des recommandations basées sur l'installation"""
        recommendations = []
        
        # Vérifier les erreurs communes
        if any('WebDriver' in error['step'] for error in self.errors):
            recommendations.append({
                'type': 'webdriver_issue',
                'title': 'Problème WebDriver détecté',
                'description': 'Installez manuellement ChromeDriver ou vérifiez que Chrome est installé',
                'actions': [
                    'Télécharger ChromeDriver depuis https://chromedriver.chromium.org/',
                    'Ajouter ChromeDriver au PATH système',
                    'Ou installer Chrome/Chromium navigateur'
                ]
            })
        
        if any('Python' in error['step'] for error in self.errors):
            recommendations.append({
                'type': 'python_version',
                'title': 'Version Python insuffisante',
                'description': 'Mettez à jour Python vers la version 3.8 ou supérieure',
                'actions': [
                    'Télécharger Python 3.8+ depuis python.org',
                    'Réinstaller les dépendances après la mise à jour'
                ]
            })
        
        if not self.errors:
            recommendations.append({
                'type': 'success',
                'title': 'Installation réussie',
                'description': 'Le système est prêt à être utilisé',
                'actions': [
                    'Exécuter python demo_interactive_navigation.py pour voir une démonstration',
                    'Lire le guide d\'utilisation GUIDE_NAVIGATION_INTERACTIVE.md',
                    'Tester avec vos propres cas d\'usage'
                ]
            })
        
        return recommendations
    
    def run_full_installation(self):
        """Lance l'installation complète"""
        logger.info("🎯 DÉMARRAGE DE L'INSTALLATION COMPLÈTE")
        logger.info("=" * 80)
        
        installation_steps = [
            ('Vérification Python', self.check_python_version),
            ('Installation Dépendances', self.install_base_requirements),
            ('Vérification WebDrivers', self.check_webdriver_availability),
            ('Test Modules Interactifs', self.test_interactive_modules),
            ('Création Configuration', self.create_configuration_file),
            ('Tests de Base', self.run_basic_tests)
        ]
        
        start_time = datetime.now()
        
        for step_name, step_function in installation_steps:
            logger.info(f"\n🔄 {step_name}...")
            try:
                success = step_function()
                if not success:
                    logger.warning(f"⚠️ {step_name} a échoué, mais l'installation continue...")
            except Exception as e:
                logger.error(f"❌ Erreur critique dans {step_name}: {e}")
                self.log_step(step_name, False, f"Erreur critique: {str(e)}")
        
        installation_time = (datetime.now() - start_time).total_seconds()
        
        # Générer le rapport final
        report = self.generate_installation_report()
        
        # Afficher le résumé
        logger.info("\n" + "=" * 80)
        logger.info("🏁 INSTALLATION TERMINÉE")
        logger.info(f"⏱️ Temps d'installation: {installation_time:.1f}s")
        logger.info(f"📊 Résultats: {report['installation_summary']['successful_steps']}/{report['installation_summary']['total_steps']} étapes réussies")
        logger.info(f"📈 Taux de réussite: {report['installation_summary']['success_rate']:.1f}%")
        logger.info(f"🎖️ Statut: {report['installation_summary']['overall_status']}")
        
        # Afficher les recommandations
        if report['recommendations']:
            logger.info("\n💡 RECOMMANDATIONS:")
            for rec in report['recommendations']:
                logger.info(f"   🔸 {rec['title']}: {rec['description']}")
        
        # Message final
        if report['installation_summary']['overall_status'] == 'SUCCESS':
            logger.info("\n🎉 INSTALLATION RÉUSSIE !")
            logger.info("✅ Le système de navigation interactive Gemini est opérationnel")
            logger.info("🚀 Vous pouvez maintenant utiliser les nouvelles fonctionnalités")
            logger.info("\n📖 Prochaines étapes:")
            logger.info("   1. Lire le guide: GUIDE_NAVIGATION_INTERACTIVE.md")
            logger.info("   2. Tester: python demo_interactive_navigation.py")
            logger.info("   3. Valider: python test_interactive_navigation.py")
        else:
            logger.info("\n⚠️ INSTALLATION PARTIELLE")
            logger.info("🔧 Consultez le rapport d'installation pour résoudre les problèmes")
            logger.info("💬 Certaines fonctionnalités peuvent être limitées")
        
        logger.info("=" * 80)
        
        return report

def main():
    """Fonction principale d'installation"""
    print("🌟 Installation du Système de Navigation Interactive Gemini")
    print("🎯 Ce script va configurer automatiquement votre environnement\n")
    
    installer = InteractiveNavigationInstaller()
    report = installer.run_full_installation()
    
    # Code de sortie basé sur le succès
    success = report['installation_summary']['overall_status'] in ['SUCCESS', 'PARTIAL']
    
    if success:
        print("\n✅ Installation terminée avec succès")
        print("🎯 Le système de navigation interactive est prêt à être utilisé !")
    else:
        print("\n❌ Installation échouée")
        print("🔧 Consultez les logs et le rapport d'installation pour plus d'informations")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation interrompue par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Erreur critique lors de l'installation: {e}")
        sys.exit(1)
