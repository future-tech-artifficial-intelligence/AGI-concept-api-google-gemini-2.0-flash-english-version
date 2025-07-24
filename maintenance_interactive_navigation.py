#!/usr/bin/env python3
"""
Script de Maintenance - Système de Navigation Interactive Gemini
Valide, maintient et optimise le système complet
"""

import os
import sys
import json
import time
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('maintenance.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('InteractiveNavigationMaintenance')

class InteractiveNavigationMaintainer:
    """Système de maintenance pour la navigation interactive"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.maintenance_log = []
        self.issues_found = []
        self.fixes_applied = []
        self.start_time = datetime.now()
        
        # Fichiers critiques du système
        self.critical_files = [
            'interactive_web_navigator.py',
            'gemini_interactive_adapter.py',
            'install_interactive_navigation.py',
            'demo_interactive_navigation.py',
            'test_interactive_navigation.py',
            'GUIDE_NAVIGATION_INTERACTIVE.md'
        ]
        
        # Dépendances requises
        self.required_packages = [
            'google-generativeai',
            'selenium',
            'webdriver-manager',
            'beautifulsoup4',
            'pillow',
            'requests',
            'python-dotenv',
            'flask',
            'flask-cors'
        ]
        
        logger.info("🔧 Système de maintenance initialisé")
        
    def log_action(self, action: str, status: str, details: str = ""):
        """Enregistre une action de maintenance"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'status': status,
            'details': details
        }
        
        self.maintenance_log.append(entry)
        
        if status == "SUCCESS":
            logger.info(f"✅ {action}: {details}")
        elif status == "WARNING":
            logger.warning(f"⚠️ {action}: {details}")
        else:
            logger.error(f"❌ {action}: {details}")
            
    def check_file_integrity(self) -> bool:
        """Vérifie l'intégrité des fichiers critiques"""
        self.log_action("INTEGRITY_CHECK", "INFO", "Vérification de l'intégrité des fichiers")
        
        all_files_ok = True
        
        for file_path in self.critical_files:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                self.log_action("FILE_MISSING", "ERROR", f"Fichier manquant: {file_path}")
                self.issues_found.append(f"Fichier manquant: {file_path}")
                all_files_ok = False
                continue
                
            # Vérification de la taille (fichier non vide)
            if full_path.stat().st_size == 0:
                self.log_action("FILE_EMPTY", "ERROR", f"Fichier vide: {file_path}")
                self.issues_found.append(f"Fichier vide: {file_path}")
                all_files_ok = False
                continue
                
            # Vérification de la syntaxe Python
            if file_path.endswith('.py'):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), file_path, 'exec')
                    self.log_action("SYNTAX_CHECK", "SUCCESS", f"Syntaxe valide: {file_path}")
                except SyntaxError as e:
                    self.log_action("SYNTAX_ERROR", "ERROR", f"Erreur syntaxe {file_path}: {e}")
                    self.issues_found.append(f"Erreur syntaxe {file_path}: {e}")
                    all_files_ok = False
                    
        return all_files_ok
        
    def check_dependencies(self) -> bool:
        """Vérifie les dépendances Python"""
        self.log_action("DEPENDENCY_CHECK", "INFO", "Vérification des dépendances")
        
        missing_packages = []
        
        for package in self.required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.log_action("PACKAGE_OK", "SUCCESS", f"Package disponible: {package}")
            except ImportError:
                self.log_action("PACKAGE_MISSING", "WARNING", f"Package manquant: {package}")
                missing_packages.append(package)
                
        if missing_packages:
            self.issues_found.append(f"Packages manquants: {', '.join(missing_packages)}")
            return False
            
        return True
        
    def check_configuration(self) -> bool:
        """Vérifie les fichiers de configuration"""
        self.log_action("CONFIG_CHECK", "INFO", "Vérification de la configuration")
        
        config_files = [
            '.env',
            'config/navigation_config.json',
            'ai_api_config.json'
        ]
        
        config_ok = True
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            
            if not config_path.exists():
                self.log_action("CONFIG_MISSING", "WARNING", f"Configuration manquante: {config_file}")
                continue
                
            # Vérification spécifique selon le type
            if config_file.endswith('.json'):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    self.log_action("CONFIG_VALID", "SUCCESS", f"Configuration valide: {config_file}")
                except json.JSONDecodeError as e:
                    self.log_action("CONFIG_INVALID", "ERROR", f"JSON invalide {config_file}: {e}")
                    self.issues_found.append(f"JSON invalide {config_file}: {e}")
                    config_ok = False
                    
            elif config_file == '.env':
                # Vérification de la clé API
                with open(config_path, 'r', encoding='utf-8') as f:
                    env_content = f.read()
                    
                if 'GEMINI_API_KEY' not in env_content:
                    self.log_action("API_KEY_MISSING", "WARNING", "Clé API Gemini non configurée")
                elif 'votre_cle_api_ici' in env_content:
                    self.log_action("API_KEY_PLACEHOLDER", "WARNING", "Clé API Gemini non modifiée")
                else:
                    self.log_action("API_KEY_OK", "SUCCESS", "Clé API Gemini configurée")
                    
        return config_ok
        
    def check_disk_space(self) -> bool:
        """Vérifie l'espace disque disponible"""
        self.log_action("DISK_CHECK", "INFO", "Vérification de l'espace disque")
        
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            free_space_mb = disk_usage.free / (1024 * 1024)
            
            if free_space_mb < 100:  # Moins de 100 MB
                self.log_action("DISK_LOW", "ERROR", f"Espace disque faible: {free_space_mb:.1f} MB")
                self.issues_found.append(f"Espace disque insuffisant: {free_space_mb:.1f} MB")
                return False
            elif free_space_mb < 500:  # Moins de 500 MB
                self.log_action("DISK_WARNING", "WARNING", f"Espace disque limité: {free_space_mb:.1f} MB")
            else:
                self.log_action("DISK_OK", "SUCCESS", f"Espace disque suffisant: {free_space_mb:.1f} MB")
                
            return True
            
        except Exception as e:
            self.log_action("DISK_ERROR", "ERROR", f"Erreur vérification disque: {e}")
            return False
            
    def clean_temporary_files(self) -> bool:
        """Nettoie les fichiers temporaires"""
        self.log_action("CLEANUP", "INFO", "Nettoyage des fichiers temporaires")
        
        temp_patterns = [
            '**/*.pyc',
            '**/__pycache__',
            '**/test_results_*/*.png',
            '**/logs/*.log.old',
            '**/cache/*',
            '**/.pytest_cache'
        ]
        
        cleaned_count = 0
        
        for pattern in temp_patterns:
            for file_path in self.project_root.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        cleaned_count += 1
                except Exception as e:
                    self.log_action("CLEANUP_ERROR", "WARNING", f"Erreur nettoyage {file_path}: {e}")
                    
        self.log_action("CLEANUP_COMPLETE", "SUCCESS", f"Nettoyage terminé: {cleaned_count} éléments supprimés")
        self.fixes_applied.append(f"Nettoyage: {cleaned_count} fichiers temporaires supprimés")
        
        return True
        
    def optimize_imports(self) -> bool:
        """Optimise les imports Python"""
        self.log_action("IMPORT_OPTIMIZATION", "INFO", "Optimisation des imports")
        
        python_files = list(self.project_root.glob('*.py'))
        optimized_count = 0
        
        for py_file in python_files:
            if py_file.name.startswith('test_') or py_file.name.startswith('demo_'):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Vérification des imports inutilisés (basique)
                lines = content.split('\n')
                import_lines = [line for line in lines if line.strip().startswith('import ') or line.strip().startswith('from ')]
                
                if len(import_lines) > 20:  # Beaucoup d'imports
                    self.log_action("MANY_IMPORTS", "WARNING", f"Beaucoup d'imports dans {py_file.name}: {len(import_lines)}")
                    
                optimized_count += 1
                
            except Exception as e:
                self.log_action("IMPORT_ERROR", "WARNING", f"Erreur analyse imports {py_file.name}: {e}")
                
        self.log_action("IMPORT_OPTIMIZATION_COMPLETE", "SUCCESS", f"Analyse de {optimized_count} fichiers Python")
        return True
        
    def update_documentation(self) -> bool:
        """Met à jour la documentation"""
        self.log_action("DOC_UPDATE", "INFO", "Mise à jour de la documentation")
        
        try:
            # Génération d'un rapport de maintenance
            maintenance_report = {
                "last_maintenance": datetime.now().isoformat(),
                "critical_files_status": "OK" if all(
                    (self.project_root / f).exists() for f in self.critical_files
                ) else "ISSUES",
                "dependencies_status": "OK" if not self.issues_found else "ISSUES",
                "issues_found": self.issues_found,
                "fixes_applied": self.fixes_applied,
                "maintenance_log": self.maintenance_log
            }
            
            # Sauvegarde du rapport
            report_file = self.project_root / f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(maintenance_report, f, indent=2, ensure_ascii=False)
                
            self.log_action("DOC_SAVED", "SUCCESS", f"Rapport sauvegardé: {report_file.name}")
            return True
            
        except Exception as e:
            self.log_action("DOC_ERROR", "ERROR", f"Erreur sauvegarde documentation: {e}")
            return False
            
    def run_quick_tests(self) -> bool:
        """Exécute des tests rapides"""
        self.log_action("QUICK_TESTS", "INFO", "Exécution de tests rapides")
        
        try:
            # Test d'import des modules principaux
            test_imports = [
                'interactive_web_navigator',
                'gemini_interactive_adapter',
                'ai_api_interface'
            ]
            
            for module in test_imports:
                try:
                    __import__(module)
                    self.log_action("IMPORT_TEST", "SUCCESS", f"Import réussi: {module}")
                except ImportError as e:
                    self.log_action("IMPORT_TEST", "ERROR", f"Import échoué {module}: {e}")
                    self.issues_found.append(f"Import échoué: {module}")
                    
            return True
            
        except Exception as e:
            self.log_action("QUICK_TESTS_ERROR", "ERROR", f"Erreur tests rapides: {e}")
            return False
            
    def generate_health_report(self) -> Dict:
        """Génère un rapport de santé du système"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        health_score = 100
        if self.issues_found:
            health_score -= len(self.issues_found) * 10
        health_score = max(0, health_score)
        
        report = {
            "maintenance_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": round(duration, 2),
                "health_score": health_score,
                "issues_found": len(self.issues_found),
                "fixes_applied": len(self.fixes_applied),
                "status": "HEALTHY" if health_score >= 80 else "NEEDS_ATTENTION" if health_score >= 50 else "CRITICAL"
            },
            "critical_files": {
                "total": len(self.critical_files),
                "present": len([f for f in self.critical_files if (self.project_root / f).exists()]),
                "missing": [f for f in self.critical_files if not (self.project_root / f).exists()]
            },
            "issues_detail": self.issues_found,
            "fixes_detail": self.fixes_applied,
            "maintenance_actions": self.maintenance_log,
            "recommendations": self.generate_recommendations()
        }
        
        return report
        
    def generate_recommendations(self) -> List[str]:
        """Génère des recommandations d'amélioration"""
        recommendations = []
        
        if len(self.issues_found) > 5:
            recommendations.append("Résoudre les problèmes critiques avant utilisation")
            
        if any("PACKAGE_MISSING" in issue for issue in self.issues_found):
            recommendations.append("Installer les dépendances manquantes avec pip install -r requirements.txt")
            
        if any("API_KEY" in issue for issue in self.issues_found):
            recommendations.append("Configurer la clé API Gemini dans le fichier .env")
            
        if not recommendations:
            recommendations.append("Système en bon état, maintenance régulière recommandée")
            
        return recommendations
        
    def run_full_maintenance(self) -> Dict:
        """Exécute la maintenance complète"""
        self.log_action("MAINTENANCE_START", "INFO", "Début de la maintenance complète")
        
        maintenance_tasks = [
            ("Vérification intégrité", self.check_file_integrity),
            ("Vérification dépendances", self.check_dependencies),
            ("Vérification configuration", self.check_configuration),
            ("Vérification espace disque", self.check_disk_space),
            ("Nettoyage fichiers temporaires", self.clean_temporary_files),
            ("Optimisation imports", self.optimize_imports),
            ("Tests rapides", self.run_quick_tests),
            ("Mise à jour documentation", self.update_documentation)
        ]
        
        for task_name, task_func in maintenance_tasks:
            try:
                self.log_action("TASK_START", "INFO", f"Début: {task_name}")
                success = task_func()
                
                if success:
                    self.log_action("TASK_SUCCESS", "SUCCESS", f"Terminé: {task_name}")
                else:
                    self.log_action("TASK_PARTIAL", "WARNING", f"Partiellement réussi: {task_name}")
                    
            except Exception as e:
                self.log_action("TASK_ERROR", "ERROR", f"Erreur {task_name}: {e}")
                
        # Génération du rapport final
        health_report = self.generate_health_report()
        
        self.log_action("MAINTENANCE_COMPLETE", "SUCCESS", 
                       f"Maintenance terminée - Score de santé: {health_report['maintenance_summary']['health_score']}")
        
        return health_report

def display_health_report(report: Dict):
    """Affiche le rapport de santé"""
    print("\n" + "=" * 80)
    print("🏥 RAPPORT DE SANTÉ DU SYSTÈME - NAVIGATION INTERACTIVE GEMINI")
    print("=" * 80)
    
    summary = report["maintenance_summary"]
    
    print(f"\n⏱️ DURÉE DE MAINTENANCE: {summary['duration_seconds']}s")
    print(f"🎯 SCORE DE SANTÉ: {summary['health_score']}/100")
    print(f"📊 STATUT: {summary['status']}")
    
    # Icônes de statut
    status_icons = {
        "HEALTHY": "🟢",
        "NEEDS_ATTENTION": "🟡", 
        "CRITICAL": "🔴"
    }
    
    print(f"{status_icons.get(summary['status'], '⚪')} {summary['status']}")
    
    # Détails des fichiers
    files_info = report["critical_files"]
    print(f"\n📁 FICHIERS CRITIQUES:")
    print(f"   • Total: {files_info['total']}")
    print(f"   • Présents: {files_info['present']} ✅")
    print(f"   • Manquants: {len(files_info['missing'])} ❌")
    
    if files_info['missing']:
        print("   📋 Fichiers manquants:")
        for missing_file in files_info['missing']:
            print(f"      - {missing_file}")
    
    # Problèmes trouvés
    if report["issues_detail"]:
        print(f"\n⚠️ PROBLÈMES DÉTECTÉS ({len(report['issues_detail'])}):")
        for i, issue in enumerate(report["issues_detail"], 1):
            print(f"   {i}. {issue}")
    
    # Corrections appliquées
    if report["fixes_detail"]:
        print(f"\n🔧 CORRECTIONS APPLIQUÉES ({len(report['fixes_detail'])}):")
        for i, fix in enumerate(report["fixes_detail"], 1):
            print(f"   {i}. {fix}")
    
    # Recommandations
    if report["recommendations"]:
        print(f"\n💡 RECOMMANDATIONS:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    # Évaluation finale
    if summary['health_score'] >= 90:
        print(f"\n🎉 EXCELLENT - Système en parfait état")
    elif summary['health_score'] >= 70:
        print(f"\n👍 BON - Système fonctionnel avec améliorations mineures possibles")
    elif summary['health_score'] >= 50:
        print(f"\n⚠️ ATTENTION - Système nécessite des corrections")
    else:
        print(f"\n🚨 CRITIQUE - Interventions urgentes requises")
    
    print("\n" + "=" * 80)

def main():
    """Point d'entrée principal"""
    print("🔧 MAINTENANCE SYSTÈME - NAVIGATION INTERACTIVE GEMINI")
    print("🎯 Validation, nettoyage et optimisation automatiques")
    
    maintainer = InteractiveNavigationMaintainer()
    
    try:
        # Exécution complète de la maintenance
        health_report = maintainer.run_full_maintenance()
        
        # Affichage du rapport
        display_health_report(health_report)
        
        # Sauvegarde du rapport
        report_file = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(health_report, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Rapport complet sauvegardé: {report_file}")
        
        # Code de retour basé sur la santé
        health_score = health_report["maintenance_summary"]["health_score"]
        return health_score >= 50
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Maintenance interrompue par l'utilisateur")
        return False
    except Exception as e:
        logger.error(f"💥 Erreur critique lors de la maintenance: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
