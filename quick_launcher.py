#!/usr/bin/env python3
"""
Lanceur Rapide - Système de Navigation Interactive Gemini
Interface simple pour démarrer rapidement toutes les fonctionnalités
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

class InteractiveNavigationLauncher:
    """Lanceur pour le système de navigation interactive"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_loaded = False
        self.available_actions = {}
        
    def print_header(self):
        """Affiche l'en-tête du lanceur"""
        print("=" * 80)
        print("🚀 LANCEUR SYSTÈME DE NAVIGATION INTERACTIVE GEMINI")
        print("🎯 Interface de démarrage rapide")
        print("=" * 80)
        
    def print_menu(self):
        """Affiche le menu principal"""
        print("\n📋 ACTIONS DISPONIBLES:")
        print("=" * 50)
        
        actions = {
            "1": ("🏗️  Installation", "Installer le système complet", "install"),
            "2": ("🎭 Démonstration", "Lancer la démonstration interactive", "demo"),
            "3": ("🧪 Tests", "Exécuter les tests automatisés", "test"),
            "4": ("🔧 Maintenance", "Effectuer la maintenance système", "maintenance"),
            "5": ("🌐 Navigation", "Démarrer la navigation interactive", "navigate"),
            "6": ("📊 Rapport", "Générer un rapport de statut", "status"),
            "7": ("🔍 Diagnostic", "Diagnostiquer les problèmes", "diagnose"),
            "8": ("📖 Guide", "Afficher le guide d'utilisation", "guide"),
            "9": ("⚙️  Configuration", "Configurer le système", "config"),
            "0": ("🚪 Quitter", "Fermer le lanceur", "exit")
        }
        
        for key, (icon, desc, action) in actions.items():
            print(f"   {key}. {icon} {desc}")
            self.available_actions[key] = action
            
        print("=" * 50)
        
    def check_prerequisites(self) -> bool:
        """Vérifie les prérequis système"""
        print("\n🔍 Vérification des prérequis...")
        
        # Vérification Python
        if sys.version_info < (3, 8):
            print(f"❌ Python 3.8+ requis (version actuelle: {sys.version_info.major}.{sys.version_info.minor})")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} détecté")
        
        # Vérification des fichiers critiques
        critical_files = [
            'interactive_web_navigator.py',
            'gemini_interactive_adapter.py',
            'install_interactive_navigation.py'
        ]
        
        missing_files = []
        for file_name in critical_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
                
        if missing_files:
            print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
            return False
        print("✅ Fichiers critiques présents")
        
        # Vérification de la configuration
        env_file = self.project_root / '.env'
        if not env_file.exists():
            print("⚠️ Fichier .env manquant - configuration requise")
            return False
            
        # Vérification de la clé API
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()
            
        if 'GEMINI_API_KEY' not in env_content or 'votre_cle_api_ici' in env_content:
            print("⚠️ Clé API Gemini non configurée")
            return False
        print("✅ Configuration de base présente")
        
        return True
        
    def run_installation(self):
        """Lance l'installation"""
        print("\n🏗️ INSTALLATION DU SYSTÈME")
        print("-" * 40)
        
        install_script = self.project_root / 'install_interactive_navigation.py'
        if not install_script.exists():
            print("❌ Script d'installation non trouvé")
            return False
            
        try:
            print("🚀 Lancement de l'installation...")
            result = subprocess.run([sys.executable, str(install_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Installation terminée avec succès")
                return True
            else:
                print(f"❌ Installation échouée (code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"💥 Erreur lors de l'installation: {e}")
            return False
            
    def run_demo(self):
        """Lance la démonstration"""
        print("\n🎭 DÉMONSTRATION INTERACTIVE")
        print("-" * 40)
        
        demo_script = self.project_root / 'demo_interactive_navigation.py'
        if not demo_script.exists():
            print("❌ Script de démonstration non trouvé")
            return False
            
        try:
            print("🎯 Lancement de la démonstration...")
            result = subprocess.run([sys.executable, str(demo_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Démonstration terminée")
                return True
            else:
                print(f"⚠️ Démonstration terminée avec des avertissements")
                return True
                
        except Exception as e:
            print(f"💥 Erreur lors de la démonstration: {e}")
            return False
            
    def run_tests(self):
        """Lance les tests"""
        print("\n🧪 TESTS AUTOMATISÉS")
        print("-" * 40)
        
        test_script = self.project_root / 'test_interactive_navigation.py'
        if not test_script.exists():
            print("❌ Script de test non trouvé")
            return False
            
        try:
            print("🔬 Exécution des tests...")
            result = subprocess.run([sys.executable, str(test_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Tous les tests sont passés")
                return True
            else:
                print(f"⚠️ Certains tests ont échoué (code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"💥 Erreur lors des tests: {e}")
            return False
            
    def run_maintenance(self):
        """Lance la maintenance"""
        print("\n🔧 MAINTENANCE SYSTÈME")
        print("-" * 40)
        
        maintenance_script = self.project_root / 'maintenance_interactive_navigation.py'
        if not maintenance_script.exists():
            print("❌ Script de maintenance non trouvé")
            return False
            
        try:
            print("🛠️ Lancement de la maintenance...")
            result = subprocess.run([sys.executable, str(maintenance_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Maintenance terminée avec succès")
                return True
            else:
                print(f"⚠️ Maintenance terminée avec des avertissements")
                return True
                
        except Exception as e:
            print(f"💥 Erreur lors de la maintenance: {e}")
            return False
            
    def start_navigation(self):
        """Démarre la navigation interactive"""
        print("\n🌐 NAVIGATION INTERACTIVE")
        print("-" * 40)
        
        navigator_script = self.project_root / 'interactive_web_navigator.py'
        if not navigator_script.exists():
            print("❌ Navigateur interactif non trouvé")
            return False
            
        print("🎯 Démarrage de la navigation interactive...")
        print("💡 Utilisez Ctrl+C pour arrêter")
        
        try:
            result = subprocess.run([sys.executable, str(navigator_script)], 
                                  capture_output=False, text=True)
            
            print(f"\n🏁 Navigation terminée (code: {result.returncode})")
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ Navigation interrompue par l'utilisateur")
            return True
        except Exception as e:
            print(f"💥 Erreur lors de la navigation: {e}")
            return False
            
    def generate_status_report(self):
        """Génère un rapport de statut"""
        print("\n📊 RAPPORT DE STATUT")
        print("-" * 40)
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "project_root": str(self.project_root),
            "files_status": {},
            "config_status": {}
        }
        
        # Vérification des fichiers
        critical_files = [
            'interactive_web_navigator.py',
            'gemini_interactive_adapter.py',
            'install_interactive_navigation.py',
            'demo_interactive_navigation.py',
            'test_interactive_navigation.py',
            'maintenance_interactive_navigation.py',
            'GUIDE_NAVIGATION_INTERACTIVE.md'
        ]
        
        for file_name in critical_files:
            file_path = self.project_root / file_name
            status["files_status"][file_name] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
            
        # Vérification de la configuration
        config_files = ['.env', 'ai_api_config.json']
        for config_file in config_files:
            config_path = self.project_root / config_file
            status["config_status"][config_file] = {
                "exists": config_path.exists(),
                "configured": True  # Simplifié pour cet exemple
            }
            
        # Affichage du rapport
        print(f"📅 Timestamp: {status['timestamp']}")
        print(f"🐍 Python: {status['python_version']}")
        print(f"📁 Projet: {status['project_root']}")
        
        print("\n📁 FICHIERS:")
        for file_name, file_info in status["files_status"].items():
            status_icon = "✅" if file_info["exists"] else "❌"
            size_info = f"({file_info['size']} bytes)" if file_info["exists"] else ""
            print(f"   {status_icon} {file_name} {size_info}")
            
        print("\n⚙️ CONFIGURATION:")
        for config_name, config_info in status["config_status"].items():
            status_icon = "✅" if config_info["exists"] else "❌"
            print(f"   {status_icon} {config_name}")
            
        # Sauvegarde du rapport
        report_file = f"status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Rapport sauvegardé: {report_file}")
        
        return True
        
    def run_diagnostics(self):
        """Lance le diagnostic"""
        print("\n🔍 DIAGNOSTIC SYSTÈME")
        print("-" * 40)
        
        issues = []
        
        # Diagnostic des imports
        print("🔬 Test des imports...")
        test_modules = [
            'google.generativeai',
            'selenium',
            'requests',
            'bs4',
            'PIL'
        ]
        
        for module in test_modules:
            try:
                __import__(module)
                print(f"   ✅ {module}")
            except ImportError:
                print(f"   ❌ {module}")
                issues.append(f"Module manquant: {module}")
                
        # Diagnostic de la configuration
        print("\n⚙️ Vérification configuration...")
        env_file = self.project_root / '.env'
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                env_content = f.read()
                
            if 'GEMINI_API_KEY=votre_cle_api_ici' in env_content:
                print("   ⚠️ Clé API non configurée")
                issues.append("Clé API Gemini non configurée")
            else:
                print("   ✅ Configuration API")
        else:
            print("   ❌ Fichier .env manquant")
            issues.append("Fichier de configuration manquant")
            
        # Résumé du diagnostic
        print(f"\n📋 RÉSUMÉ:")
        if not issues:
            print("🎉 Aucun problème détecté !")
        else:
            print(f"⚠️ {len(issues)} problème(s) détecté(s):")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
                
        return len(issues) == 0
        
    def show_guide(self):
        """Affiche le guide d'utilisation"""
        print("\n📖 GUIDE D'UTILISATION")
        print("-" * 40)
        
        guide_file = self.project_root / 'GUIDE_NAVIGATION_INTERACTIVE.md'
        if guide_file.exists():
            print(f"📚 Guide disponible: {guide_file.name}")
            print("💡 Ouvrez ce fichier dans un éditeur markdown pour la documentation complète")
            
            # Affichage des premières lignes
            try:
                with open(guide_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:20]
                    
                print("\n📋 APERÇU:")
                for line in lines:
                    print(f"   {line.rstrip()}")
                    
                if len(lines) >= 20:
                    print("   ... (voir le fichier complet pour plus de détails)")
                    
            except Exception as e:
                print(f"❌ Erreur lecture du guide: {e}")
        else:
            print("❌ Guide non trouvé")
            print("💡 Consultez README.md ou la documentation en ligne")
            
        return True
        
    def configure_system(self):
        """Configuration interactive du système"""
        print("\n⚙️ CONFIGURATION SYSTÈME")
        print("-" * 40)
        
        env_file = self.project_root / '.env'
        
        print("🔑 Configuration de la clé API Gemini")
        current_key = "non configurée"
        
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'GEMINI_API_KEY=' in content:
                    for line in content.split('\n'):
                        if line.startswith('GEMINI_API_KEY='):
                            key_value = line.split('=', 1)[1]
                            if key_value and key_value != 'votre_cle_api_ici':
                                current_key = "configurée"
                            break
                            
        print(f"📊 Statut actuel: {current_key}")
        
        if current_key == "non configurée":
            print("\n💡 Pour configurer votre clé API:")
            print("   1. Obtenez une clé API sur https://makersuite.google.com/app/apikey")
            print("   2. Éditez le fichier .env")
            print("   3. Remplacez 'votre_cle_api_ici' par votre vraie clé")
            print("   4. Relancez ce configurateur")
        else:
            print("✅ Clé API configurée")
            
        return True
        
    def run_interactive_menu(self):
        """Lance le menu interactif principal"""
        while True:
            self.print_header()
            
            # Vérification rapide des prérequis
            prereq_ok = self.check_prerequisites()
            if not prereq_ok:
                print("\n⚠️ ATTENTION: Prérequis non satisfaits")
                print("💡 Recommandation: Commencez par l'installation (option 1)")
                
            self.print_menu()
            
            try:
                choice = input("\n🎯 Votre choix (0-9): ").strip()
                
                if choice not in self.available_actions:
                    print("❌ Choix invalide. Veuillez sélectionner 0-9.")
                    input("Appuyez sur Entrée pour continuer...")
                    continue
                    
                action = self.available_actions[choice]
                
                if action == "exit":
                    print("\n👋 Au revoir !")
                    break
                    
                # Exécution de l'action
                print(f"\n🚀 Exécution: {action}")
                time.sleep(0.5)  # Petite pause pour l'UX
                
                if action == "install":
                    self.run_installation()
                elif action == "demo":
                    self.run_demo()
                elif action == "test":
                    self.run_tests()
                elif action == "maintenance":
                    self.run_maintenance()
                elif action == "navigate":
                    self.start_navigation()
                elif action == "status":
                    self.generate_status_report()
                elif action == "diagnose":
                    self.run_diagnostics()
                elif action == "guide":
                    self.show_guide()
                elif action == "config":
                    self.configure_system()
                    
                input("\n⏸️ Appuyez sur Entrée pour revenir au menu...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Au revoir !")
                break
            except Exception as e:
                print(f"\n💥 Erreur: {e}")
                input("Appuyez sur Entrée pour continuer...")

def main():
    """Point d'entrée principal"""
    launcher = InteractiveNavigationLauncher()
    
    try:
        launcher.run_interactive_menu()
        return True
    except Exception as e:
        print(f"💥 Erreur critique: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
