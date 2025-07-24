#!/usr/bin/env python3
"""
Script de test complet du système Searx intelligent v2
Vérifie tous les composants et dépendances avec gestion d'erreurs avancée
"""

import sys
import logging
import traceback
import time
from pathlib import Path

# Configuration du logging pour les tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('SearxSystemTest')

class SearxSystemTester:
    """Testeur complet du système Searx"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def test_imports(self):
        """Test des imports des modules principaux"""
        print("🔍 TEST DES IMPORTS")
        print("=" * 30)
        
        tests = [
            ("requests", "Requêtes HTTP"),
            ("psutil", "Gestion des processus"),
            ("bs4", "BeautifulSoup pour parsing HTML"),
            ("selenium", "Automation web"),
            ("PIL", "Pillow pour images"),
            ("json", "JSON standard"),
            ("socket", "Sockets réseau"),
            ("subprocess", "Processus système"),
            ("platform", "Information plateforme"),
            ("docker", "Client Docker Python")
        ]
        
        success_count = 0
        for module, description in tests:
            try:
                __import__(module)
                print(f"✅ {module:12} - {description}")
                success_count += 1
            except ImportError as e:
                print(f"❌ {module:12} - {description} - ERREUR: {e}")
        
        print(f"\n📊 Résultat: {success_count}/{len(tests)} modules disponibles")
        self.results['imports'] = success_count == len(tests)
        return self.results['imports']

    def test_port_manager(self):
        """Test du gestionnaire de ports"""
        print("\n🔧 TEST DU GESTIONNAIRE DE PORTS")
        print("=" * 40)
        
        try:
            from port_manager import PortManager
            pm = PortManager()
            
            # Test de détection de port
            port_8080_available = pm.is_port_available(8080)
            print(f"📍 Port 8080 disponible: {'✅ Oui' if port_8080_available else '❌ Non'}")
            
            if not port_8080_available:
                process = pm.get_process_using_port(8080)
                if process:
                    print(f"🔍 Processus sur 8080: {process['name']} (PID: {process['pid']})")
                    print(f"   Commande: {process['cmdline'][:50]}...")
            
            # Test de recherche de port alternatif
            alt_port = pm.find_available_port(8081, 5)
            if alt_port:
                print(f"🔄 Port alternatif trouvé: {alt_port}")
            else:
                print("⚠️  Aucun port alternatif trouvé")
            
            # Test de génération de configuration
            config_success, port, compose_file = pm.setup_searx_with_available_port()
            if config_success:
                print(f"✅ Configuration générée: {compose_file} (port {port})")
            else:
                print("⚠️  Impossible de générer une configuration")
            
            print("✅ Gestionnaire de ports fonctionnel")
            self.results['port_manager'] = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur gestionnaire de ports: {e}")
            self.results['port_manager'] = False
            return False

    def test_searx_interface(self):
        """Test de l'interface Searx"""
        print("\n🔍 TEST DE L'INTERFACE SEARX")
        print("=" * 35)
        
        try:
            from searx_interface import SearxInterface
            
            # Créer une instance sans démarrer Searx
            searx = SearxInterface()
            print("✅ Interface Searx créée")
            
            # Vérifier l'initialisation du gestionnaire de ports
            if searx.port_manager:
                print("✅ Gestionnaire de ports intégré")
            else:
                print("⚠️  Gestionnaire de ports non initialisé")
            
            # Vérifier l'initialisation de la capture visuelle
            if searx.visual_capture:
                print("✅ Capture visuelle intégrée")
            else:
                print("⚠️  Capture visuelle non initialisée (normal si ChromeDriver absent)")
            
            print("✅ Interface Searx fonctionnelle")
            self.results['searx_interface'] = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur interface Searx: {e}")
            self.results['searx_interface'] = False
            return False

    def test_docker(self):
        """Test de la disponibilité de Docker"""
        print("\n🐳 TEST DE DOCKER")
        print("=" * 20)
        
        try:
            import subprocess
            
            # Vérifier si Docker est installé
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Docker disponible: {version}")
                
                # Vérifier si Docker fonctionne
                result = subprocess.run(['docker', 'ps'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print("✅ Docker daemon actif")
                    
                    # Vérifier Docker Compose
                    result = subprocess.run(['docker-compose', '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"✅ Docker Compose disponible: {result.stdout.strip()}")
                    else:
                        print("⚠️  Docker Compose non disponible")
                    
                    self.results['docker'] = True
                    return True
                else:
                    print("⚠️  Docker installé mais daemon non actif")
                    print("💡 Démarrez Docker Desktop")
                    self.results['docker'] = False
                    return False
            else:
                print("❌ Docker non installé")
                print("💡 Installez Docker Desktop")
                self.results['docker'] = False
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Docker ne répond pas (timeout)")
            self.results['docker'] = False
            return False
        except FileNotFoundError:
            print("❌ Docker non trouvé dans le PATH")
            self.results['docker'] = False
            return False
        except Exception as e:
            print(f"❌ Erreur Docker: {e}")
            self.results['docker'] = False
            return False

    def test_files(self):
        """Test de la présence des fichiers nécessaires"""
        print("\n📋 TEST DES FICHIERS SYSTÈME")
        print("=" * 32)
        
        required_files = [
            ("port_manager.py", "Gestionnaire de ports"),
            ("searx_interface.py", "Interface Searx"),
            ("searx_smart_start.py", "Script de démarrage"),
            ("searx_manager.bat", "Script Windows"),
            ("requirements.txt", "Dépendances Python")
        ]
        
        optional_files = [
            ("docker-compose.searx.yml", "Config Docker principale"),
            ("docker-compose.searx-alt.yml", "Config Docker alternative"),
            ("free_port_8080.bat", "Script libération port"),
            ("searx_visual_capture.py", "Capture visuelle")
        ]
        
        required_count = 0
        optional_count = 0
        
        print("📁 Fichiers requis:")
        for filename, description in required_files:
            if Path(filename).exists():
                print(f"✅ {filename:25} - {description}")
                required_count += 1
            else:
                print(f"❌ {filename:25} - {description} - MANQUANT")
        
        print("\n📁 Fichiers optionnels:")
        for filename, description in optional_files:
            if Path(filename).exists():
                print(f"✅ {filename:25} - {description}")
                optional_count += 1
            else:
                print(f"⚠️  {filename:25} - {description} - Non trouvé")
        
        total_required = len(required_files)
        total_optional = len(optional_files)
        
        print(f"\n📊 Fichiers requis: {required_count}/{total_required}")
        print(f"📊 Fichiers optionnels: {optional_count}/{total_optional}")
        
        self.results['files'] = required_count == total_required
        return self.results['files']

    def test_smart_start(self):
        """Test du script de démarrage intelligent"""
        print("\n🚀 TEST DU DÉMARRAGE INTELLIGENT")
        print("=" * 38)
        
        try:
            # Import du module de démarrage
            import searx_smart_start
            print("✅ Module de démarrage intelligent importé")
            
            # Tester les fonctions principales
            if hasattr(searx_smart_start, 'main'):
                print("✅ Fonction main() disponible")
            
            if hasattr(searx_smart_start, 'show_status'):
                print("✅ Fonction show_status() disponible")
            
            if hasattr(searx_smart_start, 'stop_all'):
                print("✅ Fonction stop_all() disponible")
            
            print("✅ Script de démarrage intelligent fonctionnel")
            self.results['smart_start'] = True
            return True
            
        except Exception as e:
            print(f"❌ Erreur script démarrage: {e}")
            self.results['smart_start'] = False
            return False

    def run_full_test(self):
        """Exécute tous les tests et génère un rapport"""
        print("🎯 DÉBUT DES TESTS SYSTÈME SEARX")
        print("=" * 60)
        print(f"📅 Date: {time.ctime()}")
        print(f"🖥️  Plateforme: {sys.platform}")
        print(f"🐍 Python: {sys.version}")
        print("=" * 60)
        
        tests = [
            ("Imports Python", self.test_imports),
            ("Gestionnaire de ports", self.test_port_manager),
            ("Interface Searx", self.test_searx_interface),
            ("Docker", self.test_docker),
            ("Fichiers système", self.test_files),
            ("Démarrage intelligent", self.test_smart_start)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ ERREUR CRITIQUE dans {test_name}: {e}")
                self.results[test_name.lower().replace(' ', '_')] = False
        
        # Générer le rapport final
        self._generate_final_report(passed_tests, total_tests)
        
        return passed_tests >= total_tests * 0.8

    def _generate_final_report(self, passed_tests, total_tests):
        """Génère le rapport final"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📋 RAPPORT FINAL - SYSTÈME SEARX INTELLIGENT")
        print("=" * 60)
        
        print(f"⏱️  Durée des tests: {elapsed_time:.2f} secondes")
        print(f"🏆 Score: {passed_tests}/{total_tests} tests réussis")
        
        # Détail des résultats
        print("\n📊 DÉTAIL DES RÉSULTATS:")
        for test_name, result in self.results.items():
            status = "✅ SUCCÈS" if result else "❌ ÉCHEC"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        # Statut global
        success_rate = passed_tests / total_tests
        if success_rate >= 0.9:
            print("\n🎉 EXCELLENT! Système complètement fonctionnel")
            print("🚀 Prêt pour production - Lancez: python searx_smart_start.py")
        elif success_rate >= 0.7:
            print("\n✅ BON! Système largement fonctionnel")
            print("💡 Quelques améliorations possibles")
        elif success_rate >= 0.5:
            print("\n⚠️  MOYEN! Système partiellement fonctionnel")
            print("🔧 Corrections nécessaires")
        else:
            print("\n❌ CRITIQUE! Système non fonctionnel")
            print("🆘 Intervention urgente requise")
        
        # Recommandations spécifiques
        self._generate_recommendations()

    def _generate_recommendations(self):
        """Génère des recommandations basées sur les résultats"""
        print("\n💡 RECOMMANDATIONS SPÉCIFIQUES:")
        
        if not self.results.get('imports', True):
            print("📦 DÉPENDANCES:")
            print("   - Exécutez: pip install -r requirements.txt")
            print("   - Vérifiez votre environnement Python")
        
        if not self.results.get('docker', True):
            print("🐳 DOCKER:")
            print("   - Installez Docker Desktop: https://docker.com/products/docker-desktop")
            print("   - Démarrez le service Docker")
            print("   - Vérifiez que Docker fonctionne: docker ps")
        
        if not self.results.get('files', True):
            print("📁 FICHIERS:")
            print("   - Vérifiez l'intégrité du projet")
            print("   - Re-téléchargez les fichiers manquants")
        
        if not self.results.get('port_manager', True):
            print("🔧 GESTIONNAIRE DE PORTS:")
            print("   - Installez psutil: pip install psutil")
            print("   - Vérifiez les permissions système")
        
        print("\n🔗 RESSOURCES:")
        print("   - Documentation Searx: https://searx.github.io/searx/")
        print("   - Guide Docker: https://docs.docker.com/get-started/")
        print("   - Support Python: https://python.org/downloads/")

def main():
    """Fonction principale"""
    try:
        tester = SearxSystemTester()
        success = tester.run_full_test()
        
        if success:
            print("\n🎯 PROCHAINES ÉTAPES:")
            print("1. Lancez: python searx_smart_start.py")
            print("2. Ou utilisez: searx_manager.bat (Windows)")
            print("3. Testez l'interface web une fois démarré")
        else:
            print("\n🔧 ACTIONS REQUISES:")
            print("1. Corrigez les erreurs signalées")
            print("2. Relancez ce test: python test_searx_complete.py")
            print("3. Contactez le support si problèmes persistants")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Test interrompu par l'utilisateur")
        return 2
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())
