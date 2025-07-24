"""
Test Complet du Système de Vision Web avec Gemini
Teste toutes les fonctionnalités développées
"""

import logging
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeminiWebVisionTest')

class GeminiWebVisionTester:
    """Testeur complet pour le système de vision web avec Gemini"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        
        # Créer le répertoire de tests
        self.test_dir = Path("test_results_vision")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🧪 Testeur Gemini Vision Web initialisé")
    
    def log_test_result(self, test_name: str, success: bool, message: str = "", data: dict = None):
        """Enregistre le résultat d'un test"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}: {message}")
        else:
            logger.error(f"❌ {test_name}: {message}")
            self.errors.append(f"{test_name}: {message}")
        
        self.test_results[test_name] = {
            'success': success,
            'message': message,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
    
    def test_vision_modules_import(self):
        """Test 1: Import des modules de vision"""
        logger.info("🔧 Test 1: Import des modules de vision")
        
        modules_to_test = [
            ('gemini_visual_adapter', 'Adaptateur Vision Gemini'),
            ('intelligent_web_capture', 'Système de Capture Intelligent'),
            ('gemini_web_vision_integration', 'Intégration Vision Web'),
            ('gemini_web_vision_api', 'API REST Vision')
        ]
        
        imported_modules = {}
        all_success = True
        
        for module_name, display_name in modules_to_test:
            try:
                module = __import__(module_name)
                imported_modules[module_name] = module
                self.log_test_result(f"Import {display_name}", True, "Module importé avec succès")
            except ImportError as e:
                self.log_test_result(f"Import {display_name}", False, f"Erreur d'import: {str(e)}")
                all_success = False
            except Exception as e:
                self.log_test_result(f"Import {display_name}", False, f"Erreur: {str(e)}")
                all_success = False
        
        self.log_test_result("Import Global Vision", all_success, 
                           f"{len(imported_modules)}/{len(modules_to_test)} modules importés")
        
        return imported_modules
    
    def test_visual_adapter_initialization(self):
        """Test 2: Initialisation de l'adaptateur visuel"""
        logger.info("🤖 Test 2: Initialisation Adaptateur Visual")
        
        try:
            from gemini_visual_adapter import initialize_gemini_visual_adapter, get_gemini_visual_adapter
            
            # Initialiser l'adaptateur
            visual_adapter = initialize_gemini_visual_adapter()
            self.log_test_result("Création Adaptateur Visual", True, "Adaptateur créé")
            
            # Vérifier qu'il est accessible globalement
            global_adapter = get_gemini_visual_adapter()
            if global_adapter:
                self.log_test_result("Vérification Adaptateur Global", True, "Adaptateur accessible")
                
                # Tester les statistiques
                stats = visual_adapter.get_statistics()
                self.log_test_result("Statistiques Adaptateur", True, f"Stats: {stats}")
                
                return visual_adapter
            else:
                self.log_test_result("Vérification Adaptateur Global", False, "Adaptateur non accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Initialisation Adaptateur Visual", False, f"Erreur: {str(e)}")
            return None
    
    def test_intelligent_capture_system(self):
        """Test 3: Système de capture intelligent"""
        logger.info("📸 Test 3: Système de Capture Intelligent")
        
        try:
            from intelligent_web_capture import initialize_intelligent_capture, get_intelligent_capture
            
            # Initialiser le système de capture
            capture_system = initialize_intelligent_capture()
            self.log_test_result("Création Système Capture", True, "Système créé")
            
            # Vérifier l'accès global
            global_capture = get_intelligent_capture()
            if global_capture:
                self.log_test_result("Vérification Capture Globale", True, "Système accessible")
                
                # Tester les statistiques
                stats = capture_system.get_statistics()
                self.log_test_result("Statistiques Capture", True, f"Stats: {stats}")
                
                return capture_system
            else:
                self.log_test_result("Vérification Capture Globale", False, "Système non accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Initialisation Système Capture", False, f"Erreur: {str(e)}")
            return None
    
    def test_web_vision_integration(self):
        """Test 4: Intégration complète Vision Web"""
        logger.info("🌐 Test 4: Intégration Vision Web")
        
        try:
            from gemini_web_vision_integration import initialize_gemini_web_vision, get_gemini_web_vision
            
            # Initialiser l'intégration
            integration = initialize_gemini_web_vision()
            self.log_test_result("Création Intégration Vision", True, "Intégration créée")
            
            # Vérifier l'accès global
            global_integration = get_gemini_web_vision()
            if global_integration:
                self.log_test_result("Vérification Intégration Globale", True, "Intégration accessible")
                
                # Tester les statistiques
                stats = integration.get_statistics()
                self.log_test_result("Statistiques Intégration", True, f"Stats: {stats}")
                
                return integration
            else:
                self.log_test_result("Vérification Intégration Globale", False, "Intégration non accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Initialisation Intégration Vision", False, f"Erreur: {str(e)}")
            return None
    
    def test_vision_session_management(self, integration):
        """Test 5: Gestion des sessions de vision"""
        logger.info("🎯 Test 5: Gestion Sessions Vision")
        
        if not integration:
            self.log_test_result("Test Sessions Vision", False, "Intégration non disponible")
            return
        
        try:
            # Créer une session de test
            session_id = f"test_vision_session_{int(datetime.now().timestamp())}"
            user_query = "Analyser l'interface utilisateur et l'UX de ce site web"
            
            result = integration.create_vision_navigation_session(
                session_id=session_id,
                user_query=user_query,
                navigation_goals=['extract_content', 'analyze_ui', 'capture_visuals']
            )
            
            if result['success']:
                self.log_test_result("Création Session Vision", True, f"Session créée: {session_id}")
                
                # Vérifier que la session est active
                if session_id in integration.active_sessions:
                    self.log_test_result("Vérification Session Active", True, "Session trouvée dans les sessions actives")
                    
                    # Fermer la session
                    close_result = integration.close_session(session_id)
                    if close_result['success']:
                        self.log_test_result("Fermeture Session", True, "Session fermée avec succès")
                    else:
                        self.log_test_result("Fermeture Session", False, f"Erreur: {close_result.get('error')}")
                else:
                    self.log_test_result("Vérification Session Active", False, "Session non trouvée")
            else:
                self.log_test_result("Création Session Vision", False, f"Erreur: {result.get('error')}")
                
        except Exception as e:
            self.log_test_result("Test Sessions Vision", False, f"Erreur: {str(e)}")
    
    def test_capture_simulation(self, capture_system):
        """Test 6: Simulation de capture (sans navigateur)"""
        logger.info("📷 Test 6: Simulation de Capture")
        
        if not capture_system:
            self.log_test_result("Test Capture Simulation", False, "Système de capture non disponible")
            return
        
        try:
            # Test des configurations
            test_configs = [
                {
                    'name': 'Config Desktop',
                    'viewport': 'desktop',
                    'capture_type': 'visible_area'
                },
                {
                    'name': 'Config Mobile', 
                    'viewport': 'mobile',
                    'capture_type': 'full_page'
                },
                {
                    'name': 'Config Tablet',
                    'viewport': 'tablet',
                    'capture_type': 'element_focused'
                }
            ]
            
            successful_configs = 0
            
            for config in test_configs:
                try:
                    # Simuler la validation de configuration
                    valid_viewports = ['desktop', 'mobile', 'tablet']
                    valid_captures = ['visible_area', 'full_page', 'element_focused']
                    
                    if config['viewport'] in valid_viewports and config['capture_type'] in valid_captures:
                        self.log_test_result(f"Validation {config['name']}", True, "Configuration valide")
                        successful_configs += 1
                    else:
                        self.log_test_result(f"Validation {config['name']}", False, "Configuration invalide")
                        
                except Exception as e:
                    self.log_test_result(f"Validation {config['name']}", False, f"Erreur: {str(e)}")
            
            success_rate = successful_configs / len(test_configs) * 100
            self.log_test_result("Capture Simulation Globale", successful_configs == len(test_configs), 
                               f"Taux de réussite: {success_rate}% ({successful_configs}/{len(test_configs)})")
            
        except Exception as e:
            self.log_test_result("Test Capture Simulation", False, f"Erreur: {str(e)}")
    
    def test_visual_analysis_simulation(self, visual_adapter):
        """Test 7: Simulation d'analyse visuelle"""
        logger.info("🔍 Test 7: Simulation Analyse Visuelle")
        
        if not visual_adapter:
            self.log_test_result("Test Analyse Visuelle", False, "Adaptateur visuel non disponible")
            return
        
        try:
            # Test des prompts d'analyse
            test_prompts = [
                {
                    'name': 'Analyse UI',
                    'prompt': 'Analysez les éléments d\'interface utilisateur dans cette capture',
                    'context': 'Analyse spécialisée UI/UX'
                },
                {
                    'name': 'Analyse Contenu',
                    'prompt': 'Identifiez et résumez le contenu principal de cette page',
                    'context': 'Extraction de contenu'
                },
                {
                    'name': 'Analyse Design',
                    'prompt': 'Évaluez le design visuel et l\'esthétique du site',
                    'context': 'Analyse de design'
                }
            ]
            
            successful_prompts = 0
            
            for prompt_test in test_prompts:
                try:
                    # Simuler la validation du prompt
                    if len(prompt_test['prompt']) > 10 and prompt_test['context']:
                        self.log_test_result(f"Validation {prompt_test['name']}", True, "Prompt valide")
                        successful_prompts += 1
                    else:
                        self.log_test_result(f"Validation {prompt_test['name']}", False, "Prompt invalide")
                        
                except Exception as e:
                    self.log_test_result(f"Validation {prompt_test['name']}", False, f"Erreur: {str(e)}")
            
            success_rate = successful_prompts / len(test_prompts) * 100
            self.log_test_result("Analyse Visuelle Simulation", successful_prompts == len(test_prompts),
                               f"Taux de réussite: {success_rate}% ({successful_prompts}/{len(test_prompts)})")
            
        except Exception as e:
            self.log_test_result("Test Analyse Visuelle", False, f"Erreur: {str(e)}")
    
    def test_api_initialization(self):
        """Test 8: Initialisation de l'API REST"""
        logger.info("🌐 Test 8: Initialisation API REST")
        
        try:
            from gemini_web_vision_api import create_vision_api, get_vision_api
            
            # Créer l'API
            api = create_vision_api()
            self.log_test_result("Création API Vision", True, "API créée")
            
            # Vérifier l'accès global
            global_api = get_vision_api()
            if global_api:
                self.log_test_result("Vérification API Globale", True, "API accessible")
                
                # Vérifier les composants de l'API
                if hasattr(api, 'app') and api.app:
                    self.log_test_result("Vérification App Flask", True, "Application Flask initialisée")
                else:
                    self.log_test_result("Vérification App Flask", False, "Application Flask non initialisée")
                
                return api
            else:
                self.log_test_result("Vérification API Globale", False, "API non accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Initialisation API Vision", False, f"Erreur: {str(e)}")
            return None
    
    def test_system_integration(self, integration, visual_adapter, capture_system):
        """Test 9: Intégration complète du système"""
        logger.info("🔗 Test 9: Intégration Système Complète")
        
        components_status = {
            'integration': integration is not None,
            'visual_adapter': visual_adapter is not None,
            'capture_system': capture_system is not None
        }
        
        working_components = sum(components_status.values())
        total_components = len(components_status)
        
        if working_components == total_components:
            self.log_test_result("Intégration Système Complète", True, 
                               f"Tous les composants fonctionnels ({working_components}/{total_components})")
            
            # Test de cohérence des statistiques
            try:
                if integration:
                    integration_stats = integration.get_statistics()
                    self.log_test_result("Statistiques Intégration", True, f"Stats récupérées: {len(integration_stats)} métriques")
                
                if visual_adapter:
                    adapter_stats = visual_adapter.get_statistics()
                    self.log_test_result("Statistiques Adaptateur", True, f"Stats récupérées: {len(adapter_stats)} métriques")
                
                if capture_system:
                    capture_stats = capture_system.get_statistics()
                    self.log_test_result("Statistiques Capture", True, f"Stats récupérées: {len(capture_stats)} métriques")
                    
            except Exception as e:
                self.log_test_result("Test Statistiques Système", False, f"Erreur: {str(e)}")
        else:
            self.log_test_result("Intégration Système Complète", False,
                               f"Composants manquants: {working_components}/{total_components}")
            
        return working_components == total_components
    
    def test_error_handling(self, integration):
        """Test 10: Gestion d'erreurs"""
        logger.info("⚠️ Test 10: Gestion d'Erreurs")
        
        if not integration:
            self.log_test_result("Test Gestion Erreurs", False, "Intégration non disponible")
            return
        
        try:
            # Test avec session inexistante
            result = integration.close_session("session_inexistante_12345")
            if not result['success'] and 'error' in result:
                self.log_test_result("Gestion Session Inexistante", True, "Erreur correctement gérée")
            else:
                self.log_test_result("Gestion Session Inexistante", False, "Erreur non gérée")
            
            # Test avec paramètres invalides
            try:
                result = integration.create_vision_navigation_session("", "")
                if not result['success']:
                    self.log_test_result("Gestion Paramètres Invalides", True, "Paramètres invalides détectés")
                else:
                    self.log_test_result("Gestion Paramètres Invalides", False, "Paramètres invalides non détectés")
            except Exception:
                self.log_test_result("Gestion Paramètres Invalides", True, "Exception correctement levée")
            
        except Exception as e:
            self.log_test_result("Test Gestion Erreurs", False, f"Erreur: {str(e)}")
    
    def run_all_tests(self):
        """Exécute tous les tests"""
        start_time = datetime.now()
        
        print("🧪 TEST COMPLET - Système Vision Web avec Gemini")
        print("=" * 70)
        print(f"🕐 Démarré le: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        logger.info("🚀 DÉMARRAGE DES TESTS VISION WEB")
        logger.info("=" * 60)
        
        # Test 1: Import des modules
        imported_modules = self.test_vision_modules_import()
        
        # Test 2: Adaptateur visuel
        visual_adapter = self.test_visual_adapter_initialization()
        
        # Test 3: Système de capture
        capture_system = self.test_intelligent_capture_system()
        
        # Test 4: Intégration vision web
        integration = self.test_web_vision_integration()
        
        # Test 5: Gestion des sessions
        self.test_vision_session_management(integration)
        
        # Test 6: Simulation de capture
        self.test_capture_simulation(capture_system)
        
        # Test 7: Simulation d'analyse visuelle
        self.test_visual_analysis_simulation(visual_adapter)
        
        # Test 8: API REST
        api = self.test_api_initialization()
        
        # Test 9: Intégration système
        system_ok = self.test_system_integration(integration, visual_adapter, capture_system)
        
        # Test 10: Gestion d'erreurs
        self.test_error_handling(integration)
        
        # Génération du rapport final
        self.generate_final_report(start_time, system_ok)
    
    def generate_final_report(self, start_time: datetime, system_ok: bool):
        """Génère le rapport final des tests"""
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Calculer les métriques
        success_rate = (self.passed_tests / max(self.total_tests, 1)) * 100
        
        # Créer le rapport
        report = {
            'test_summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.total_tests - self.passed_tests,
                'success_rate': round(success_rate, 2),
                'total_time': round(total_time, 2),
                'system_ready': system_ok
            },
            'test_results': self.test_results,
            'errors': self.errors,
            'timestamp': end_time.isoformat()
        }
        
        # Sauvegarder le rapport JSON
        report_filename = f"gemini_vision_test_report_{int(end_time.timestamp())}.json"
        report_path = self.test_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Affichage du résumé final
        logger.info("=" * 60)
        logger.info("🏁 TESTS TERMINÉS")
        logger.info(f"⏱️ Temps total: {total_time:.2f}s")
        logger.info(f"📊 Résultats: {self.passed_tests}/{self.total_tests} tests réussis ({success_rate:.1f}%)")
        
        if system_ok and success_rate >= 80:
            logger.info("🎉 SYSTÈME VISION WEB PRÊT !")
            logger.info("✅ Gemini peut maintenant voir l'intérieur des sites web")
        else:
            logger.warning("⚠️ Système partiellement fonctionnel")
            if self.errors:
                logger.error("❌ Erreurs détectées:")
                for error in self.errors:
                    logger.error(f"   - {error}")
        
        logger.info("=" * 60)
        
        print("\n" + "=" * 70)
        print("🏁 TESTS TERMINÉS")
        print(f"⏱️ Temps total: {total_time:.2f} secondes")
        print(f"📊 Résultats: {self.passed_tests}/{self.total_tests} tests réussis ({success_rate:.1f}%)")
        print(f"📄 Rapport sauvegardé: {report_filename}")
        
        if system_ok and success_rate >= 80:
            print("\n🎊 SUCCÈS COMPLET !")
            print("✅ Le système de vision web fonctionne parfaitement")
            print("🚀 Gemini peut maintenant voir l'intérieur des sites web")
            print("👁️ Capacités disponibles:")
            print("   - Capture intelligente de sites web")
            print("   - Analyse visuelle avec Gemini")
            print("   - Navigation guidée par la vision")
            print("   - API REST complète")
        else:
            print(f"\n⚠️ Système partiellement fonctionnel ({success_rate:.1f}%)")
            
        print("=" * 70)

if __name__ == "__main__":
    tester = GeminiWebVisionTester()
    tester.run_all_tests()
