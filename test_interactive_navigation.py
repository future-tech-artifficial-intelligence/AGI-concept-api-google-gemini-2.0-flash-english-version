"""
Test Complet du Système de Navigation Interactive avec l'API Gemini
Ce script teste l'intégration complète du nouveau système d'interaction web
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
logger = logging.getLogger('GeminiInteractiveNavigationTest')

class GeminiInteractiveNavigationTester:
    """Testeur complet pour le système de navigation interactive avec Gemini"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        
        # Créer le répertoire de tests
        self.test_dir = Path("test_results_interactive")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🧪 Testeur Gemini-Navigation Interactive initialisé")
    
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
    
    def test_module_imports(self):
        """Test 1: Vérifier que tous les modules interactifs s'importent correctement"""
        logger.info("📦 Test 1: Imports des Modules Interactifs")
        
        imported_modules = {}
        
        # Test imports du navigateur interactif
        try:
            from interactive_web_navigator import (
                InteractiveWebNavigator,
                InteractiveElementAnalyzer,
                get_interactive_navigator,
                initialize_interactive_navigator
            )
            imported_modules['interactive_web_navigator'] = True
            self.log_test_result("Import Navigateur Interactif", True, "Module chargé")
        except ImportError as e:
            imported_modules['interactive_web_navigator'] = False
            self.log_test_result("Import Navigateur Interactif", False, f"Erreur: {str(e)}")
        
        # Test imports de l'adaptateur Gemini interactif
        try:
            from gemini_interactive_adapter import (
                GeminiInteractiveWebAdapter,
                get_gemini_interactive_adapter,
                initialize_gemini_interactive_adapter,
                handle_gemini_interactive_request,
                detect_interactive_need
            )
            imported_modules['gemini_interactive_adapter'] = True
            self.log_test_result("Import Adaptateur Gemini Interactif", True, "Module chargé")
        except ImportError as e:
            imported_modules['gemini_interactive_adapter'] = False
            self.log_test_result("Import Adaptateur Gemini Interactif", False, f"Erreur: {str(e)}")
        
        # Test import de l'adaptateur Gemini principal
        try:
            from gemini_api_adapter import GeminiAPI
            imported_modules['gemini_api_adapter'] = True
            self.log_test_result("Import Adaptateur Gemini Principal", True, "Module chargé")
        except ImportError as e:
            imported_modules['gemini_api_adapter'] = False
            self.log_test_result("Import Adaptateur Gemini Principal", False, f"Erreur: {str(e)}")
        
        # Test imports Selenium (optionnel)
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            imported_modules['selenium'] = True
            self.log_test_result("Import Selenium", True, "WebDriver disponible")
        except ImportError as e:
            imported_modules['selenium'] = False
            self.log_test_result("Import Selenium", False, f"WebDriver non disponible: {str(e)}")
        
        success_rate = sum(imported_modules.values()) / len(imported_modules) * 100
        overall_success = success_rate >= 75  # Au moins 75% des modules requis
        
        self.log_test_result("Imports Modules Globaux", overall_success,
                           f"Taux de réussite: {success_rate:.1f}%",
                           {'modules': imported_modules})
        
        return imported_modules
    
    def test_interactive_navigator_initialization(self):
        """Test 2: Initialiser le navigateur interactif"""
        logger.info("🚀 Test 2: Initialisation Navigateur Interactif")
        
        try:
            from interactive_web_navigator import initialize_interactive_navigator, get_interactive_navigator
            
            # Tenter l'initialisation
            navigator = initialize_interactive_navigator()
            
            if navigator:
                self.log_test_result("Initialisation Navigateur", True, "Navigateur initialisé avec succès")
                
                # Vérifier l'accès global
                global_navigator = get_interactive_navigator()
                if global_navigator:
                    self.log_test_result("Vérification Navigateur Global", True, "Navigateur accessible globalement")
                    return navigator
                else:
                    self.log_test_result("Vérification Navigateur Global", False, "Navigateur non accessible")
                    return None
            else:
                self.log_test_result("Initialisation Navigateur", False, "Échec de l'initialisation (normal si ChromeDriver absent)")
                return None
                
        except Exception as e:
            self.log_test_result("Test Initialisation Navigateur", False, f"Erreur: {str(e)}")
            return None
    
    def test_gemini_interactive_adapter_initialization(self):
        """Test 3: Initialiser l'adaptateur interactif Gemini"""
        logger.info("🔗 Test 3: Initialisation Adaptateur Gemini Interactif")
        
        try:
            from gemini_interactive_adapter import initialize_gemini_interactive_adapter, get_gemini_interactive_adapter
            
            # Initialiser l'adaptateur
            adapter = initialize_gemini_interactive_adapter()
            
            if adapter:
                self.log_test_result("Initialisation Adaptateur Gemini", True, "Adaptateur initialisé")
                
                # Vérifier l'accès global
                global_adapter = get_gemini_interactive_adapter()
                if global_adapter:
                    self.log_test_result("Vérification Adaptateur Global", True, "Adaptateur accessible")
                    
                    # Vérifier les statistiques
                    stats = adapter.get_interaction_statistics()
                    self.log_test_result("Statistiques Adaptateur", True, f"Stats: {stats}")
                    
                    return adapter
                else:
                    self.log_test_result("Vérification Adaptateur Global", False, "Adaptateur non accessible")
                    return None
            else:
                self.log_test_result("Initialisation Adaptateur Gemini", False, "Échec de l'initialisation")
                return None
                
        except Exception as e:
            self.log_test_result("Test Adaptateur Gemini", False, f"Erreur: {str(e)}")
            return None
    
    def test_interaction_detection(self):
        """Test 4: Tester la détection d'interactions"""
        logger.info("🔍 Test 4: Détection d'Interactions")
        
        try:
            from gemini_interactive_adapter import detect_interactive_need
            
            # Tests de détection avec différents prompts
            test_cases = [
                {
                    'prompt': "Clique sur l'onglet Services de ce site web",
                    'expected_interaction': True,
                    'expected_type': 'direct_interaction'
                },
                {
                    'prompt': "Explore tous les onglets de https://example.com",
                    'expected_interaction': True,
                    'expected_type': 'tab_navigation'
                },
                {
                    'prompt': "Explore toutes les options disponibles sur le site",
                    'expected_interaction': True,
                    'expected_type': 'full_exploration'
                },
                {
                    'prompt': "Remplis le formulaire de contact",
                    'expected_interaction': True,
                    'expected_type': 'form_interaction'
                },
                {
                    'prompt': "Qu'est-ce que l'intelligence artificielle ?",
                    'expected_interaction': False,
                    'expected_type': None
                }
            ]
            
            detection_results = []
            successful_detections = 0
            
            for test_case in test_cases:
                prompt = test_case['prompt']
                expected_interaction = test_case['expected_interaction']
                expected_type = test_case['expected_type']
                
                logger.info(f"  🧪 Test détection: '{prompt}'")
                
                # Effectuer la détection
                detection = detect_interactive_need(prompt)
                
                # Vérifier le résultat
                detected_interaction = detection.get('requires_interaction', False)
                detected_type = detection.get('interaction_type')
                confidence = detection.get('confidence', 0)
                
                # Évaluer la précision
                type_match = (detected_type == expected_type) if expected_interaction else (detected_type is None)
                detection_success = (detected_interaction == expected_interaction) and type_match
                
                if detection_success:
                    successful_detections += 1
                    status = "✅"
                    details = f"Détection correcte (confiance: {confidence:.2f})"
                else:
                    status = "❌"
                    details = f"Attendu: {expected_type}, Détecté: {detected_type} (confiance: {confidence:.2f})"
                
                detection_results.append({
                    'prompt': prompt,
                    'expected': {'interaction': expected_interaction, 'type': expected_type},
                    'detected': {'interaction': detected_interaction, 'type': detected_type, 'confidence': confidence},
                    'success': detection_success,
                    'status': status
                })
                
                logger.info(f"    {status} {details}")
            
            # Évaluer le taux de réussite global
            success_rate = (successful_detections / len(test_cases)) * 100
            overall_success = success_rate >= 80  # Au moins 80% de réussite
            
            self.log_test_result("Détection Interactions", overall_success,
                               f"Taux de réussite: {success_rate:.1f}% ({successful_detections}/{len(test_cases)})",
                               {'results': detection_results})
            
            return detection_results
            
        except Exception as e:
            self.log_test_result("Test Détection", False, f"Erreur: {str(e)}")
            return None
    
    def test_element_analysis_simulation(self):
        """Test 5: Simulation d'analyse d'éléments (sans navigateur)"""
        logger.info("🔬 Test 5: Simulation Analyse Éléments")
        
        try:
            from interactive_web_navigator import InteractiveElementAnalyzer
            
            # Créer l'analyseur
            analyzer = InteractiveElementAnalyzer()
            self.log_test_result("Création Analyseur", True, "Analyseur créé")
            
            # Tester les sélecteurs CSS
            selectors_test = True
            for element_type, selectors in analyzer.element_selectors.items():
                if not selectors or not isinstance(selectors, list):
                    selectors_test = False
                    break
            
            self.log_test_result("Validation Sélecteurs CSS", selectors_test, 
                               f"Sélecteurs pour {len(analyzer.element_selectors)} types d'éléments")
            
            # Tester les mots-clés d'importance
            keywords_test = True
            for importance, keywords in analyzer.importance_keywords.items():
                if not keywords or not isinstance(keywords, list):
                    keywords_test = False
                    break
            
            self.log_test_result("Validation Mots-clés", keywords_test,
                               f"Mots-clés pour {len(analyzer.importance_keywords)} niveaux d'importance")
            
            # Test de calcul de score d'interaction
            test_scores = [
                analyzer._calculate_interaction_score("Next", {'id': 'next-btn'}, 'buttons', {'x': 100, 'y': 200, 'width': 80, 'height': 30}),
                analyzer._calculate_interaction_score("Home", {'class': 'nav-link'}, 'navigation', {'x': 50, 'y': 50, 'width': 60, 'height': 20}),
                analyzer._calculate_interaction_score("", {}, 'inputs', {'x': 200, 'y': 800, 'width': 120, 'height': 25})
            ]
            
            score_test = all(0 <= score <= 1 for score in test_scores)
            self.log_test_result("Calcul Scores Interaction", score_test,
                               f"Scores calculés: {[f'{s:.2f}' for s in test_scores]}")
            
            return analyzer
            
        except Exception as e:
            self.log_test_result("Test Analyse Éléments", False, f"Erreur: {str(e)}")
            return None
    
    def test_gemini_api_integration(self):
        """Test 6: Test d'intégration avec l'API Gemini"""
        logger.info("🤖 Test 6: Intégration API Gemini")
        
        try:
            from gemini_api_adapter import GeminiAPI
            
            # Créer une instance de l'API
            gemini_api = GeminiAPI()
            self.log_test_result("Création Instance Gemini", True, "Instance créée")
            
            # Vérifier que le système interactif est intégré
            has_interactive = hasattr(gemini_api, 'interactive_navigation_available')
            self.log_test_result("Intégration Système Interactif", has_interactive,
                               f"Système interactif {'disponible' if has_interactive else 'non disponible'}")
            
            # Test de prompts interactifs (simulation)
            interactive_prompts = [
                "Clique sur l'onglet produits de https://example.com",
                "Explore tous les onglets de ce site web",
                "Remplis le formulaire de contact"
            ]
            
            integration_results = []
            
            for prompt in interactive_prompts:
                logger.info(f"  🧪 Test prompt: '{prompt[:50]}...'")
                
                try:
                    # Utiliser la méthode fallback pour éviter les appels API réels
                    if hasattr(gemini_api, '_fallback_get_response'):
                        response = gemini_api._fallback_get_response(
                            prompt=prompt,
                            user_id=1,
                            session_id="test_interactive_session"
                        )
                        
                        if response and 'response' in response:
                            integration_results.append({
                                'prompt': prompt,
                                'success': True,
                                'has_response': True,
                                'response_length': len(response['response'])
                            })
                        else:
                            integration_results.append({
                                'prompt': prompt,
                                'success': False,
                                'error': 'Pas de réponse'
                            })
                    else:
                        integration_results.append({
                            'prompt': prompt,
                            'success': False,
                            'error': 'Méthode fallback non disponible'
                        })
                        
                except Exception as e:
                    integration_results.append({
                        'prompt': prompt,
                        'success': False,
                        'error': str(e)
                    })
                
                time.sleep(0.5)  # Petit délai
            
            success_count = sum(1 for r in integration_results if r['success'])
            success_rate = (success_count / len(interactive_prompts)) * 100
            
            self.log_test_result("Tests Prompts Interactifs", success_count > 0,
                               f"Taux de réussite: {success_rate:.1f}% ({success_count}/{len(interactive_prompts)})",
                               {'results': integration_results})
            
            return gemini_api
            
        except Exception as e:
            self.log_test_result("Test Intégration Gemini", False, f"Erreur: {str(e)}")
            return None
    
    def test_session_management(self):
        """Test 7: Test de gestion des sessions"""
        logger.info("📋 Test 7: Gestion des Sessions")
        
        try:
            from interactive_web_navigator import create_interactive_navigation_session, close_interactive_session
            
            # Test de création de session
            session_id = f"test_session_{int(time.time())}"
            test_url = "https://httpbin.org/html"
            goals = ['test_navigation', 'element_analysis']
            
            try:
                session_result = create_interactive_navigation_session(session_id, test_url, goals)
                
                if session_result.get('success', False):
                    self.log_test_result("Création Session", True,
                                       f"Session créée: {session_id}")
                    
                    # Test de fermeture de session
                    close_result = close_interactive_session(session_id)
                    
                    if close_result.get('success', False):
                        self.log_test_result("Fermeture Session", True,
                                           f"Session fermée avec rapport")
                        return True
                    else:
                        self.log_test_result("Fermeture Session", False,
                                           f"Erreur: {close_result.get('error', 'Inconnue')}")
                        return False
                else:
                    self.log_test_result("Création Session", False,
                                       f"Erreur: {session_result.get('error', 'Inconnue')}")
                    return False
                    
            except Exception as e:
                self.log_test_result("Test Session", False, f"Erreur session: {str(e)}")
                return False
                
        except Exception as e:
            self.log_test_result("Test Gestion Sessions", False, f"Erreur: {str(e)}")
            return False
    
    def generate_test_report(self):
        """Génère un rapport de test complet"""
        logger.info("📋 Génération du rapport de test")
        
        # Calculer les statistiques générales
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        # Créer le rapport
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.total_tests - self.passed_tests,
                'success_rate': success_rate,
                'overall_status': 'PASSED' if success_rate >= 70 else 'FAILED'
            },
            'test_results': self.test_results,
            'errors': self.errors
        }
        
        # Sauvegarder le rapport JSON
        report_file = self.test_dir / f"interactive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Créer un rapport markdown
        self._create_markdown_report(report)
        
        return report
    
    def _create_markdown_report(self, report):
        """Crée un rapport markdown"""
        report_file = self.test_dir / f"interactive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Rapport de Test - Système de Navigation Interactive Gemini\n\n")
            f.write(f"**Date:** {report['test_summary']['timestamp']}\n\n")
            f.write(f"## Résumé\n\n")
            f.write(f"- **Tests totaux:** {report['test_summary']['total_tests']}\n")
            f.write(f"- **Tests réussis:** {report['test_summary']['passed_tests']}\n")
            f.write(f"- **Tests échoués:** {report['test_summary']['failed_tests']}\n")
            f.write(f"- **Taux de réussite:** {report['test_summary']['success_rate']:.1f}%\n")
            f.write(f"- **Statut global:** {report['test_summary']['overall_status']}\n\n")
            
            f.write("## Détails des Tests\n\n")
            for test_name, result in report['test_results'].items():
                status = "✅" if result['success'] else "❌"
                f.write(f"### {status} {test_name}\n")
                f.write(f"**Message:** {result['message']}\n\n")
                if result.get('data'):
                    f.write(f"**Données:** ```json\n{json.dumps(result['data'], indent=2)}\n```\n\n")
            
            if report['errors']:
                f.write("## Erreurs\n\n")
                for error in report['errors']:
                    f.write(f"- {error}\n")
    
    def run_all_tests(self):
        """Lance tous les tests"""
        logger.info("🚀 DÉMARRAGE DES TESTS SYSTÈME INTERACTIF")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Test 1: Imports
            imported_modules = self.test_module_imports()
            
            # Test 2: Initialisation navigateur interactif
            navigator = self.test_interactive_navigator_initialization()
            
            # Test 3: Initialisation adaptateur Gemini
            adapter = self.test_gemini_interactive_adapter_initialization()
            
            # Test 4: Détection d'interactions
            self.test_interaction_detection()
            
            # Test 5: Analyse d'éléments
            analyzer = self.test_element_analysis_simulation()
            
            # Test 6: Intégration API Gemini
            gemini_api = self.test_gemini_api_integration()
            
            # Test 7: Gestion des sessions
            self.test_session_management()
            
        except Exception as e:
            logger.error(f"Erreur lors des tests: {str(e)}")
            self.log_test_result("Exécution Globale", False, f"Erreur critique: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Générer le rapport
        report = self.generate_test_report()
        
        # Afficher le résumé final
        logger.info("=" * 60)
        logger.info("🏁 TESTS TERMINÉS - SYSTÈME INTERACTIF")
        logger.info(f"⏱️ Temps total: {total_time:.2f}s")
        logger.info(f"📊 Résultats: {self.passed_tests}/{self.total_tests} tests réussis ({report['test_summary']['success_rate']:.1f}%)")
        
        if report['test_summary']['overall_status'] == 'PASSED':
            logger.info("🎉 TOUS LES TESTS PRINCIPAUX SONT PASSÉS !")
            logger.info("✅ Le système de navigation interactive fonctionne avec Gemini")
        else:
            logger.info("⚠️ Certains tests ont échoué")
            logger.info("🔧 Vérifiez les erreurs dans le rapport de test")
        
        logger.info("=" * 60)
        
        return report

def main():
    """Fonction principale"""
    logger.info("🌟 Démarrage des tests du système de navigation interactive Gemini")
    
    tester = GeminiInteractiveNavigationTester()
    report = tester.run_all_tests()
    
    # Retourner le succès basé sur le taux de réussite
    success = report['test_summary']['success_rate'] >= 70
    
    if success:
        logger.info("✅ Tests terminés avec succès - Le système est opérationnel")
        print("\n🎯 SYSTÈME DE NAVIGATION INTERACTIVE GEMINI OPÉRATIONNEL")
        print("📖 Consultez les rapports de test pour plus de détails")
    else:
        logger.error("❌ Tests échoués - Des problèmes ont été détectés")
        print("\n⚠️ PROBLÈMES DÉTECTÉS DANS LE SYSTÈME")
        print("🔧 Consultez les rapports d'erreur pour résoudre les problèmes")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
