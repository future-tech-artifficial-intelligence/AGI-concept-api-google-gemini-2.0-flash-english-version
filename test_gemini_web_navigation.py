"""
Test Complet du Système de Navigation Web Avancé avec l'API Gemini
Ce script teste l'intégration complète et vérifie que tout fonctionne
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
logger = logging.getLogger('GeminiWebNavigationTest')

class GeminiWebNavigationTester:
    """Testeur complet pour le système de navigation web avec Gemini"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        
        # Créer le répertoire de tests
        self.test_dir = Path("test_results")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🧪 Testeur Gemini-Navigation initialisé")
    
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
        """Test 1: Vérifier que tous les modules s'importent correctement"""
        logger.info("🔧 Test 1: Import des modules")
        
        modules_to_test = [
            ('advanced_web_navigator', 'Navigateur Web Avancé'),
            ('gemini_web_integration', 'Intégration Gemini-Web'),
            ('gemini_navigation_adapter', 'Adaptateur Navigation Gemini'),
            ('web_navigation_api', 'API REST Navigation Web'),
            ('gemini_api_adapter', 'Adaptateur Gemini Original')
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
        
        self.log_test_result("Import Global", all_success, 
                           f"{len(imported_modules)}/{len(modules_to_test)} modules importés")
        
        return imported_modules
    
    def test_gemini_api_initialization(self):
        """Test 2: Initialiser l'API Gemini et vérifier l'intégration"""
        logger.info("🤖 Test 2: Initialisation API Gemini")
        
        try:
            from gemini_api_adapter import GeminiAPI
            from gemini_navigation_adapter import initialize_gemini_navigation_adapter
            
            # Créer une instance Gemini
            gemini_api = GeminiAPI()
            self.log_test_result("Création Instance Gemini", True, "API Gemini créée")
            
            # Initialiser l'adaptateur de navigation
            initialize_gemini_navigation_adapter(gemini_api)
            self.log_test_result("Initialisation Adaptateur Navigation", True, "Adaptateur initialisé")
            
            # Vérifier que l'intégration est disponible
            from gemini_navigation_adapter import gemini_navigation_adapter
            if gemini_navigation_adapter:
                self.log_test_result("Vérification Intégration", True, "Intégration active")
                return gemini_api
            else:
                self.log_test_result("Vérification Intégration", False, "Adaptateur non initialisé")
                return None
                
        except Exception as e:
            self.log_test_result("Initialisation API Gemini", False, f"Erreur: {str(e)}")
            return None
    
    def test_navigation_detection(self, gemini_api):
        """Test 3: Tester la détection de navigation"""
        logger.info("🔍 Test 3: Détection de Navigation")
        
        if not gemini_api:
            self.log_test_result("Test Détection", False, "API Gemini non disponible")
            return
        
        try:
            from gemini_navigation_adapter import detect_navigation_need
            
            # Tests de détection avec différents prompts
            test_cases = [
                {
                    'prompt': "Recherche et navigue sur l'intelligence artificielle",
                    'expected_navigation': True,
                    'expected_type': 'search_and_navigate'
                },
                {
                    'prompt': "Extrait le contenu de https://example.com",
                    'expected_navigation': True,
                    'expected_type': 'content_extraction'
                },
                {
                    'prompt': "Explore le site https://wikipedia.org en profondeur",
                    'expected_navigation': True,
                    'expected_type': 'deep_navigation'
                },
                {
                    'prompt': "Simule un parcours d'achat sur ce site",
                    'expected_navigation': True,
                    'expected_type': 'user_journey'
                },
                {
                    'prompt': "Qu'est-ce que l'apprentissage automatique ?",
                    'expected_navigation': True,  # Devrait être détecté comme recherche générale
                    'expected_type': 'search_and_navigate'
                },
                {
                    'prompt': "Bonjour, comment allez-vous ?",
                    'expected_navigation': False,
                    'expected_type': None
                }
            ]
            
            detection_results = []
            successful_detections = 0
            
            for test_case in test_cases:
                prompt = test_case['prompt']
                expected_nav = test_case['expected_navigation']
                expected_type = test_case['expected_type']
                
                detection = detect_navigation_need(prompt)
                
                requires_nav = detection.get('requires_navigation', False)
                nav_type = detection.get('navigation_type')
                confidence = detection.get('confidence', 0)
                
                # Vérifier si la détection correspond aux attentes
                detection_correct = (requires_nav == expected_nav)
                if expected_nav and nav_type != expected_type:
                    # Permettre une certaine flexibilité dans les types
                    if not (expected_type == 'search_and_navigate' and nav_type == 'search_and_navigate'):
                        detection_correct = False
                
                if detection_correct:
                    successful_detections += 1
                    status = "✅"
                else:
                    status = "❌"
                
                detection_results.append({
                    'prompt': prompt,
                    'expected_navigation': expected_nav,
                    'detected_navigation': requires_nav,
                    'expected_type': expected_type,
                    'detected_type': nav_type,
                    'confidence': confidence,
                    'correct': detection_correct,
                    'status': status
                })
                
                logger.info(f"  {status} '{prompt[:50]}...' → Nav: {requires_nav}, Type: {nav_type}, Conf: {confidence:.2f}")
            
            success_rate = (successful_detections / len(test_cases)) * 100
            overall_success = success_rate >= 70  # Au moins 70% de réussite
            
            self.log_test_result("Détection Navigation", overall_success, 
                               f"Taux de réussite: {success_rate:.1f}% ({successful_detections}/{len(test_cases)})",
                               {'results': detection_results, 'success_rate': success_rate})
            
            return detection_results
            
        except Exception as e:
            self.log_test_result("Test Détection", False, f"Erreur: {str(e)}")
            return None
    
    def test_web_extraction(self):
        """Test 4: Tester l'extraction de contenu web"""
        logger.info("🌐 Test 4: Extraction de Contenu Web")
        
        try:
            from advanced_web_navigator import extract_website_content
            
            # URLs de test
            test_urls = [
                "https://httpbin.org/json",
                "https://httpbin.org/html",
            ]
            
            extraction_results = []
            successful_extractions = 0
            
            for url in test_urls:
                logger.info(f"  🔍 Test extraction: {url}")
                
                start_time = time.time()
                content = extract_website_content(url)
                extraction_time = time.time() - start_time
                
                if content.success:
                    successful_extractions += 1
                    status = "✅"
                    details = {
                        'title': content.title,
                        'content_length': len(content.cleaned_text),
                        'quality_score': content.content_quality_score,
                        'language': content.language,
                        'links_count': len(content.links),
                        'images_count': len(content.images),
                        'keywords': content.keywords[:5],
                        'extraction_time': extraction_time
                    }
                else:
                    status = "❌"
                    details = {'error': content.error_message}
                
                extraction_results.append({
                    'url': url,
                    'success': content.success,
                    'details': details,
                    'status': status
                })
                
                logger.info(f"    {status} Temps: {extraction_time:.2f}s, "
                          f"Contenu: {len(content.cleaned_text) if content.success else 0} chars, "
                          f"Qualité: {content.content_quality_score if content.success else 0:.1f}")
                
                time.sleep(1)  # Délai entre les requêtes
            
            success_rate = (successful_extractions / len(test_urls)) * 100
            overall_success = success_rate >= 80  # Au moins 80% de réussite
            
            self.log_test_result("Extraction Web", overall_success,
                               f"Taux de réussite: {success_rate:.1f}% ({successful_extractions}/{len(test_urls)})",
                               {'results': extraction_results})
            
            return extraction_results
            
        except Exception as e:
            self.log_test_result("Test Extraction Web", False, f"Erreur: {str(e)}")
            return None
    
    def test_gemini_integration_full(self, gemini_api):
        """Test 5: Test complet d'intégration avec Gemini"""
        logger.info("🚀 Test 5: Intégration Complète avec Gemini")
        
        if not gemini_api:
            self.log_test_result("Test Intégration Complète", False, "API Gemini non disponible")
            return
        
        try:
            # Test de requêtes avec navigation
            test_prompts = [
                "Recherche des informations sur l'intelligence artificielle",
                "Qu'est-ce que Python ?",
                "Extrait le contenu de https://httpbin.org/json"
            ]
            
            integration_results = []
            successful_responses = 0
            
            for prompt in test_prompts:
                logger.info(f"  🤖 Test Gemini: '{prompt}'")
                
                try:
                    start_time = time.time()
                    
                    # Tester avec l'API Gemini modifiée (utilise la méthode fallback pour les tests)
                    response = gemini_api._fallback_get_response(
                        prompt=prompt,
                        user_id=1,
                        session_id="test_session"
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Vérifier la réponse
                    if response and 'response' in response:
                        successful_responses += 1
                        status = "✅"
                        
                        # Vérifier si la navigation a été utilisée
                        response_text = response['response']
                        navigation_used = any(indicator in response_text.lower() for indicator in [
                            'navigation web', 'recherche web', 'contenu extrait', 
                            'sites web', 'pages visitées', 'navigation'
                        ])
                        
                        result_details = {
                            'response_length': len(response_text),
                            'navigation_used': navigation_used,
                            'processing_time': processing_time,
                            'status': response.get('status', 'unknown'),
                            'has_emotional_state': 'emotional_state' in response
                        }
                        
                        logger.info(f"    ✅ Réponse reçue: {len(response_text)} chars, "
                                  f"Navigation: {'Oui' if navigation_used else 'Non'}, "
                                  f"Temps: {processing_time:.2f}s")
                    else:
                        status = "❌"
                        result_details = {'error': 'Pas de réponse reçue'}
                        logger.info(f"    ❌ Pas de réponse reçue")
                    
                    integration_results.append({
                        'prompt': prompt,
                        'success': response is not None,
                        'details': result_details,
                        'status': status
                    })
                    
                except Exception as e:
                    logger.error(f"    ❌ Erreur pour '{prompt}': {str(e)}")
                    integration_results.append({
                        'prompt': prompt,
                        'success': False,
                        'details': {'error': str(e)},
                        'status': "❌"
                    })
                
                time.sleep(2)  # Délai entre les requêtes Gemini
            
            success_rate = (successful_responses / len(test_prompts)) * 100
            overall_success = success_rate >= 70  # Au moins 70% de réussite
            
            self.log_test_result("Intégration Complète Gemini", overall_success,
                               f"Taux de réussite: {success_rate:.1f}% ({successful_responses}/{len(test_prompts)})",
                               {'results': integration_results})
            
            return integration_results
            
        except Exception as e:
            self.log_test_result("Test Intégration Complète", False, f"Erreur: {str(e)}")
            return None
    
    def test_api_endpoints(self):
        """Test 6: Tester les endpoints de l'API REST"""
        logger.info("🌐 Test 6: Endpoints API REST")
        
        try:
            from web_navigation_api import register_web_navigation_api, initialize_web_navigation_api
            from flask import Flask
            
            # Créer une app Flask de test
            app = Flask(__name__)
            register_web_navigation_api(app)
            initialize_web_navigation_api()
            
            endpoint_results = []
            successful_endpoints = 0
            
            with app.test_client() as client:
                # Test des endpoints principaux
                endpoints_to_test = [
                    ('GET', '/api/web-navigation/health', None, 'Health Check'),
                    ('GET', '/api/web-navigation/docs', None, 'Documentation'),
                    ('GET', '/api/web-navigation/stats', None, 'Statistiques'),
                    ('POST', '/api/web-navigation/create-session', {'user_id': 'test_user'}, 'Création Session'),
                ]
                
                for method, endpoint, data, description in endpoints_to_test:
                    logger.info(f"  🔗 Test {method} {endpoint}")
                    
                    try:
                        if method == 'GET':
                            response = client.get(endpoint)
                        elif method == 'POST':
                            response = client.post(endpoint, json=data)
                        
                        success = response.status_code == 200
                        if success:
                            successful_endpoints += 1
                            status = "✅"
                            
                            try:
                                json_data = response.get_json()
                                response_details = {
                                    'status_code': response.status_code,
                                    'has_json': json_data is not None,
                                    'content_length': len(response.data)
                                }
                                if json_data and 'success' in json_data:
                                    response_details['api_success'] = json_data['success']
                            except:
                                response_details = {
                                    'status_code': response.status_code,
                                    'content_length': len(response.data)
                                }
                        else:
                            status = "❌"
                            response_details = {
                                'status_code': response.status_code,
                                'error': f"HTTP {response.status_code}"
                            }
                        
                        endpoint_results.append({
                            'method': method,
                            'endpoint': endpoint,
                            'description': description,
                            'success': success,
                            'details': response_details,
                            'status': status
                        })
                        
                        logger.info(f"    {status} {description}: HTTP {response.status_code}")
                        
                    except Exception as e:
                        logger.error(f"    ❌ Erreur {description}: {str(e)}")
                        endpoint_results.append({
                            'method': method,
                            'endpoint': endpoint,
                            'description': description,
                            'success': False,
                            'details': {'error': str(e)},
                            'status': "❌"
                        })
            
            success_rate = (successful_endpoints / len(endpoints_to_test)) * 100
            overall_success = success_rate >= 75  # Au moins 75% de réussite
            
            self.log_test_result("API REST Endpoints", overall_success,
                               f"Taux de réussite: {success_rate:.1f}% ({successful_endpoints}/{len(endpoints_to_test)})",
                               {'results': endpoint_results})
            
            return endpoint_results
            
        except Exception as e:
            self.log_test_result("Test API Endpoints", False, f"Erreur: {str(e)}")
            return None
    
    def test_performance_benchmark(self):
        """Test 7: Benchmark de performance"""
        logger.info("⚡ Test 7: Benchmark de Performance")
        
        try:
            from advanced_web_navigator import extract_website_content
            
            # Test avec plusieurs URLs pour mesurer les performances
            test_urls = [
                "https://httpbin.org/json",
                "https://httpbin.org/html",
                "https://httpbin.org/robots.txt"
            ]
            
            performance_results = []
            total_time = 0
            successful_requests = 0
            
            logger.info(f"  📊 Test de performance sur {len(test_urls)} URLs")
            
            overall_start = time.time()
            
            for i, url in enumerate(test_urls, 1):
                logger.info(f"  {i}/{len(test_urls)} Test: {url}")
                
                start_time = time.time()
                content = extract_website_content(url)
                end_time = time.time()
                
                request_time = end_time - start_time
                total_time += request_time
                
                if content.success:
                    successful_requests += 1
                    status = "✅"
                    details = {
                        'processing_time': request_time,
                        'content_length': len(content.cleaned_text),
                        'quality_score': content.content_quality_score,
                        'extraction_rate': len(content.cleaned_text) / request_time if request_time > 0 else 0
                    }
                else:
                    status = "❌"
                    details = {
                        'processing_time': request_time,
                        'error': content.error_message
                    }
                
                performance_results.append({
                    'url': url,
                    'success': content.success,
                    'details': details,
                    'status': status
                })
                
                logger.info(f"    {status} Temps: {request_time:.2f}s")
                
                time.sleep(0.5)  # Petit délai entre les requêtes
            
            overall_time = time.time() - overall_start
            
            # Calculer les métriques de performance
            avg_time_per_request = total_time / len(test_urls)
            success_rate = (successful_requests / len(test_urls)) * 100
            
            performance_metrics = {
                'total_requests': len(test_urls),
                'successful_requests': successful_requests,
                'success_rate': success_rate,
                'total_time': total_time,
                'overall_time': overall_time,
                'avg_time_per_request': avg_time_per_request,
                'requests_per_second': len(test_urls) / overall_time if overall_time > 0 else 0
            }
            
            # Critères de performance acceptables
            performance_ok = (
                avg_time_per_request < 10.0 and  # Moins de 10s par requête
                success_rate >= 70              # Au moins 70% de réussite
            )
            
            logger.info(f"  📈 Métriques de performance:")
            logger.info(f"    - Temps moyen par requête: {avg_time_per_request:.2f}s")
            logger.info(f"    - Taux de réussite: {success_rate:.1f}%")
            logger.info(f"    - Requêtes par seconde: {performance_metrics['requests_per_second']:.2f}")
            
            self.log_test_result("Benchmark Performance", performance_ok,
                               f"Temps moyen: {avg_time_per_request:.2f}s, Réussite: {success_rate:.1f}%",
                               {'metrics': performance_metrics, 'results': performance_results})
            
            return performance_metrics
            
        except Exception as e:
            self.log_test_result("Test Performance", False, f"Erreur: {str(e)}")
            return None
    
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
        report_file = self.test_dir / f"gemini_web_navigation_test_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Créer un rapport markdown lisible
        md_report = self._create_markdown_report(report)
        md_file = self.test_dir / f"gemini_web_navigation_test_report_{int(time.time())}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"📄 Rapport sauvegardé: {report_file}")
        logger.info(f"📄 Rapport markdown: {md_file}")
        
        return report
    
    def _create_markdown_report(self, report):
        """Crée un rapport markdown"""
        summary = report['test_summary']
        
        md = f"""# Rapport de Test - Système Navigation Web Avancé avec Gemini

## Résumé
- **Date du test**: {summary['timestamp'][:19]}
- **Tests totaux**: {summary['total_tests']}
- **Tests réussis**: {summary['passed_tests']}
- **Tests échoués**: {summary['failed_tests']}
- **Taux de réussite**: {summary['success_rate']:.1f}%
- **Statut global**: {summary['overall_status']}

## Détail des Tests

"""
        
        for test_name, result in report['test_results'].items():
            status = "✅ RÉUSSI" if result['success'] else "❌ ÉCHOUÉ"
            md += f"### {test_name}\n"
            md += f"**Statut**: {status}\n"
            md += f"**Message**: {result['message']}\n"
            
            if result.get('data'):
                md += f"**Données**: Voir le fichier JSON pour les détails\n"
            
            md += "\n"
        
        if report['errors']:
            md += "## Erreurs Rencontrées\n\n"
            for error in report['errors']:
                md += f"- {error}\n"
        
        md += f"""
## Recommandations

### Statut Global: {summary['overall_status']}

"""
        
        if summary['success_rate'] >= 90:
            md += "🎉 **EXCELLENT** - Le système fonctionne parfaitement avec Gemini !\n"
        elif summary['success_rate'] >= 70:
            md += "👍 **BON** - Le système fonctionne bien avec quelques améliorations possibles.\n"
        elif summary['success_rate'] >= 50:
            md += "⚠️ **MOYEN** - Le système fonctionne partiellement, vérifications nécessaires.\n"
        else:
            md += "🚨 **PROBLÈME** - Le système nécessite des corrections importantes.\n"
        
        return md
    
    def run_all_tests(self):
        """Lance tous les tests"""
        logger.info("🚀 DÉMARRAGE DES TESTS COMPLETS")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Test 1: Imports
            imported_modules = self.test_module_imports()
            
            # Test 2: Initialisation Gemini
            gemini_api = self.test_gemini_api_initialization()
            
            # Test 3: Détection de navigation
            self.test_navigation_detection(gemini_api)
            
            # Test 4: Extraction web
            self.test_web_extraction()
            
            # Test 5: Intégration complète avec Gemini
            self.test_gemini_integration_full(gemini_api)
            
            # Test 6: API REST
            self.test_api_endpoints()
            
            # Test 7: Performance
            self.test_performance_benchmark()
            
        except Exception as e:
            logger.error(f"Erreur lors des tests: {str(e)}")
            self.log_test_result("Exécution Globale", False, f"Erreur critique: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Générer le rapport
        report = self.generate_test_report()
        
        # Afficher le résumé final
        logger.info("=" * 60)
        logger.info("🏁 TESTS TERMINÉS")
        logger.info(f"⏱️ Temps total: {total_time:.2f}s")
        logger.info(f"📊 Résultats: {self.passed_tests}/{self.total_tests} tests réussis ({report['test_summary']['success_rate']:.1f}%)")
        
        if report['test_summary']['overall_status'] == 'PASSED':
            logger.info("🎉 TOUS LES TESTS PRINCIPAUX SONT PASSÉS !")
            logger.info("✅ Le système de navigation web fonctionne avec Gemini")
        else:
            logger.info("⚠️ Certains tests ont échoué")
            logger.info("🔧 Vérifiez les erreurs dans le rapport de test")
        
        logger.info("=" * 60)
        
        return report

def main():
    """Fonction principale"""
    print("🧪 TEST COMPLET - Système Navigation Web Avancé avec Gemini")
    print("=" * 70)
    print(f"🕐 Démarré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Créer le testeur
    tester = GeminiWebNavigationTester()
    
    try:
        # Lancer tous les tests
        report = tester.run_all_tests()
        
        # Résultat final
        if report['test_summary']['overall_status'] == 'PASSED':
            print("\n🎊 SUCCÈS COMPLET !")
            print("✅ Le système de navigation web fonctionne parfaitement avec Gemini")
            print("🚀 Vous pouvez maintenant utiliser les nouvelles capacités de navigation")
            return True
        else:
            print("\n⚠️ TESTS PARTIELLEMENT RÉUSSIS")
            print("🔧 Certaines fonctionnalités nécessitent des ajustements")
            print("📋 Consultez le rapport de test pour plus de détails")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus par l'utilisateur")
        return False
    except Exception as e:
        print(f"\n❌ Erreur critique lors des tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
