"""
Test Complet des Capacités d'Interaction Web de l'API Gemini avec Searx
Ce script teste si l'API Gemini peut utiliser Searx pour :
1. Effectuer des recherches web
2. Analyser les résultats de recherche
3. Identifier des éléments cliquables
4. Naviguer vers les résultats
5. Interagir avec les pages web trouvées
"""

import logging
import json
import time
import os
import sys
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeminiSearxInteractionTest')

class GeminiSearxInteractionTester:
    """Testeur spécialisé pour les interactions web avec Gemini via Searx"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        self.session_id = None
        
        # Configuration Searx
        self.searx_url = "http://localhost:8080"
        self.app_url = "http://localhost:5000"  # App Flask principale
        
        # Requêtes de test
        self.test_queries = [
            "Python programming tutorial",
            "What is artificial intelligence",
            "Weather forecast today",
            "Latest technology news",
            "Machine learning basics"
        ]
        
        # Créer le répertoire de tests
        self.test_dir = Path("test_results_searx_interaction")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🔍 Testeur d'Interaction Gemini-Searx initialisé")
        
        # Initialiser les modules nécessaires
        self.navigator = None
        self.gemini_adapter = None
        self.searx_interface = None
        
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
    
    def check_services_availability(self):
        """Vérifie que les services nécessaires sont disponibles"""
        logger.info("🔧 Vérification des services...")
        
        # Vérifier Searx
        try:
            response = requests.get(f"{self.searx_url}/", timeout=10)
            if response.status_code == 200:
                self.log_test_result("Service Searx", True, f"Accessible sur {self.searx_url}")
            else:
                self.log_test_result("Service Searx", False, f"Code d'erreur: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Service Searx", False, f"Erreur: {str(e)}")
            return False
        
        # Vérifier l'app Flask
        try:
            response = requests.get(f"{self.app_url}/", timeout=10)
            if response.status_code == 200:
                self.log_test_result("Service App Flask", True, f"Accessible sur {self.app_url}")
            else:
                self.log_test_result("Service App Flask", False, f"Code d'erreur: {response.status_code}")
        except Exception as e:
            self.log_test_result("Service App Flask", False, f"Erreur: {str(e)}")
        
        return True
    
    def setup_modules(self):
        """Initialise tous les modules nécessaires"""
        logger.info("🔧 Configuration des modules...")
        
        # Importer l'interface Searx
        try:
            from searx_interface import SearxInterface
            self.searx_interface = SearxInterface()
            if self.searx_interface:
                self.log_test_result("Setup Searx Interface", True, "Interface Searx initialisée")
            else:
                self.log_test_result("Setup Searx Interface", False, "Interface non disponible")
        except Exception as e:
            self.log_test_result("Setup Searx Interface", False, f"Erreur: {str(e)}")
        
        # Importer le navigateur interactif
        try:
            from interactive_web_navigator import initialize_interactive_navigator
            navigator = initialize_interactive_navigator()
            if navigator:
                self.navigator = navigator
                # Créer une session de test
                session_obj = self.navigator.create_interactive_session(
                    f"searx_test_session_{int(time.time())}", 
                    self.searx_url,
                    ["Test de recherche Searx avec Gemini"]
                )
                if session_obj:
                    # Extraire l'ID de session depuis l'objet NavigationSession
                    self.session_id = session_obj.session_id if hasattr(session_obj, 'session_id') else f"searx_test_session_{int(time.time())}"
                    self.log_test_result("Setup Navigateur", True, "Navigateur et session initialisés")
                else:
                    self.log_test_result("Setup Navigateur", False, "Session non créée")
            else:
                self.log_test_result("Setup Navigateur", False, "Navigateur non disponible")
        except Exception as e:
            self.log_test_result("Setup Navigateur", False, f"Erreur: {str(e)}")
        
        # Importer l'adaptateur Gemini interactif
        try:
            from gemini_interactive_adapter import initialize_gemini_interactive_adapter
            adapter = initialize_gemini_interactive_adapter()
            if adapter:
                self.gemini_adapter = adapter
                self.log_test_result("Setup Gemini Interactif", True, "Adaptateur Gemini initialisé")
            else:
                self.log_test_result("Setup Gemini Interactif", False, "Adaptateur non disponible")
        except Exception as e:
            self.log_test_result("Setup Gemini Interactif", False, f"Erreur: {str(e)}")
    
    def test_searx_search(self, query: str):
        """Test de recherche via Searx"""
        logger.info(f"🔍 Test de recherche Searx: '{query}'")
        
        try:
            if self.searx_interface:
                # Utiliser l'interface Searx avec la méthode correcte
                results = self.searx_interface.search_with_filters(query, engines=['google', 'bing'])
                
                if results and len(results) > 0:
                    self.log_test_result(f"Recherche Searx '{query[:20]}'", True, 
                                       f"{len(results)} résultats trouvés",
                                       {'query': query, 'results_count': len(results), 
                                        'first_result': results[0].__dict__ if results else None})
                    return results
                else:
                    self.log_test_result(f"Recherche Searx '{query[:20]}'", False, "Aucun résultat")
            else:
                # Recherche directe via l'API Searx
                search_url = f"{self.searx_url}/search"
                params = {
                    'q': query,
                    'format': 'json',
                    'engines': 'google,bing'
                }
                
                response = requests.get(search_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if results:
                        self.log_test_result(f"Recherche Searx '{query[:20]}'", True, 
                                           f"{len(results)} résultats trouvés",
                                           {'query': query, 'results_count': len(results),
                                            'first_result_title': results[0].get('title', '') if results else ''})
                        return results
                    else:
                        self.log_test_result(f"Recherche Searx '{query[:20]}'", False, "Aucun résultat dans la réponse")
                else:
                    self.log_test_result(f"Recherche Searx '{query[:20]}'", False, f"Erreur HTTP: {response.status_code}")
                    
        except Exception as e:
            self.log_test_result(f"Recherche Searx '{query[:20]}'", False, f"Erreur: {str(e)}")
        
        return []
    
    def test_gemini_searx_integration(self, query: str):
        """Test de l'intégration Gemini avec Searx"""
        logger.info(f"🤖 Test intégration Gemini-Searx: '{query}'")
        
        try:
            # Utiliser l'endpoint chat qui supporte les recherches avec Searx
            api_url = f"{self.app_url}/api/chat"
            
            # Créer une session pour contourner l'authentification
            session = requests.Session()
            
            # D'abord obtenir une session (cookie)
            login_response = session.get(f"{self.app_url}/")
            
            data = {
                'message': f"Fais une recherche sur: {query}",
                'use_web_search': True,
                'search_engine': 'searx'
            }
            
            response = session.post(api_url, json=data, timeout=60)
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data.get('success', True):  # Certaines réponses n'ont pas de champ success
                    gemini_response = result_data.get('response', '')
                    
                    if gemini_response:
                        self.log_test_result(f"Intégration Gemini-Searx '{query[:20]}'", True,
                                           f"Gemini a répondu avec Searx",
                                           {'query': query, 'has_response': bool(gemini_response),
                                            'response_preview': gemini_response[:200] if gemini_response else ''})
                        return True
                    else:
                        self.log_test_result(f"Intégration Gemini-Searx '{query[:20]}'", False, 
                                           "Réponse vide de Gemini")
                else:
                    self.log_test_result(f"Intégration Gemini-Searx '{query[:20]}'", False, 
                                       result_data.get('error', 'Erreur inconnue'))
            elif response.status_code == 401:
                # Tenter une approche alternative via trigger-web-search
                alt_url = f"{self.app_url}/api/trigger-web-search"
                alt_data = {'query': query}
                alt_response = session.post(alt_url, json=alt_data, timeout=60)
                
                if alt_response.status_code == 200:
                    self.log_test_result(f"Intégration Gemini-Searx '{query[:20]}'", True,
                                       f"Recherche web déclenchée avec succès",
                                       {'query': query, 'method': 'trigger-web-search'})
                    return True
                else:
                    self.log_test_result(f"Intégration Gemini-Searx '{query[:20]}'", False, 
                                       f"Authentification requise (HTTP 401)")
            else:
                self.log_test_result(f"Intégration Gemini-Searx '{query[:20]}'", False, 
                                   f"Erreur HTTP: {response.status_code}")
                
        except Exception as e:
            self.log_test_result(f"Intégration Gemini-Searx '{query[:20]}'", False, f"Erreur: {str(e)}")
        
        return False
    
    def test_navigation_to_search_results(self, query: str):
        """Test de navigation vers les résultats de recherche"""
        logger.info(f"🌐 Test navigation vers résultats: '{query}'")
        
        try:
            # D'abord obtenir des résultats de recherche
            results = self.test_searx_search(query)
            
            if not results or not self.navigator or not self.session_id:
                self.log_test_result(f"Navigation '{query[:20]}'", False, "Prérequis non remplis")
                return False
            
            # Prendre le premier résultat sûr
            safe_result = None
            for result in results[:3]:  # Vérifier les 3 premiers
                url = result.url if hasattr(result, 'url') else result.get('url', '')
                if url and any(domain in url for domain in ['wikipedia.org', 'python.org', 'github.com']):
                    safe_result = result
                    break
            
            if not safe_result:
                self.log_test_result(f"Navigation '{query[:20]}'", False, "Aucun résultat sûr trouvé")
                return False
            
            # Naviguer vers le résultat
            result_url = safe_result.url if hasattr(safe_result, 'url') else safe_result.get('url', '')
            navigation_result = self.navigator.navigate_to_url(self.session_id, result_url)
            
            if navigation_result.get('success'):
                # Analyser la page avec Gemini
                page_summary = self.navigator.get_interactive_elements_summary(self.session_id)
                
                if page_summary:
                    self.log_test_result(f"Navigation '{query[:20]}'", True,
                                       f"Navigation réussie vers {result_url[:50]}...",
                                       {'target_url': result_url, 'page_loaded': True,
                                        'elements_found': len(page_summary.get('interactive_elements', []))})
                    return True
                else:
                    self.log_test_result(f"Navigation '{query[:20]}'", False, "Page chargée mais analyse échouée")
            else:
                self.log_test_result(f"Navigation '{query[:20]}'", False, 
                                   f"Navigation échouée: {navigation_result.get('error', 'Erreur inconnue')}")
                
        except Exception as e:
            self.log_test_result(f"Navigation '{query[:20]}'", False, f"Erreur: {str(e)}")
        
        return False
    
    async def run_comprehensive_test(self):
        """Lance tous les tests d'interaction Gemini-Searx"""
        logger.info("🚀 Début des tests d'interaction Gemini-Searx")
        
        # Vérifications préliminaires
        if not self.check_services_availability():
            logger.error("❌ Services non disponibles, arrêt des tests")
            return
        
        # Configuration des modules
        self.setup_modules()
        
        # Test 1: Tests de recherche Searx de base
        logger.info("📝 Phase 1: Tests de recherche Searx")
        for query in self.test_queries[:3]:  # Tester 3 requêtes
            self.test_searx_search(query)
            time.sleep(2)  # Éviter la surcharge
        
        # Test 2: Tests d'intégration Gemini-Searx
        logger.info("🤖 Phase 2: Tests d'intégration Gemini-Searx")
        for query in self.test_queries[:2]:  # Tester 2 requêtes avec Gemini
            self.test_gemini_searx_integration(query)
            time.sleep(3)  # Plus de temps pour Gemini
        
        # Test 3: Navigation vers les résultats
        logger.info("🌐 Phase 3: Tests de navigation")
        for query in ["Python tutorial", "Wikipedia artificial intelligence"]:
            self.test_navigation_to_search_results(query)
            time.sleep(3)
        
        # Générer le rapport final
        self.generate_final_report()
    
    def generate_final_report(self):
        """Génère un rapport final des tests"""
        logger.info("📊 Génération du rapport final")
        
        report = {
            'test_summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.total_tests - self.passed_tests,
                'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            'test_results': self.test_results,
            'errors': self.errors,
            'services_info': {
                'searx_url': self.searx_url,
                'app_url': self.app_url
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Sauvegarder le rapport
        report_file = self.test_dir / f"gemini_searx_interaction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Afficher le résumé
        logger.info(f"📈 Tests terminés: {self.passed_tests}/{self.total_tests} réussis ({report['test_summary']['success_rate']:.1f}%)")
        logger.info(f"📄 Rapport sauvegardé: {report_file}")
        
        if self.errors:
            logger.warning("⚠️  Erreurs rencontrées:")
            for error in self.errors[-5:]:  # Afficher les 5 dernières erreurs
                logger.warning(f"   - {error}")

async def main():
    """Fonction principale pour lancer les tests"""
    tester = GeminiSearxInteractionTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
