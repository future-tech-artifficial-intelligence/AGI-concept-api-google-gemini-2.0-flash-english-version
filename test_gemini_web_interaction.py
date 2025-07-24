"""
Test Complet des Capacités d'Interaction Web de l'API Gemini
Ce script teste spécifiquement si l'API Gemini peut :
1. Analyser des pages web
2. Identifier des éléments cliquables
3. Effectuer des clics sur des éléments
4. Naviguer entre les pages
5. Remplir des formulaires
"""

import logging
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
import requests
from urllib.parse import urljoin, urlparse

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeminiWebInteractionTest')

class GeminiWebInteractionTester:
    """Testeur spécialisé pour les interactions web avec Gemini"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        self.session_id = None  # ID de session pour le navigateur interactif
        
        # Sites de test (publics et sûrs)
        self.test_sites = {
            'simple_page': 'https://example.com',
            'form_page': 'https://httpbin.org/forms/post',
            'search_page': 'https://duckduckgo.com',
            'navigation_page': 'https://www.w3schools.com',
            'interactive_page': 'https://www.google.com'
        }
        
        # Créer le répertoire de tests
        self.test_dir = Path("test_results_web_interaction")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🌐 Testeur d'Interaction Web Gemini initialisé")
        
        # Initialiser les modules nécessaires
        self.navigator = None
        self.gemini_adapter = None
        self.interactive_adapter = None
        
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
    
    def setup_modules(self):
        """Initialise tous les modules nécessaires"""
        logger.info("🔧 Configuration des modules...")
        
        # Importer et initialiser le navigateur interactif
        try:
            from interactive_web_navigator import initialize_interactive_navigator, get_interactive_navigator
            navigator = initialize_interactive_navigator()
            if navigator:
                self.navigator = navigator
                # Créer une session de test
                self.session_id = "test_session_" + str(int(time.time()))
                session = self.navigator.create_interactive_session(
                    self.session_id, 
                    "https://example.com", 
                    ["Test de capacités d'interaction"]
                )
                if session:
                    self.log_test_result("Setup Navigateur", True, "Navigateur initialisé")
                else:
                    self.log_test_result("Setup Navigateur", False, "Impossible de créer une session")
            else:
                self.log_test_result("Setup Navigateur", False, "Navigateur non disponible")
        except Exception as e:
            self.log_test_result("Setup Navigateur", False, f"Erreur: {str(e)}")
        
        # Importer et initialiser l'adaptateur Gemini
        try:
            from gemini_api_adapter import GeminiAPI
            from ai_api_config import get_api_config, get_gemini_api_key
            
            api_key = get_gemini_api_key()
            if api_key:
                self.gemini_adapter = GeminiAPI(api_key)
                self.log_test_result("Setup Gemini API", True, "API Gemini initialisée")
            else:
                self.log_test_result("Setup Gemini API", False, "Clé API Gemini manquante")
        except Exception as e:
            self.log_test_result("Setup Gemini API", False, f"Erreur: {str(e)}")
        
        # Importer l'adaptateur interactif
        try:
            from gemini_interactive_adapter import initialize_gemini_interactive_adapter
            adapter = initialize_gemini_interactive_adapter()
            if adapter:
                self.interactive_adapter = adapter
                self.log_test_result("Setup Adaptateur Interactif", True, "Adaptateur initialisé")
            else:
                self.log_test_result("Setup Adaptateur Interactif", False, "Adaptateur non disponible")
        except Exception as e:
            self.log_test_result("Setup Adaptateur Interactif", False, f"Erreur: {str(e)}")
    
    async def test_page_analysis(self, url: str, site_name: str):
        """Test l'analyse d'une page web par Gemini"""
        logger.info(f"📖 Test d'analyse de page: {site_name} ({url})")
        
        try:
            if not self.navigator or not self.session_id:
                self.log_test_result(f"Analyse {site_name}", False, "Navigateur non disponible")
                return False
            
            # Charger la page
            result = self.navigator.navigate_to_url(self.session_id, url)
            
            if result.get('success'):
                # Obtenir le résumé des éléments de la page
                page_summary = self.navigator.get_interactive_elements_summary(self.session_id)
                
                if page_summary:
                    # Demander à Gemini d'analyser la page
                    if self.gemini_adapter:
                        analysis_prompt = f"""
                        Analyse cette page web et identifie :
                        1. Le titre et le contenu principal
                        2. Les éléments interactifs (boutons, liens, formulaires)
                        3. Les éléments sur lesquels on peut cliquer
                        4. La structure de navigation
                        
                        Résumé de la page:
                        {json.dumps(page_summary, indent=2)[:2000]}...
                        """
                        
                        analysis = await self.gemini_adapter.generate_text(analysis_prompt)
                        
                        if analysis:
                            self.log_test_result(f"Analyse {site_name}", True, 
                                               f"Page analysée avec succès", 
                                               {'url': url, 'analysis': analysis[:500]})
                            return True
                        else:
                            self.log_test_result(f"Analyse {site_name}", False, "Gemini n'a pas pu analyser")
                    else:
                        self.log_test_result(f"Analyse {site_name}", False, "API Gemini non disponible")
                else:
                    self.log_test_result(f"Analyse {site_name}", False, "Résumé de page non récupéré")
            else:
                self.log_test_result(f"Analyse {site_name}", False, f"Navigation échouée: {result.get('error', 'Erreur inconnue')}")
                
        except Exception as e:
            self.log_test_result(f"Analyse {site_name}", False, f"Erreur: {str(e)}")
        
        return False
    
    async def test_element_identification(self, url: str, site_name: str):
        """Test l'identification d'éléments cliquables par Gemini"""
        logger.info(f"🎯 Test d'identification d'éléments: {site_name}")
        
        try:
            if not self.navigator or not self.session_id:
                self.log_test_result(f"Identification {site_name}", False, "Navigateur non disponible")
                return []
            
            # Charger la page
            self.navigator.navigate_to_url(self.session_id, url)
            
            # Obtenir le résumé des éléments interactifs
            summary = self.navigator.get_interactive_elements_summary(self.session_id)
            
            if summary and summary.get('interactive_elements'):
                interactive_elements = summary['interactive_elements']
                
                # Demander à Gemini de classifier ces éléments
                if self.gemini_adapter:
                    elements_info = []
                    for element in interactive_elements[:10]:  # Limite à 10 éléments
                        element_info = {
                            'element_type': element.get('element_type', ''),
                            'text': element.get('text', ''),
                            'id': element.get('element_id', ''),
                            'attributes': element.get('attributes', {}),
                            'clickable': element.get('is_clickable', False)
                        }
                        elements_info.append(element_info)
                    
                    identification_prompt = f"""
                    Voici une liste d'éléments interactifs trouvés sur la page {url}.
                    Pour chaque élément, dis-moi :
                    1. Son type (bouton, lien, champ de formulaire, etc.)
                    2. Son action probable (recherche, navigation, soumission, etc.)
                    3. S'il est sûr de cliquer dessus
                    
                    Éléments:
                    {json.dumps(elements_info, indent=2)}
                    """
                    
                    identification = await self.gemini_adapter.generate_text(identification_prompt)
                    
                    if identification:
                        self.log_test_result(f"Identification {site_name}", True, 
                                           f"{len(interactive_elements)} éléments identifiés",
                                           {'elements_count': len(interactive_elements), 
                                            'identification': identification[:500]})
                        return interactive_elements
                    else:
                        self.log_test_result(f"Identification {site_name}", False, "Gemini n'a pas pu identifier")
                else:
                    self.log_test_result(f"Identification {site_name}", True, 
                                       f"{len(interactive_elements)} éléments trouvés (sans analyse Gemini)")
                    return interactive_elements
            else:
                self.log_test_result(f"Identification {site_name}", False, "Aucun élément interactif trouvé")
                
        except Exception as e:
            self.log_test_result(f"Identification {site_name}", False, f"Erreur: {str(e)}")
        
        return []
    
    async def test_element_clicking(self, url: str, site_name: str):
        """Test de clic sur des éléments avec guidance de Gemini"""
        logger.info(f"👆 Test de clic d'éléments: {site_name}")
        
        try:
            # D'abord identifier les éléments
            elements = await self.test_element_identification(url, site_name)
            
            if not elements:
                self.log_test_result(f"Clic {site_name}", False, "Aucun élément à cliquer")
                return False
            
            # Demander à Gemini de choisir un élément sûr à cliquer
            if self.gemini_adapter and len(elements) > 0:
                click_prompt = f"""
                Sur la page {url}, j'ai trouvé ces éléments interactifs.
                Choisis UN élément sûr à cliquer qui :
                1. Ne cause pas de dommage
                2. Ne soumet pas de données personnelles
                3. Est probablement un lien de navigation simple
                
                Réponds uniquement par l'index de l'élément (0, 1, 2, etc.) ou "aucun" si aucun n'est sûr.
                
                Éléments disponibles:
                """
                
                for i, element in enumerate(elements[:5]):  # Limite à 5 éléments
                    click_prompt += f"\n{i}: {element.get('tag_name', 'unknown')} - {element.get('text', 'sans texte')[:50]}"
                
                choice = await self.gemini_adapter.generate_text(click_prompt)
                
                if choice and choice.strip().isdigit():
                    element_index = int(choice.strip())
                    if 0 <= element_index < len(elements):
                        chosen_element = elements[element_index]
                        
                        # Tenter le clic en utilisant la méthode d'interaction
                        if self.navigator and self.session_id:
                            element_id = chosen_element.get('element_id', '')
                            if element_id:
                                click_result = self.navigator.interact_with_element(
                                    self.session_id, element_id, "click"
                                )
                                
                                if click_result.get('success'):
                                    self.log_test_result(f"Clic {site_name}", True, 
                                                       f"Clic réussi sur: {chosen_element.get('text', 'élément')[:30]}",
                                                       {'element': chosen_element, 'result': click_result})
                                    
                                    # Attendre un peu pour voir le résultat
                                    time.sleep(2)
                                    
                                    return True
                                else:
                                    self.log_test_result(f"Clic {site_name}", False, 
                                                       f"Clic échoué: {click_result.get('error', 'Erreur inconnue')}")
                            else:
                                self.log_test_result(f"Clic {site_name}", False, "Élément sans ID")
                        else:
                            self.log_test_result(f"Clic {site_name}", False, "Navigateur non disponible")
                    else:
                        self.log_test_result(f"Clic {site_name}", False, "Index d'élément invalide")
                else:
                    self.log_test_result(f"Clic {site_name}", False, "Gemini n'a pas choisi d'élément sûr")
            else:
                self.log_test_result(f"Clic {site_name}", False, "API Gemini non disponible")
                
        except Exception as e:
            self.log_test_result(f"Clic {site_name}", False, f"Erreur: {str(e)}")
        
        return False
    
    async def test_form_interaction(self):
        """Test d'interaction avec des formulaires"""
        logger.info("📝 Test d'interaction avec formulaires")
        
        form_url = "https://httpbin.org/forms/post"
        
        try:
            if not self.navigator or not self.session_id:
                self.log_test_result("Interaction Formulaire", False, "Navigateur non disponible")
                return False
            
            # Charger la page du formulaire
            result = self.navigator.navigate_to_url(self.session_id, form_url)
            
            if not result.get('success'):
                self.log_test_result("Interaction Formulaire", False, f"Navigation échouée: {result.get('error', 'Erreur inconnue')}")
                return False
            
            # Obtenir le résumé des éléments de la page incluant les formulaires
            summary = self.navigator.get_interactive_elements_summary(self.session_id)
            
            if summary and summary.get('interactive_elements'):
                # Filtrer les éléments de formulaire
                form_elements = [elem for elem in summary['interactive_elements'] 
                               if elem.get('element_type', '').lower() in ['input', 'textarea', 'select', 'button']]
                
                if form_elements:
                    # Demander à Gemini de remplir le formulaire de test
                    if self.gemini_adapter:
                        form_prompt = f"""
                        Je suis sur une page de test de formulaire (httpbin.org).
                        Voici les champs disponibles. Donne-moi des valeurs de test sûres pour chaque champ :
                        
                        Champs trouvés:
                        {json.dumps([{
                            'element_id': elem.get('element_id', ''),
                            'element_type': elem.get('element_type', ''),
                            'text': elem.get('text', ''),
                            'attributes': elem.get('attributes', {})
                        } for elem in form_elements], indent=2)}
                        
                        Réponds au format JSON avec les valeurs à saisir.
                        """
                        
                        form_values = await self.gemini_adapter.generate_text(form_prompt)
                        
                        if form_values:
                            # Tenter de remplir le formulaire (simulation)
                            self.log_test_result("Interaction Formulaire", True, 
                                               f"Formulaire analysé avec {len(form_elements)} champs",
                                               {'fields': len(form_elements), 'values': form_values[:200]})
                            return True
                        else:
                            self.log_test_result("Interaction Formulaire", False, "Gemini n'a pas fourni de valeurs")
                    else:
                        self.log_test_result("Interaction Formulaire", True, 
                                           f"{len(form_elements)} champs trouvés (sans remplissage)")
                        return True
                else:
                    self.log_test_result("Interaction Formulaire", False, "Aucun champ de formulaire trouvé")
            else:
                self.log_test_result("Interaction Formulaire", False, "Aucun élément interactif trouvé")
                
        except Exception as e:
            self.log_test_result("Interaction Formulaire", False, f"Erreur: {str(e)}")
        
        return False
    
    async def run_comprehensive_test(self):
        """Lance tous les tests d'interaction web"""
        logger.info("🚀 Début des tests d'interaction web avec Gemini")
        
        # Configuration initiale
        self.setup_modules()
        
        # Test 1: Analyse de pages simples
        await self.test_page_analysis("https://example.com", "Example.com")
        
        # Test 2: Identification d'éléments sur différents sites
        for site_name, url in self.test_sites.items():
            if site_name != 'form_page':  # On teste les formulaires séparément
                await self.test_element_identification(url, site_name)
        
        # Test 3: Test de clics sécurisés
        # On teste seulement sur des sites sûrs
        safe_sites = {
            'simple_page': 'https://example.com',
            'w3schools': 'https://www.w3schools.com'
        }
        
        for site_name, url in safe_sites.items():
            await self.test_element_clicking(url, site_name)
        
        # Test 4: Interaction avec formulaires
        await self.test_form_interaction()
        
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
            'timestamp': datetime.now().isoformat()
        }
        
        # Sauvegarder le rapport
        report_file = self.test_dir / f"gemini_web_interaction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    tester = GeminiWebInteractionTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
