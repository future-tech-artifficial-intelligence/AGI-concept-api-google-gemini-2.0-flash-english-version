"""
Intégration du Système de Navigation Web Avancé avec Searx et Gemini
Ce module connecte le navigateur avancé avec l'API Gemini et Searx
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from urllib.parse import urljoin, urlparse

from advanced_web_navigator import AdvancedWebNavigator, WebPageContent, NavigationPath

# Configuration du logging
logger = logging.getLogger('GeminiWebIntegration')

class GeminiWebNavigationIntegration:
    """Intégration navigation web pour l'API Gemini"""
    
    def __init__(self, searx_interface=None):
        self.navigator = AdvancedWebNavigator()
        self.searx_interface = searx_interface
        
        # Configuration
        self.max_search_results = 5
        self.max_navigation_depth = 3
        self.max_pages_per_site = 8
        self.content_quality_threshold = 3.0
        
        # Cache des recherches récentes
        self.search_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        # Répertoire pour les rapports Gemini
        self.reports_dir = Path("data/gemini_web_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Intégration Gemini-Navigation initialisée")
    
    def search_and_navigate_for_gemini(self, query: str, user_context: str = "") -> Dict[str, Any]:
        """
        Effectue une recherche et navigation complète pour Gemini
        
        Args:
            query: Requête de recherche
            user_context: Contexte utilisateur pour personnaliser la recherche
            
        Returns:
            Dictionnaire avec contenu structuré pour Gemini
        """
        search_id = f"search_{int(time.time())}"
        logger.info(f"🔍 Recherche Gemini: {query} (ID: {search_id})")
        
        # Vérifier le cache
        cache_key = f"{query}_{hash(user_context)}"
        if cache_key in self.search_cache:
            cached_result, cached_time = self.search_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.info("📋 Résultat récupéré du cache")
                return cached_result
        
        try:
            # Phase 1: Recherche avec Searx
            search_results = self._perform_searx_search(query)
            
            if not search_results:
                logger.warning("⚠️ Aucun résultat de recherche")
                return self._create_empty_result(query, "Aucun résultat trouvé")
            
            # Phase 2: Navigation dans les résultats
            navigation_results = []
            total_content_extracted = 0
            
            for i, search_result in enumerate(search_results[:self.max_search_results]):
                try:
                    logger.info(f"🚀 Navigation site {i+1}: {search_result['url']}")
                    
                    # Navigation en profondeur
                    nav_path = self.navigator.navigate_deep(
                        start_url=search_result['url'],
                        max_depth=self.max_navigation_depth,
                        max_pages=self.max_pages_per_site,
                        navigation_strategy='quality_first',
                        content_filter=self._quality_content_filter
                    )
                    
                    if nav_path.visited_pages:
                        navigation_results.append({
                            'search_result': search_result,
                            'navigation_path': nav_path,
                            'pages_extracted': len(nav_path.visited_pages),
                            'content_length': nav_path.total_content_extracted
                        })
                        total_content_extracted += nav_path.total_content_extracted
                        
                        logger.info(f"✅ Site navigué: {len(nav_path.visited_pages)} pages, {nav_path.total_content_extracted} caractères")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur navigation site {search_result['url']}: {str(e)}")
                    continue
            
            # Phase 3: Synthèse pour Gemini
            gemini_report = self._create_gemini_report(
                query=query,
                user_context=user_context,
                search_results=search_results,
                navigation_results=navigation_results,
                search_id=search_id
            )
            
            # Mettre en cache
            self.search_cache[cache_key] = (gemini_report, datetime.now())
            
            # Sauvegarder le rapport
            self._save_gemini_report(gemini_report, search_id)
            
            logger.info(f"🎯 Recherche terminée: {len(navigation_results)} sites navigués, {total_content_extracted} caractères extraits")
            
            return gemini_report
            
        except Exception as e:
            logger.error(f"❌ Erreur dans la recherche Gemini: {str(e)}")
            return self._create_error_result(query, str(e))
    
    def extract_specific_content(self, url: str, content_requirements: List[str]) -> Dict[str, Any]:
        """
        Extrait du contenu spécifique d'une URL selon les exigences
        
        Args:
            url: URL à analyser
            content_requirements: Liste des types de contenu requis
                                ['summary', 'details', 'links', 'images', 'structure']
        """
        logger.info(f"🎯 Extraction spécifique: {url}")
        
        try:
            # Extraction du contenu
            page_content = self.navigator.extract_page_content(url)
            
            if not page_content.success:
                return {
                    'success': False,
                    'error': page_content.error_message,
                    'url': url
                }
            
            # Préparer la réponse selon les exigences
            extracted_content = {
                'success': True,
                'url': url,
                'title': page_content.title,
                'extraction_timestamp': page_content.extraction_timestamp.isoformat(),
                'content_quality_score': page_content.content_quality_score,
                'language': page_content.language
            }
            
            # Ajouter le contenu selon les exigences
            if 'summary' in content_requirements:
                extracted_content['summary'] = page_content.summary
            
            if 'details' in content_requirements:
                extracted_content['main_content'] = page_content.main_content
                extracted_content['cleaned_text'] = page_content.cleaned_text[:2000]  # Limite pour Gemini
            
            if 'links' in content_requirements:
                extracted_content['links'] = page_content.links[:20]  # Top 20 liens
            
            if 'images' in content_requirements:
                extracted_content['images'] = page_content.images[:10]  # Top 10 images
            
            if 'structure' in content_requirements:
                extracted_content['content_sections'] = page_content.content_sections
                extracted_content['keywords'] = page_content.keywords
            
            if 'navigation' in content_requirements:
                extracted_content['navigation_elements'] = page_content.navigation_elements
            
            if 'metadata' in content_requirements:
                extracted_content['metadata'] = page_content.metadata
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction spécifique {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def navigate_user_journey(self, start_url: str, user_intent: str) -> Dict[str, Any]:
        """
        Simule un parcours utilisateur sur un site selon son intention
        
        Args:
            start_url: URL de départ
            user_intent: Intention de l'utilisateur ('buy', 'learn', 'contact', 'explore')
        """
        logger.info(f"👤 Parcours utilisateur: {user_intent} depuis {start_url}")
        
        try:
            # Configuration selon l'intention
            intent_config = {
                'buy': {
                    'keywords': ['prix', 'acheter', 'commander', 'panier', 'produit'],
                    'max_depth': 4,
                    'max_pages': 15
                },
                'learn': {
                    'keywords': ['guide', 'tutoriel', 'formation', 'cours', 'apprendre'],
                    'max_depth': 3,
                    'max_pages': 10
                },
                'contact': {
                    'keywords': ['contact', 'support', 'aide', 'téléphone', 'email'],
                    'max_depth': 2,
                    'max_pages': 8
                },
                'explore': {
                    'keywords': ['voir', 'découvrir', 'plus', 'détail', 'information'],
                    'max_depth': 3,
                    'max_pages': 12
                }
            }
            
            config = intent_config.get(user_intent, intent_config['explore'])
            
            # Navigation avec filtre d'intention
            def intent_filter(page_content: WebPageContent) -> bool:
                text_lower = page_content.cleaned_text.lower()
                title_lower = page_content.title.lower()
                
                # Vérifier la présence de mots-clés d'intention
                keyword_score = sum(1 for keyword in config['keywords'] 
                                  if keyword in text_lower or keyword in title_lower)
                
                return keyword_score > 0 and page_content.content_quality_score >= 2.0
            
            # Navigation
            nav_path = self.navigator.navigate_deep(
                start_url=start_url,
                max_depth=config['max_depth'],
                max_pages=config['max_pages'],
                navigation_strategy='quality_first',
                content_filter=intent_filter
            )
            
            # Analyser le parcours
            journey_analysis = self._analyze_user_journey(nav_path, user_intent, config['keywords'])
            
            return {
                'success': True,
                'start_url': start_url,
                'user_intent': user_intent,
                'pages_visited': len(nav_path.visited_pages),
                'journey_analysis': journey_analysis,
                'navigation_path': {
                    'session_id': nav_path.session_id,
                    'total_content': nav_path.total_content_extracted,
                    'navigation_depth': nav_path.navigation_depth
                },
                'key_pages': self._extract_key_pages(nav_path, config['keywords'])
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur parcours utilisateur: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'start_url': start_url,
                'user_intent': user_intent
            }
    
    def _perform_searx_search(self, query: str) -> List[Dict[str, Any]]:
        """Effectue une recherche avec Searx"""
        if not self.searx_interface:
            logger.warning("⚠️ Interface Searx non disponible, utilisation de résultats simulés")
            return self._simulate_search_results(query)
        
        try:
            # Utiliser l'interface Searx
            search_results = self.searx_interface.search(query, categories=['general'], max_results=10)
            
            # Convertir au format attendu
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'title': result.title,
                    'url': result.url,
                    'content': result.content,
                    'engine': result.engine,
                    'score': result.score
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche Searx: {str(e)}")
            return self._simulate_search_results(query)
    
    def _simulate_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Simule des résultats de recherche (fallback)"""
        # Résultats simulés basés sur la requête
        base_results = [
            {
                'title': f'Résultat 1 pour {query}',
                'url': 'https://fr.wikipedia.org/wiki/Intelligence_artificielle',
                'content': f'Information détaillée sur {query}',
                'engine': 'wikipedia',
                'score': 0.9
            },
            {
                'title': f'Guide complet sur {query}',
                'url': 'https://www.futura-sciences.com/',
                'content': f'Guide et explication de {query}',
                'engine': 'futura-sciences',
                'score': 0.8
            }
        ]
        
        return base_results
    
    def _quality_content_filter(self, page_content: WebPageContent) -> bool:
        """Filtre les pages selon leur qualité"""
        return (page_content.content_quality_score >= self.content_quality_threshold and
                len(page_content.cleaned_text) > 200 and
                page_content.title != "Page sans titre")
    
    def _create_gemini_report(self, query: str, user_context: str, 
                            search_results: List[Dict], 
                            navigation_results: List[Dict],
                            search_id: str) -> Dict[str, Any]:
        """Crée un rapport structuré pour Gemini"""
        
        # Extraire le meilleur contenu
        best_content = []
        all_keywords = set()
        total_pages = 0
        
        for nav_result in navigation_results:
            nav_path = nav_result['navigation_path']
            total_pages += len(nav_path.visited_pages)
            
            for page in nav_path.visited_pages:
                if page.content_quality_score >= 4.0:  # Seulement le meilleur contenu
                    best_content.append({
                        'url': page.url,
                        'title': page.title,
                        'summary': page.summary,
                        'main_content': page.main_content[:1000],  # Limite pour Gemini
                        'keywords': page.keywords,
                        'quality_score': page.content_quality_score,
                        'language': page.language
                    })
                    all_keywords.update(page.keywords)
        
        # Créer une synthèse intelligente
        content_synthesis = self._synthesize_content(best_content)
        
        # Rapport final
        return {
            'search_id': search_id,
            'query': query,
            'user_context': user_context,
            'timestamp': datetime.now().isoformat(),
            'search_summary': {
                'sites_searched': len(search_results),
                'sites_navigated': len(navigation_results),
                'total_pages_visited': total_pages,
                'high_quality_pages': len(best_content)
            },
            'content_synthesis': content_synthesis,
            'best_content': best_content[:5],  # Top 5 contenus
            'aggregated_keywords': list(all_keywords)[:20],  # Top 20 mots-clés
            'navigation_insights': self._generate_navigation_insights(navigation_results),
            'recommended_actions': self._generate_recommendations(query, best_content),
            'success': True
        }
    
    def _synthesize_content(self, content_list: List[Dict]) -> str:
        """Synthétise le contenu extrait"""
        if not content_list:
            return "Aucun contenu de qualité trouvé."
        
        # Combiner les résumés
        all_summaries = [content['summary'] for content in content_list if content['summary']]
        
        # Extraire les informations clés
        key_info = []
        for content in content_list:
            key_info.append(f"• {content['title']}: {content['summary'][:150]}...")
        
        synthesis = f"Synthèse basée sur {len(content_list)} pages de qualité:\n\n"
        synthesis += "\n".join(key_info[:5])  # Top 5
        
        return synthesis
    
    def _generate_navigation_insights(self, navigation_results: List[Dict]) -> List[str]:
        """Génère des insights sur la navigation"""
        insights = []
        
        for nav_result in navigation_results:
            nav_path = nav_result['navigation_path']
            site_domain = urlparse(nav_result['search_result']['url']).netloc
            
            insights.append(f"Site {site_domain}: {len(nav_path.visited_pages)} pages explorées, "
                          f"profondeur {nav_path.navigation_depth}")
        
        return insights
    
    def _generate_recommendations(self, query: str, content_list: List[Dict]) -> List[str]:
        """Génère des recommandations basées sur le contenu"""
        recommendations = []
        
        if not content_list:
            recommendations.append("Essayer une recherche avec d'autres mots-clés")
            return recommendations
        
        # Recommandations basées sur la qualité
        high_quality_count = sum(1 for c in content_list if c['quality_score'] >= 7.0)
        if high_quality_count > 0:
            recommendations.append(f"{high_quality_count} sources de très haute qualité identifiées")
        
        # Recommandations linguistiques
        languages = set(c['language'] for c in content_list if c['language'])
        if 'fr' in languages and 'en' in languages:
            recommendations.append("Contenu disponible en français et anglais")
        
        return recommendations
    
    def _analyze_user_journey(self, nav_path: NavigationPath, intent: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyse le parcours utilisateur"""
        analysis = {
            'intent_satisfaction': 0.0,
            'journey_efficiency': 0.0,
            'content_relevance': 0.0,
            'key_findings': []
        }
        
        if not nav_path.visited_pages:
            return analysis
        
        # Calculer la satisfaction d'intention
        intent_pages = 0
        for page in nav_path.visited_pages:
            text_lower = page.cleaned_text.lower()
            if any(keyword in text_lower for keyword in keywords):
                intent_pages += 1
        
        analysis['intent_satisfaction'] = intent_pages / len(nav_path.visited_pages)
        
        # Calculer l'efficacité du parcours
        avg_quality = sum(p.content_quality_score for p in nav_path.visited_pages) / len(nav_path.visited_pages)
        analysis['journey_efficiency'] = min(avg_quality / 10.0, 1.0)
        
        # Pertinence du contenu
        total_content = sum(len(p.cleaned_text) for p in nav_path.visited_pages)
        analysis['content_relevance'] = min(total_content / 10000, 1.0)  # Normaliser
        
        # Findings clés
        analysis['key_findings'] = [
            f"Parcours de {len(nav_path.visited_pages)} pages",
            f"Profondeur de navigation: {nav_path.navigation_depth}",
            f"Score de qualité moyen: {avg_quality:.1f}/10"
        ]
        
        return analysis
    
    def _extract_key_pages(self, nav_path: NavigationPath, keywords: List[str]) -> List[Dict[str, Any]]:
        """Extrait les pages clés du parcours"""
        key_pages = []
        
        for page in nav_path.visited_pages:
            # Score basé sur qualité + pertinence mots-clés
            keyword_score = sum(1 for keyword in keywords 
                              if keyword in page.cleaned_text.lower())
            
            total_score = page.content_quality_score + keyword_score
            
            if total_score >= 5.0:  # Seuil pour les pages clés
                key_pages.append({
                    'url': page.url,
                    'title': page.title,
                    'summary': page.summary,
                    'score': total_score,
                    'keywords_found': keyword_score
                })
        
        # Trier par score et retourner le top
        key_pages.sort(key=lambda x: x['score'], reverse=True)
        return key_pages[:5]
    
    def _create_empty_result(self, query: str, reason: str) -> Dict[str, Any]:
        """Crée un résultat vide"""
        return {
            'search_id': f"empty_{int(time.time())}",
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'reason': reason,
            'search_summary': {
                'sites_searched': 0,
                'sites_navigated': 0,
                'total_pages_visited': 0,
                'high_quality_pages': 0
            },
            'content_synthesis': f"Aucun résultat trouvé pour: {query}",
            'best_content': [],
            'aggregated_keywords': [],
            'navigation_insights': [],
            'recommended_actions': ["Essayer avec d'autres mots-clés", "Vérifier l'orthographe"]
        }
    
    def _create_error_result(self, query: str, error: str) -> Dict[str, Any]:
        """Crée un résultat d'erreur"""
        return {
            'search_id': f"error_{int(time.time())}",
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'error': error,
            'search_summary': {
                'sites_searched': 0,
                'sites_navigated': 0,
                'total_pages_visited': 0,
                'high_quality_pages': 0
            },
            'content_synthesis': f"Erreur lors de la recherche: {error}",
            'best_content': [],
            'aggregated_keywords': [],
            'navigation_insights': [],
            'recommended_actions': ["Réessayer plus tard", "Vérifier la connexion"]
        }
    
    def _save_gemini_report(self, report: Dict[str, Any], search_id: str):
        """Sauvegarde le rapport pour Gemini"""
        try:
            filename = f"gemini_report_{search_id}.json"
            filepath = self.reports_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 Rapport Gemini sauvegardé: {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde rapport: {str(e)}")

# Instance globale
gemini_web_integration = None

def initialize_gemini_web_integration(searx_interface=None):
    """Initialise l'intégration Gemini-Web"""
    global gemini_web_integration
    gemini_web_integration = GeminiWebNavigationIntegration(searx_interface)
    logger.info("🚀 Intégration Gemini-Web initialisée")

def search_web_for_gemini(query: str, user_context: str = "") -> Dict[str, Any]:
    """Interface publique pour Gemini"""
    if not gemini_web_integration:
        initialize_gemini_web_integration()
    
    return gemini_web_integration.search_and_navigate_for_gemini(query, user_context)

def extract_content_for_gemini(url: str, requirements: List[str] = None) -> Dict[str, Any]:
    """Interface publique pour extraction spécifique"""
    if not gemini_web_integration:
        initialize_gemini_web_integration()
    
    if requirements is None:
        requirements = ['summary', 'details', 'links']
    
    return gemini_web_integration.extract_specific_content(url, requirements)

def simulate_user_journey(start_url: str, intent: str) -> Dict[str, Any]:
    """Interface publique pour parcours utilisateur"""
    if not gemini_web_integration:
        initialize_gemini_web_integration()
    
    return gemini_web_integration.navigate_user_journey(start_url, intent)

if __name__ == "__main__":
    print("=== Test de l'Intégration Gemini-Navigation ===")
    
    # Initialiser
    initialize_gemini_web_integration()
    
    # Test de recherche
    test_query = "intelligence artificielle apprentissage automatique"
    print(f"Test de recherche: {test_query}")
    
    result = search_web_for_gemini(test_query, "utilisateur intéressé par l'IA")
    
    if result['success']:
        print(f"✅ Recherche réussie: {result['search_summary']['total_pages_visited']} pages visitées")
        print(f"✅ Synthèse: {result['content_synthesis'][:200]}...")
        print(f"✅ Mots-clés: {result['aggregated_keywords'][:10]}")
    else:
        print(f"❌ Échec: {result.get('error', 'Erreur inconnue')}")
