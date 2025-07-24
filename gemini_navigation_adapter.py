"""
Intégration du Système de Navigation Web Avancé avec l'Adapter Gemini
Ce module connecte le nouveau système de navigation avec l'adapter Gemini existant
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from gemini_web_integration import (
    initialize_gemini_web_integration,
    search_web_for_gemini,
    extract_content_for_gemini,
    simulate_user_journey
)
from advanced_web_navigator import extract_website_content, navigate_website_deep

# Configuration du logging
logger = logging.getLogger('GeminiAdapterIntegration')

class GeminiWebNavigationAdapter:
    """Adaptateur pour intégrer la navigation web avec l'API Gemini existante"""
    
    def __init__(self, gemini_api_instance=None):
        self.gemini_api = gemini_api_instance
        self.navigation_enabled = True
        self.max_content_length = 8000  # Limite pour Gemini
        
        # Initialiser l'intégration web
        initialize_gemini_web_integration()
        
        # Compteurs et statistiques
        self.navigation_stats = {
            'total_requests': 0,
            'successful_navigations': 0,
            'content_extractions': 0,
            'searches_performed': 0
        }
        
        logger.info("🔗 Adaptateur Gemini-Navigation initialisé")
    
    def detect_navigation_request(self, prompt: str) -> Dict[str, Any]:
        """
        Détecte si le prompt nécessite une navigation web avancée
        
        Args:
            prompt: Le prompt de l'utilisateur
            
        Returns:
            Dict contenant le type de navigation et les paramètres
        """
        prompt_lower = prompt.lower()
        
        # Mots-clés pour navigation profonde
        deep_navigation_keywords = [
            'explore le site', 'navigue dans', 'parcours le site', 'visite toutes les pages',
            'analyse complète du site', 'navigation profonde', 'explore en détail'
        ]
        
        # Mots-clés pour extraction spécifique
        extraction_keywords = [
            'extrait le contenu de', 'analyse cette page', 'récupère les informations de',
            'contenu de cette url', 'détails de la page'
        ]
        
        # Mots-clés pour recherche et navigation
        search_navigation_keywords = [
            'recherche et navigue', 'trouve et explore', 'cherche et analyse',
            'recherche détaillée', 'information complète sur'
        ]
        
        # Mots-clés pour parcours utilisateur
        user_journey_keywords = [
            'parcours utilisateur', 'expérience utilisateur', 'navigation utilisateur',
            'comme un utilisateur', 'simule un achat', 'processus d\'achat'
        ]
        
        detection_result = {
            'requires_navigation': False,
            'navigation_type': None,
            'confidence': 0.0,
            'extracted_params': {}
        }
        
        # Détecter le type de navigation
        if any(keyword in prompt_lower for keyword in deep_navigation_keywords):
            detection_result.update({
                'requires_navigation': True,
                'navigation_type': 'deep_navigation',
                'confidence': 0.9
            })
            
            # Extraire l'URL si présente
            url_match = self._extract_url_from_prompt(prompt)
            if url_match:
                detection_result['extracted_params']['start_url'] = url_match
                
        elif any(keyword in prompt_lower for keyword in extraction_keywords):
            detection_result.update({
                'requires_navigation': True,
                'navigation_type': 'content_extraction',
                'confidence': 0.8
            })
            
            url_match = self._extract_url_from_prompt(prompt)
            if url_match:
                detection_result['extracted_params']['url'] = url_match
                
        elif any(keyword in prompt_lower for keyword in search_navigation_keywords):
            detection_result.update({
                'requires_navigation': True,
                'navigation_type': 'search_and_navigate',
                'confidence': 0.9
            })
            
            query = self._extract_search_query_from_prompt(prompt)
            if query:
                detection_result['extracted_params']['query'] = query
                
        elif any(keyword in prompt_lower for keyword in user_journey_keywords):
            detection_result.update({
                'requires_navigation': True,
                'navigation_type': 'user_journey',
                'confidence': 0.7
            })
            
            url_match = self._extract_url_from_prompt(prompt)
            intent = self._extract_user_intent_from_prompt(prompt)
            if url_match:
                detection_result['extracted_params']['start_url'] = url_match
            if intent:
                detection_result['extracted_params']['user_intent'] = intent
        
        # Détecter les requêtes de recherche web générales qui pourraient bénéficier de navigation
        elif self._is_general_web_search(prompt):
            detection_result.update({
                'requires_navigation': True,
                'navigation_type': 'search_and_navigate',
                'confidence': 0.6
            })
            detection_result['extracted_params']['query'] = prompt
        
        return detection_result
    
    def handle_navigation_request(self, prompt: str, user_id: int = 1, 
                                session_id: str = None) -> Dict[str, Any]:
        """
        Traite une requête de navigation web
        
        Args:
            prompt: Le prompt de l'utilisateur
            user_id: ID de l'utilisateur
            session_id: ID de la session
            
        Returns:
            Résultat de la navigation formaté pour Gemini
        """
        if not self.navigation_enabled:
            return {
                'success': False,
                'error': 'Navigation web désactivée',
                'fallback_required': True
            }
        
        self.navigation_stats['total_requests'] += 1
        
        try:
            # Détecter le type de navigation
            detection = self.detect_navigation_request(prompt)
            
            if not detection['requires_navigation']:
                return {
                    'success': False,
                    'error': 'Navigation non détectée',
                    'fallback_required': True
                }
            
            logger.info(f"🎯 Navigation détectée: {detection['navigation_type']} (confiance: {detection['confidence']})")
            
            # Traiter selon le type de navigation
            if detection['navigation_type'] == 'search_and_navigate':
                result = self._handle_search_and_navigate(detection, prompt, user_id)
                
            elif detection['navigation_type'] == 'content_extraction':
                result = self._handle_content_extraction(detection, prompt)
                
            elif detection['navigation_type'] == 'deep_navigation':
                result = self._handle_deep_navigation(detection, prompt)
                
            elif detection['navigation_type'] == 'user_journey':
                result = self._handle_user_journey(detection, prompt)
                
            else:
                return {
                    'success': False,
                    'error': f'Type de navigation non pris en charge: {detection["navigation_type"]}',
                    'fallback_required': True
                }
            
            # Formater pour Gemini
            if result.get('success', False):
                self.navigation_stats['successful_navigations'] += 1
                gemini_response = self._format_for_gemini(result, detection['navigation_type'], prompt)
                
                logger.info(f"✅ Navigation réussie: {detection['navigation_type']}")
                return gemini_response
            else:
                logger.warning(f"⚠️ Navigation échouée: {result.get('error', 'Erreur inconnue')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Erreur de navigation'),
                    'fallback_required': True
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de la navigation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_required': True
            }
    
    def _handle_search_and_navigate(self, detection: Dict, prompt: str, user_id: int) -> Dict[str, Any]:
        """Traite une requête de recherche et navigation"""
        query = detection['extracted_params'].get('query', prompt)
        user_context = f"User {user_id} - Recherche avancée"
        
        logger.info(f"🔍 Recherche et navigation: {query}")
        
        result = search_web_for_gemini(query, user_context)
        self.navigation_stats['searches_performed'] += 1
        
        return result
    
    def _handle_content_extraction(self, detection: Dict, prompt: str) -> Dict[str, Any]:
        """Traite une requête d'extraction de contenu"""
        url = detection['extracted_params'].get('url')
        
        if not url:
            return {
                'success': False,
                'error': 'URL non trouvée dans le prompt'
            }
        
        logger.info(f"🎯 Extraction de contenu: {url}")
        
        # Déterminer les exigences d'extraction basées sur le prompt
        requirements = self._determine_extraction_requirements(prompt)
        
        result = extract_content_for_gemini(url, requirements)
        self.navigation_stats['content_extractions'] += 1
        
        return result
    
    def _handle_deep_navigation(self, detection: Dict, prompt: str) -> Dict[str, Any]:
        """Traite une requête de navigation profonde"""
        start_url = detection['extracted_params'].get('start_url')
        
        if not start_url:
            return {
                'success': False,
                'error': 'URL de départ non trouvée dans le prompt'
            }
        
        logger.info(f"🚀 Navigation profonde: {start_url}")
        
        # Paramètres par défaut ou extraits du prompt
        max_depth = self._extract_number_from_prompt(prompt, 'profondeur', 3)
        max_pages = self._extract_number_from_prompt(prompt, 'pages', 10)
        
        nav_path = navigate_website_deep(start_url, max_depth, max_pages)
        
        # Convertir en format compatible
        result = {
            'success': True,
            'navigation_summary': {
                'start_url': nav_path.start_url,
                'pages_visited': len(nav_path.visited_pages),
                'navigation_depth': nav_path.navigation_depth,
                'total_content_extracted': nav_path.total_content_extracted
            },
            'visited_pages': [
                {
                    'url': page.url,
                    'title': page.title,
                    'summary': page.summary,
                    'content_quality_score': page.content_quality_score,
                    'keywords': page.keywords[:10]
                }
                for page in nav_path.visited_pages
            ]
        }
        
        return result
    
    def _handle_user_journey(self, detection: Dict, prompt: str) -> Dict[str, Any]:
        """Traite une requête de parcours utilisateur"""
        start_url = detection['extracted_params'].get('start_url')
        user_intent = detection['extracted_params'].get('user_intent', 'explore')
        
        if not start_url:
            return {
                'success': False,
                'error': 'URL de départ non trouvée dans le prompt'
            }
        
        logger.info(f"👤 Parcours utilisateur: {user_intent} depuis {start_url}")
        
        result = simulate_user_journey(start_url, user_intent)
        return result
    
    def _format_for_gemini(self, result: Dict[str, Any], navigation_type: str, 
                          original_prompt: str) -> Dict[str, Any]:
        """Formate le résultat pour l'API Gemini"""
        
        # Créer un résumé adapté au type de navigation
        if navigation_type == 'search_and_navigate':
            summary = self._create_search_summary(result)
            
        elif navigation_type == 'content_extraction':
            summary = self._create_extraction_summary(result)
            
        elif navigation_type == 'deep_navigation':
            summary = self._create_navigation_summary(result)
            
        elif navigation_type == 'user_journey':
            summary = self._create_journey_summary(result)
            
        else:
            summary = "Navigation web effectuée avec succès."
        
        # Préparer le contenu pour Gemini
        gemini_content = {
            'web_navigation_summary': summary,
            'navigation_type': navigation_type,
            'data_extracted': True,
            'content_length': len(str(result)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Ajouter des détails spécifiques selon le type
        if navigation_type == 'search_and_navigate' and 'best_content' in result:
            gemini_content['key_findings'] = [
                f"📄 {content['title']}: {content['summary'][:200]}..."
                for content in result['best_content'][:3]
            ]
            
        elif navigation_type == 'content_extraction' and 'summary' in result:
            gemini_content['extracted_summary'] = result['summary']
            
        elif navigation_type == 'deep_navigation' and 'visited_pages' in result:
            gemini_content['pages_explored'] = len(result['visited_pages'])
            gemini_content['top_pages'] = [
                f"📄 {page['title']} (Score: {page['content_quality_score']:.1f})"
                for page in sorted(result['visited_pages'], 
                                 key=lambda x: x['content_quality_score'], reverse=True)[:3]
            ]
        
        return {
            'success': True,
            'navigation_performed': True,
            'gemini_ready_content': gemini_content,
            'raw_data': result if len(str(result)) < self.max_content_length else None,
            'content_truncated': len(str(result)) >= self.max_content_length
        }
    
    def _create_search_summary(self, result: Dict[str, Any]) -> str:
        """Crée un résumé de recherche pour Gemini"""
        if not result.get('success', False):
            return "❌ La recherche web n'a pas abouti."
        
        search_summary = result.get('search_summary', {})
        sites_navigated = search_summary.get('sites_navigated', 0)
        pages_visited = search_summary.get('total_pages_visited', 0)
        
        summary = f"🌐 **Recherche web effectuée avec succès !**\n\n"
        summary += f"J'ai navigué sur {sites_navigated} sites web et analysé {pages_visited} pages.\n\n"
        
        if 'content_synthesis' in result:
            summary += f"**Synthèse des informations trouvées :**\n{result['content_synthesis']}\n\n"
        
        if 'aggregated_keywords' in result and result['aggregated_keywords']:
            keywords = ', '.join(result['aggregated_keywords'][:10])
            summary += f"**Mots-clés identifiés :** {keywords}\n\n"
        
        summary += "Les informations détaillées ont été intégrées dans ma base de connaissances."
        
        return summary
    
    def _create_extraction_summary(self, result: Dict[str, Any]) -> str:
        """Crée un résumé d'extraction pour Gemini"""
        if not result.get('success', False):
            return f"❌ Impossible d'extraire le contenu de l'URL : {result.get('error', 'Erreur inconnue')}"
        
        summary = f"📄 **Contenu extrait avec succès !**\n\n"
        summary += f"**Titre :** {result.get('title', 'Non spécifié')}\n"
        summary += f"**URL :** {result.get('url', 'Non spécifiée')}\n"
        summary += f"**Langue :** {result.get('language', 'Non détectée')}\n"
        summary += f"**Score de qualité :** {result.get('content_quality_score', 0):.1f}/10\n\n"
        
        if 'summary' in result:
            summary += f"**Résumé :**\n{result['summary']}\n\n"
        
        if 'keywords' in result and result['keywords']:
            keywords = ', '.join(result['keywords'][:8])
            summary += f"**Mots-clés :** {keywords}\n\n"
        
        return summary
    
    def _create_navigation_summary(self, result: Dict[str, Any]) -> str:
        """Crée un résumé de navigation pour Gemini"""
        if not result.get('success', False):
            return "❌ La navigation profonde a échoué."
        
        nav_summary = result.get('navigation_summary', {})
        pages_visited = nav_summary.get('pages_visited', 0)
        depth = nav_summary.get('navigation_depth', 0)
        
        summary = f"🚀 **Navigation profonde effectuée !**\n\n"
        summary += f"J'ai exploré {pages_visited} pages avec une profondeur de navigation de {depth} niveaux.\n\n"
        
        if 'visited_pages' in result and result['visited_pages']:
            summary += "**Pages les plus pertinentes :**\n"
            top_pages = sorted(result['visited_pages'], 
                             key=lambda x: x['content_quality_score'], reverse=True)[:3]
            
            for i, page in enumerate(top_pages, 1):
                summary += f"{i}. **{page['title']}** (Score: {page['content_quality_score']:.1f})\n"
                summary += f"   📄 {page['summary'][:150]}...\n\n"
        
        return summary
    
    def _create_journey_summary(self, result: Dict[str, Any]) -> str:
        """Crée un résumé de parcours utilisateur pour Gemini"""
        if not result.get('success', False):
            return f"❌ Simulation du parcours utilisateur échouée : {result.get('error', 'Erreur inconnue')}"
        
        pages_visited = result.get('pages_visited', 0)
        user_intent = result.get('user_intent', 'explore')
        
        intent_names = {
            'buy': 'achat',
            'learn': 'apprentissage',
            'contact': 'contact',
            'explore': 'exploration'
        }
        
        intent_text = intent_names.get(user_intent, user_intent)
        
        summary = f"👤 **Parcours utilisateur simulé avec succès !**\n\n"
        summary += f"J'ai simulé un parcours d'**{intent_text}** sur {pages_visited} pages.\n\n"
        
        if 'journey_analysis' in result:
            analysis = result['journey_analysis']
            satisfaction = analysis.get('intent_satisfaction', 0) * 100
            summary += f"**Satisfaction de l'intention :** {satisfaction:.1f}%\n\n"
        
        if 'key_pages' in result and result['key_pages']:
            summary += "**Pages clés identifiées :**\n"
            for i, page in enumerate(result['key_pages'][:3], 1):
                summary += f"{i}. **{page['title']}**\n"
                summary += f"   📄 {page['summary'][:100]}...\n\n"
        
        return summary
    
    # Méthodes utilitaires
    def _extract_url_from_prompt(self, prompt: str) -> Optional[str]:
        """Extrait une URL du prompt"""
        import re
        url_pattern = r'https?://[^\s<>"{\|}\\^`\[\]]+'
        urls = re.findall(url_pattern, prompt)
        return urls[0] if urls else None
    
    def _extract_search_query_from_prompt(self, prompt: str) -> Optional[str]:
        """Extrait une requête de recherche du prompt"""
        # Patterns pour extraire la requête
        patterns = [
            r'recherche\s+(?:et\s+navigue\s+)?["\']([^"\']+)["\']',
            r'cherche\s+(?:et\s+analyse\s+)?["\']([^"\']+)["\']',
            r'trouve\s+(?:et\s+explore\s+)?["\']([^"\']+)["\']',
            r'recherche\s+(?:et\s+navigue\s+)?(?:sur\s+)?(.+?)(?:\s+et\s|$)',
            r'cherche\s+(?:et\s+analyse\s+)?(?:sur\s+)?(.+?)(?:\s+et\s|$)'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_user_intent_from_prompt(self, prompt: str) -> str:
        """Extrait l'intention utilisateur du prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['acheter', 'achat', 'commander', 'prix']):
            return 'buy'
        elif any(word in prompt_lower for word in ['apprendre', 'formation', 'cours', 'tutoriel']):
            return 'learn'
        elif any(word in prompt_lower for word in ['contact', 'contacter', 'support', 'aide']):
            return 'contact'
        else:
            return 'explore'
    
    def _extract_number_from_prompt(self, prompt: str, context: str, default: int) -> int:
        """Extrait un nombre du prompt selon le contexte"""
        import re
        
        # Patterns pour différents contextes
        if context == 'profondeur':
            patterns = [r'profondeur\s+(?:de\s+)?(\d+)', r'(\d+)\s+niveaux?']
        elif context == 'pages':
            patterns = [r'(\d+)\s+pages?', r'maximum\s+(\d+)\s+pages?']
        else:
            patterns = [r'(\d+)']
        
        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        
        return default
    
    def _determine_extraction_requirements(self, prompt: str) -> List[str]:
        """Détermine les exigences d'extraction basées sur le prompt"""
        prompt_lower = prompt.lower()
        requirements = ['summary']  # Toujours inclure le résumé
        
        if any(word in prompt_lower for word in ['détail', 'complet', 'tout', 'entier']):
            requirements.extend(['details', 'structure'])
        
        if any(word in prompt_lower for word in ['liens', 'link', 'url']):
            requirements.append('links')
        
        if any(word in prompt_lower for word in ['images', 'photos', 'illustrations']):
            requirements.append('images')
        
        if any(word in prompt_lower for word in ['navigation', 'menu', 'naviguer']):
            requirements.append('navigation')
        
        if any(word in prompt_lower for word in ['métadonnées', 'metadata', 'propriétés']):
            requirements.append('metadata')
        
        return list(set(requirements))  # Supprimer les doublons
    
    def _is_general_web_search(self, prompt: str) -> bool:
        """Détermine si c'est une recherche web générale qui pourrait bénéficier de navigation"""
        prompt_lower = prompt.lower()
        
        # Mots-clés indiquant une recherche d'information
        search_indicators = [
            'qu\'est-ce que', 'comment', 'pourquoi', 'où trouver', 'information sur',
            'explication de', 'définition de', 'guide pour', 'tutoriel sur'
        ]
        
        # Vérifier si c'est une question ou demande d'information
        if any(indicator in prompt_lower for indicator in search_indicators):
            return True
        
        # Vérifier si c'est une requête qui se termine par une question
        if prompt.strip().endswith('?'):
            return True
        
        return False
    
    def get_navigation_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques de navigation"""
        return {
            'navigation_stats': self.navigation_stats.copy(),
            'navigation_enabled': self.navigation_enabled,
            'max_content_length': self.max_content_length
        }
    
    def enable_navigation(self):
        """Active la navigation web"""
        self.navigation_enabled = True
        logger.info("🟢 Navigation web activée")
    
    def disable_navigation(self):
        """Désactive la navigation web"""
        self.navigation_enabled = False
        logger.info("🔴 Navigation web désactivée")

# Instance globale
gemini_navigation_adapter = None

def initialize_gemini_navigation_adapter(gemini_api_instance=None):
    """Initialise l'adaptateur de navigation Gemini"""
    global gemini_navigation_adapter
    gemini_navigation_adapter = GeminiWebNavigationAdapter(gemini_api_instance)
    logger.info("🔗 Adaptateur Gemini-Navigation initialisé")

def handle_gemini_navigation_request(prompt: str, user_id: int = 1, session_id: str = None) -> Dict[str, Any]:
    """Interface publique pour les requêtes de navigation Gemini"""
    if not gemini_navigation_adapter:
        initialize_gemini_navigation_adapter()
    
    return gemini_navigation_adapter.handle_navigation_request(prompt, user_id, session_id)

def detect_navigation_need(prompt: str) -> Dict[str, Any]:
    """Interface publique pour la détection de navigation"""
    if not gemini_navigation_adapter:
        initialize_gemini_navigation_adapter()
    
    return gemini_navigation_adapter.detect_navigation_request(prompt)

if __name__ == "__main__":
    print("=== Test de l'Adaptateur Gemini-Navigation ===")
    
    # Initialiser
    initialize_gemini_navigation_adapter()
    
    # Tests de détection
    test_prompts = [
        "Recherche et navigue sur l'intelligence artificielle",
        "Extrait le contenu de https://example.com",
        "Explore le site https://wikipedia.org en profondeur",
        "Simule un parcours d'achat sur https://shop.example.com",
        "Qu'est-ce que l'apprentissage automatique ?"
    ]
    
    print("🧪 Tests de détection de navigation:")
    for prompt in test_prompts:
        detection = detect_navigation_need(prompt)
        print(f"  📝 '{prompt}'")
        print(f"     → Type: {detection['navigation_type']}, Confiance: {detection['confidence']}")
        print(f"     → Paramètres: {detection['extracted_params']}")
        print()
