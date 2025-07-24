"""
Intégration du Système de Navigation Interactive avec l'Adaptateur Gemini
Ce module connecte le nouveau système d'interaction web avec l'API Gemini
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from interactive_web_navigator import (
    get_interactive_navigator,
    initialize_interactive_navigator,
    create_interactive_navigation_session,
    interact_with_web_element,
    get_page_interactive_elements,
    close_interactive_session
)

# Configuration du logging
logger = logging.getLogger('GeminiInteractiveIntegration')

class GeminiInteractiveWebAdapter:
    """Adaptateur pour intégrer la navigation interactive avec l'API Gemini"""
    
    def __init__(self, gemini_api_instance=None):
        self.gemini_api = gemini_api_instance
        self.interactive_enabled = True
        self.max_content_length = 8000
        
        # Initialiser le navigateur interactif
        self.navigator = initialize_interactive_navigator()
        
        # Compteurs et statistiques
        self.interaction_stats = {
            'total_requests': 0,
            'interactive_sessions_created': 0,
            'successful_interactions': 0,
            'elements_clicked': 0,
            'tabs_explored': 0,
            'forms_interacted': 0
        }
        
        logger.info("🎯 Adaptateur Gemini-Navigation Interactive initialisé")
    
    def detect_interactive_request(self, prompt: str) -> Dict[str, Any]:
        """
        Détecte si le prompt nécessite une interaction avec des éléments web
        
        Args:
            prompt: Le prompt de l'utilisateur
            
        Returns:
            Dict contenant le type d'interaction et les paramètres
        """
        prompt_lower = prompt.lower()
        
        # Mots-clés pour interactions directes
        interaction_keywords = [
            'clique sur', 'cliquer sur', 'appuie sur', 'appuyer sur',
            'sélectionne', 'sélectionner', 'choisir', 'choisit',
            'ouvre l\'onglet', 'ouvrir l\'onglet', 'va dans l\'onglet',
            'remplis le formulaire', 'remplir le formulaire',
            'interagir avec', 'interagis avec'
        ]
        
        # Mots-clés pour navigation par onglets
        tab_keywords = [
            'onglet', 'onglets', 'tab', 'tabs',
            'section', 'sections', 'catégorie', 'catégories',
            'menu', 'navigation'
        ]
        
        # Mots-clés pour exploration complète
        exploration_keywords = [
            'explore toutes les options', 'parcours tous les onglets',
            'visite toutes les sections', 'analyse tous les menus',
            'teste toutes les fonctionnalités'
        ]
        
        # Mots-clés pour formulaires
        form_keywords = [
            'formulaire', 'form', 'remplis', 'saisir', 'entrer',
            'recherche', 'search', 'login', 'connexion'
        ]
        
        detection_result = {
            'requires_interaction': False,
            'interaction_type': None,
            'confidence': 0.0,
            'extracted_params': {},
            'suggested_actions': []
        }
        
        # Détecter les interactions directes
        if any(keyword in prompt_lower for keyword in interaction_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'direct_interaction',
                'confidence': 0.9
            })
            
            # Extraire l'élément cible si mentionné
            target_element = self._extract_target_element(prompt)
            if target_element:
                detection_result['extracted_params']['target_element'] = target_element
        
        # Détecter la navigation par onglets
        elif any(keyword in prompt_lower for keyword in tab_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'tab_navigation',
                'confidence': 0.8
            })
            detection_result['suggested_actions'] = ['explore_tabs', 'click_tabs']
        
        # Détecter l'exploration complète
        elif any(keyword in prompt_lower for keyword in exploration_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'full_exploration',
                'confidence': 0.85
            })
            detection_result['suggested_actions'] = ['explore_all_elements', 'systematic_navigation']
        
        # Détecter les interactions avec formulaires
        elif any(keyword in prompt_lower for keyword in form_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'form_interaction',
                'confidence': 0.7
            })
            detection_result['suggested_actions'] = ['fill_forms', 'submit_forms']
        
        # Extraire l'URL si présente
        url_match = self._extract_url_from_prompt(prompt)
        if url_match:
            detection_result['extracted_params']['url'] = url_match
        
        self.interaction_stats['total_requests'] += 1
        
        logger.info(f"🔍 Détection interaction: {detection_result['interaction_type']} "
                   f"(confiance: {detection_result['confidence']})")
        
        return detection_result
    
    def handle_interactive_request(self, prompt: str, user_id: int, 
                                 session_id: str = None) -> Dict[str, Any]:
        """
        Traite une demande d'interaction web
        
        Args:
            prompt: Le prompt de l'utilisateur
            user_id: ID de l'utilisateur
            session_id: ID de session (optionnel)
            
        Returns:
            Dict contenant la réponse et les données d'interaction
        """
        try:
            # Détecter le type d'interaction nécessaire
            detection = self.detect_interactive_request(prompt)
            
            if not detection['requires_interaction']:
                return {
                    'success': False,
                    'error': 'Aucune interaction détectée',
                    'fallback_required': True
                }
            
            # Générer un ID de session unique si non fourni
            if not session_id:
                session_id = f"interactive_{user_id}_{int(time.time())}"
            
            interaction_type = detection['interaction_type']
            
            # Traiter selon le type d'interaction
            if interaction_type == 'direct_interaction':
                result = self._handle_direct_interaction(prompt, session_id, detection)
            elif interaction_type == 'tab_navigation':
                result = self._handle_tab_navigation(prompt, session_id, detection)
            elif interaction_type == 'full_exploration':
                result = self._handle_full_exploration(prompt, session_id, detection)
            elif interaction_type == 'form_interaction':
                result = self._handle_form_interaction(prompt, session_id, detection)
            else:
                result = self._handle_generic_interaction(prompt, session_id, detection)
            
            # Enrichir la réponse avec des informations contextuelles
            if result['success']:
                result['interaction_summary'] = self._generate_interaction_summary(session_id)
                result['response'] = self._format_interaction_response(result, prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement interaction: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_required': True
            }
    
    def _handle_direct_interaction(self, prompt: str, session_id: str, 
                                 detection: Dict[str, Any]) -> Dict[str, Any]:
        """Traite une interaction directe (clic sur un élément spécifique)"""
        try:
            # Extraire l'URL si fournie
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL requise pour l\'interaction directe'}
            
            # Créer la session interactive
            navigation_result = create_interactive_navigation_session(
                session_id, url, goals=['direct_interaction']
            )
            
            if not navigation_result['success']:
                return navigation_result
            
            self.interaction_stats['interactive_sessions_created'] += 1
            
            # Obtenir les éléments interactifs
            elements_result = get_page_interactive_elements(session_id)
            
            if not elements_result['success']:
                return elements_result
            
            # Identifier l'élément cible
            target_text = detection['extracted_params'].get('target_element', '')
            target_element = self._find_best_matching_element(
                elements_result['top_interactive_elements'], target_text
            )
            
            if not target_element:
                return {
                    'success': False,
                    'error': 'Élément cible non trouvé',
                    'available_elements': elements_result['top_interactive_elements'][:5]
                }
            
            # Effectuer l'interaction
            interaction_result = interact_with_web_element(
                session_id, target_element['id'], 'click'
            )
            
            if interaction_result['success']:
                self.interaction_stats['successful_interactions'] += 1
                self.interaction_stats['elements_clicked'] += 1
            
            return {
                'success': interaction_result['success'],
                'interaction_performed': True,
                'element_interacted': target_element,
                'page_changed': interaction_result['page_changed'],
                'new_url': interaction_result.get('new_url'),
                'details': interaction_result
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur interaction directe: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_tab_navigation(self, prompt: str, session_id: str, 
                             detection: Dict[str, Any]) -> Dict[str, Any]:
        """Traite la navigation par onglets"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL requise pour la navigation par onglets'}
            
            # Créer la session
            navigation_result = create_interactive_navigation_session(
                session_id, url, goals=['tab_navigation', 'explore_tabs']
            )
            
            if not navigation_result['success']:
                return navigation_result
            
            self.interaction_stats['interactive_sessions_created'] += 1
            
            # Obtenir les éléments
            elements_result = get_page_interactive_elements(session_id)
            tabs_info = []
            tab_contents = []
            
            # Trouver tous les onglets
            for element in elements_result.get('top_interactive_elements', []):
                if element['type'] == 'tabs':
                    # Cliquer sur l'onglet
                    interaction_result = interact_with_web_element(
                        session_id, element['id'], 'click'
                    )
                    
                    if interaction_result['success']:
                        self.interaction_stats['tabs_explored'] += 1
                        
                        # Attendre le chargement du contenu
                        time.sleep(2)
                        
                        # Capturer le contenu de l'onglet
                        current_elements = get_page_interactive_elements(session_id)
                        tab_content = {
                            'tab_name': element['text'],
                            'tab_id': element['id'],
                            'content_summary': self._summarize_tab_content(current_elements),
                            'interactive_elements': len(current_elements.get('top_interactive_elements', []))
                        }
                        tab_contents.append(tab_content)
                        tabs_info.append(element)
            
            return {
                'success': True,
                'interaction_performed': True,
                'tabs_explored': len(tabs_info),
                'tabs_content': tab_contents,
                'navigation_summary': f"Exploré {len(tabs_info)} onglets avec succès"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur navigation onglets: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_full_exploration(self, prompt: str, session_id: str, 
                               detection: Dict[str, Any]) -> Dict[str, Any]:
        """Traite l'exploration complète d'un site"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL requise pour l\'exploration complète'}
            
            navigation_result = create_interactive_navigation_session(
                session_id, url, goals=['full_exploration', 'systematic_analysis']
            )
            
            if not navigation_result['success']:
                return navigation_result
            
            exploration_results = {
                'tabs_explored': 0,
                'buttons_clicked': 0,
                'forms_found': 0,
                'navigation_links_followed': 0,
                'content_discovered': [],
                'interaction_log': []
            }
            
            # Phase 1: Explorer tous les onglets
            elements_result = get_page_interactive_elements(session_id)
            
            for element in elements_result.get('top_interactive_elements', [])[:15]:  # Top 15 pour éviter trop d'interactions
                if element['score'] > 0.5:  # Seulement les éléments pertinents
                    interaction_result = interact_with_web_element(
                        session_id, element['id'], 'click'
                    )
                    
                    exploration_results['interaction_log'].append({
                        'element_text': element['text'],
                        'element_type': element['type'],
                        'success': interaction_result['success'],
                        'page_changed': interaction_result.get('page_changed', False)
                    })
                    
                    if interaction_result['success']:
                        if element['type'] == 'tabs':
                            exploration_results['tabs_explored'] += 1
                        elif element['type'] == 'buttons':
                            exploration_results['buttons_clicked'] += 1
                        elif element['type'] == 'navigation':
                            exploration_results['navigation_links_followed'] += 1
                    
                    # Petit délai entre les interactions
                    time.sleep(1.5)
            
            # Résumé final
            total_interactions = sum([
                exploration_results['tabs_explored'],
                exploration_results['buttons_clicked'],
                exploration_results['navigation_links_followed']
            ])
            
            return {
                'success': True,
                'interaction_performed': True,
                'exploration_complete': True,
                'total_interactions': total_interactions,
                'results': exploration_results,
                'summary': f"Exploration complète: {total_interactions} interactions réalisées"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur exploration complète: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_form_interaction(self, prompt: str, session_id: str, 
                               detection: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les interactions avec les formulaires"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL requise pour l\'interaction avec formulaire'}
            
            navigation_result = create_interactive_navigation_session(
                session_id, url, goals=['form_interaction']
            )
            
            if not navigation_result['success']:
                return navigation_result
            
            elements_result = get_page_interactive_elements(session_id)
            form_results = []
            
            # Trouver les formulaires et champs
            for element in elements_result.get('elements_by_type', {}).get('forms', []):
                form_results.append({
                    'form_id': element['id'],
                    'form_text': element['text'],
                    'interactions_available': ['analyze_fields', 'test_submission']
                })
                
                self.interaction_stats['forms_interacted'] += 1
            
            return {
                'success': True,
                'interaction_performed': True,
                'forms_found': len(form_results),
                'forms_details': form_results,
                'note': 'Interaction avec formulaires identifiés (saisie de données non implémentée pour sécurité)'
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur interaction formulaire: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_generic_interaction(self, prompt: str, session_id: str, 
                                  detection: Dict[str, Any]) -> Dict[str, Any]:
        """Traite les interactions génériques"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL requise'}
            
            navigation_result = create_interactive_navigation_session(session_id, url)
            
            if not navigation_result['success']:
                return navigation_result
            
            elements_result = get_page_interactive_elements(session_id)
            
            return {
                'success': True,
                'interaction_performed': False,
                'analysis_performed': True,
                'elements_discovered': elements_result.get('total_elements', 0),
                'interactive_elements': elements_result.get('top_interactive_elements', [])[:10],
                'suggestions': elements_result.get('interaction_suggestions', [])
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur interaction générique: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_target_element(self, prompt: str) -> str:
        """Extrait l'élément cible mentionné dans le prompt"""
        prompt_lower = prompt.lower()
        
        # Patterns pour identifier les éléments cibles
        patterns = [
            r'clique(?:r)? sur ["\']?([^"\']+)["\']?',
            r'appuie(?:r)? sur ["\']?([^"\']+)["\']?',
            r'sélectionne(?:r)? ["\']?([^"\']+)["\']?',
            r'l\'onglet ["\']?([^"\']+)["\']?',
            r'le bouton ["\']?([^"\']+)["\']?',
            r'le lien ["\']?([^"\']+)["\']?'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_url_from_prompt(self, prompt: str) -> Optional[str]:
        """Extrait l'URL du prompt"""
        import re
        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        match = re.search(url_pattern, prompt)
        return match.group(0) if match else None
    
    def _find_best_matching_element(self, elements: List[Dict], target_text: str) -> Optional[Dict]:
        """Trouve l'élément qui correspond le mieux au texte cible"""
        if not target_text:
            return elements[0] if elements else None
        
        target_lower = target_text.lower()
        best_match = None
        best_score = 0
        
        for element in elements:
            element_text = element.get('text', '').lower()
            
            # Score exact match
            if target_lower == element_text:
                return element
            
            # Score partial match
            if target_lower in element_text or element_text in target_lower:
                score = len(set(target_lower.split()) & set(element_text.split()))
                if score > best_score:
                    best_score = score
                    best_match = element
        
        return best_match if best_match else (elements[0] if elements else None)
    
    def _summarize_tab_content(self, elements_result: Dict[str, Any]) -> str:
        """Résume le contenu d'un onglet"""
        total_elements = elements_result.get('total_elements', 0)
        elements_by_type = elements_result.get('elements_by_type', {})
        
        summary_parts = [f"{total_elements} éléments interactifs"]
        
        for element_type, elements in elements_by_type.items():
            if elements:
                summary_parts.append(f"{len(elements)} {element_type}")
        
        return ", ".join(summary_parts)
    
    def _generate_interaction_summary(self, session_id: str) -> Dict[str, Any]:
        """Génère un résumé des interactions pour une session"""
        try:
            elements_result = get_page_interactive_elements(session_id)
            
            return {
                'session_id': session_id,
                'current_url': elements_result.get('current_url'),
                'total_elements': elements_result.get('total_elements', 0),
                'elements_by_type': {
                    element_type: len(elements)
                    for element_type, elements in elements_result.get('elements_by_type', {}).items()
                },
                'top_recommendations': elements_result.get('interaction_suggestions', [])
            }
        except Exception as e:
            logger.error(f"❌ Erreur génération résumé: {e}")
            return {}
    
    def _format_interaction_response(self, result: Dict[str, Any], original_prompt: str) -> str:
        """Formate la réponse pour l'utilisateur"""
        if not result['success']:
            return f"❌ Je n'ai pas pu effectuer l'interaction demandée: {result.get('error', 'Erreur inconnue')}"
        
        response_parts = []
        
        if result.get('interaction_performed'):
            if result.get('tabs_explored', 0) > 0:
                response_parts.append(f"✅ J'ai exploré {result['tabs_explored']} onglets sur le site.")
                
                if 'tabs_content' in result:
                    response_parts.append("\n📋 Contenu des onglets découverts:")
                    for tab in result['tabs_content'][:5]:  # Limiter à 5
                        response_parts.append(f"• {tab['tab_name']}: {tab['content_summary']}")
            
            elif result.get('element_interacted'):
                element = result['element_interacted']
                response_parts.append(f"✅ J'ai cliqué sur '{element['text'][:50]}'")
                
                if result.get('page_changed'):
                    response_parts.append("📄 La page a changé suite à cette interaction.")
            
            elif result.get('exploration_complete'):
                total = result.get('total_interactions', 0)
                response_parts.append(f"✅ J'ai effectué une exploration complète avec {total} interactions.")
                
                if 'results' in result:
                    r = result['results']
                    response_parts.append(f"📊 Résultats: {r.get('tabs_explored', 0)} onglets, "
                                        f"{r.get('buttons_clicked', 0)} boutons, "
                                        f"{r.get('navigation_links_followed', 0)} liens de navigation")
        
        else:
            response_parts.append("🔍 J'ai analysé les éléments interactifs de la page.")
            
            if result.get('elements_discovered', 0) > 0:
                response_parts.append(f"📋 {result['elements_discovered']} éléments interactifs découverts.")
        
        # Ajouter les suggestions si disponibles
        if 'interaction_summary' in result and result['interaction_summary'].get('top_recommendations'):
            response_parts.append("\n💡 Suggestions d'interaction:")
            for suggestion in result['interaction_summary']['top_recommendations'][:3]:
                response_parts.append(f"• {suggestion.get('description', 'Action suggérée')}")
        
        return "\n".join(response_parts) if response_parts else "✅ Interaction réalisée avec succès."
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'interaction"""
        return {
            'stats': self.interaction_stats,
            'navigator_stats': self.navigator.get_statistics() if self.navigator else {}
        }
    
    def cleanup_sessions(self, max_age_hours: int = 2):
        """Nettoie les sessions anciennes"""
        try:
            # À implémenter : nettoyage automatique des sessions
            logger.info(f"🧹 Nettoyage des sessions de plus de {max_age_hours}h")
        except Exception as e:
            logger.error(f"❌ Erreur nettoyage: {e}")

# Instance globale
_gemini_interactive_adapter = None

def get_gemini_interactive_adapter(gemini_api_instance=None):
    """Retourne l'instance globale de l'adaptateur interactif"""
    global _gemini_interactive_adapter
    if _gemini_interactive_adapter is None:
        _gemini_interactive_adapter = GeminiInteractiveWebAdapter(gemini_api_instance)
    return _gemini_interactive_adapter

def initialize_gemini_interactive_adapter(gemini_api_instance=None):
    """Initialise l'adaptateur interactif Gemini"""
    adapter = get_gemini_interactive_adapter(gemini_api_instance)
    logger.info("🚀 Adaptateur Gemini Interactive initialisé")
    return adapter

def handle_gemini_interactive_request(prompt: str, user_id: int, session_id: str = None):
    """Point d'entrée principal pour les requêtes interactives Gemini"""
    adapter = get_gemini_interactive_adapter()
    return adapter.handle_interactive_request(prompt, user_id, session_id)

def detect_interactive_need(prompt: str):
    """Détecte si un prompt nécessite une interaction web"""
    adapter = get_gemini_interactive_adapter()
    return adapter.detect_interactive_request(prompt)
