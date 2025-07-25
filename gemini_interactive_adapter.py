"""
Integration of the Interactive Navigation System with the artificial intelligence   API GOOGLE GEMINI 2.0 FLASH Adapter
This module connects the new web interaction system with the Gemini API
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

# Logging configuration
logger = logging.getLogger('GeminiInteractiveIntegration')

class GeminiInteractiveWebAdapter:
    """Adapter to integrate interactive navigation with the Gemini API"""
    
    def __init__(self, gemini_api_instance=None):
        self.gemini_api = gemini_api_instance
        self.interactive_enabled = True
        self.max_content_length = 8000
        
        # Initialize the interactive navigator
        self.navigator = initialize_interactive_navigator()
        
        # Counters and statistics
        self.interaction_stats = {
            'total_requests': 0,
            'interactive_sessions_created': 0,
            'successful_interactions': 0,
            'elements_clicked': 0,
            'tabs_explored': 0,
            'forms_interacted': 0
        }
        
        logger.info("🎯 Gemini-Interactive Navigation Adapter initialized")
    
    def detect_interactive_request(self, prompt: str) -> Dict[str, Any]:
        """
        Detects if the prompt requires interaction with web elements
        
        Args:
            prompt: The user's prompt
            
        Returns:
            Dict containing the interaction type and parameters
        """
        prompt_lower = prompt.lower()
        
        # Keywords for direct interactions
        interaction_keywords = [
            'click on', 'press on',
            'select', 'choose',
            'open the tab', 'go to the tab',
            'fill the form',
            'interact with'
        ]
        
        # Keywords for tab navigation
        tab_keywords = [
            'tab', 'tabs',
            'section', 'sections', 'category', 'categories',
            'menu', 'navigation'
        ]
        
        # Keywords for full exploration
        exploration_keywords = [
            'explore all options', 'browse all tabs',
            'visit all sections', 'analyze all menus',
            'test all functionalities'
        ]
        
        # Keywords for forms
        form_keywords = [
            'form', 'fill', 'enter',
            'search', 'login', 'connection'
        ]
        
        detection_result = {
            'requires_interaction': False,
            'interaction_type': None,
            'confidence': 0.0,
            'extracted_params': {},
            'suggested_actions': []
        }
        
        # Detect direct interactions
        if any(keyword in prompt_lower for keyword in interaction_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'direct_interaction',
                'confidence': 0.9
            })
            
            # Extract target element if mentioned
            target_element = self._extract_target_element(prompt)
            if target_element:
                detection_result['extracted_params']['target_element'] = target_element
        
        # Detect tab navigation
        elif any(keyword in prompt_lower for keyword in tab_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'tab_navigation',
                'confidence': 0.8
            })
            detection_result['suggested_actions'] = ['explore_tabs', 'click_tabs']
        
        # Detect full exploration
        elif any(keyword in prompt_lower for keyword in exploration_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'full_exploration',
                'confidence': 0.85
            })
            detection_result['suggested_actions'] = ['explore_all_elements', 'systematic_navigation']
        
        # Detect form interactions
        elif any(keyword in prompt_lower for keyword in form_keywords):
            detection_result.update({
                'requires_interaction': True,
                'interaction_type': 'form_interaction',
                'confidence': 0.7
            })
            detection_result['suggested_actions'] = ['fill_forms', 'submit_forms']
        
        # Extract URL if present
        url_match = self._extract_url_from_prompt(prompt)
        if url_match:
            detection_result['extracted_params']['url'] = url_match
        
        self.interaction_stats['total_requests'] += 1
        
        logger.info(f"🔍 Interaction detection: {detection_result['interaction_type']} "
                   f"(confidence: {detection_result['confidence']})")
        
        return detection_result
    
    def handle_interactive_request(self, prompt: str, user_id: int, 
                                 session_id: str = None) -> Dict[str, Any]:
        """
        Handles a web interaction request
        
        Args:
            prompt: The user's prompt
            user_id: User ID
            session_id: Session ID (optional)
            
        Returns:
            Dict containing the response and interaction data
        """
        try:
            # Detect the type of interaction needed
            detection = self.detect_interactive_request(prompt)
            
            if not detection['requires_interaction']:
                return {
                    'success': False,
                    'error': 'No interaction detected',
                    'fallback_required': True
                }
            
            # Generate a unique session ID if not provided
            if not session_id:
                session_id = f"interactive_{user_id}_{int(time.time())}"
            
            interaction_type = detection['interaction_type']
            
            # Process according to interaction type
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
            
            # Enrich the response with contextual information
            if result['success']:
                result['interaction_summary'] = self._generate_interaction_summary(session_id)
                result['response'] = self._format_interaction_response(result, prompt)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing interaction: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_required': True
            }
    
    def _handle_direct_interaction(self, prompt: str, session_id: str, 
                                 detection: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a direct interaction (clicking on a specific element)"""
        try:
            # Extract URL if provided
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL required for direct interaction'}
            
            # Create the interactive session
            navigation_result = create_interactive_navigation_session(
                session_id, url, goals=['direct_interaction']
            )
            
            if not navigation_result['success']:
                return navigation_result
            
            self.interaction_stats['interactive_sessions_created'] += 1
            
            # Get interactive elements
            elements_result = get_page_interactive_elements(session_id)
            
            if not elements_result['success']:
                return elements_result
            
            # Identify the target element
            target_text = detection['extracted_params'].get('target_element', '')
            target_element = self._find_best_matching_element(
                elements_result['top_interactive_elements'], target_text
            )
            
            if not target_element:
                return {
                    'success': False,
                    'error': 'Target element not found',
                    'available_elements': elements_result['top_interactive_elements'][:5]
                }
            
            # Perform the interaction
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
            logger.error(f"❌ Direct interaction error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_tab_navigation(self, prompt: str, session_id: str, 
                             detection: Dict[str, Any]) -> Dict[str, Any]:
        """Handles tab navigation"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL required for tab navigation'}
            
            # Create the session
            navigation_result = create_interactive_navigation_session(
                session_id, url, goals=['tab_navigation', 'explore_tabs']
            )
            
            if not navigation_result['success']:
                return navigation_result
            
            self.interaction_stats['interactive_sessions_created'] += 1
            
            # Get elements
            elements_result = get_page_interactive_elements(session_id)
            tabs_info = []
            tab_contents = []
            
            # Find all tabs
            for element in elements_result.get('top_interactive_elements', []):
                if element['type'] == 'tabs':
                    # Click on the tab
                    interaction_result = interact_with_web_element(
                        session_id, element['id'], 'click'
                    )
                    
                    if interaction_result['success']:
                        self.interaction_stats['tabs_explored'] += 1
                        
                        # Wait for content to load
                        time.sleep(2)
                        
                        # Capture tab content
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
                'navigation_summary': f"Explored {len(tabs_info)} tabs successfully"
            }
            
        except Exception as e:
            logger.error(f"❌ Tab navigation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_full_exploration(self, prompt: str, session_id: str, 
                               detection: Dict[str, Any]) -> Dict[str, Any]:
        """Handles full site exploration"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL required for full exploration'}
            
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
            
            # Phase 1: Explore all tabs
            elements_result = get_page_interactive_elements(session_id)
            
            for element in elements_result.get('top_interactive_elements', [])[:15]:  # Top 15 to avoid too many interactions
                if element['score'] > 0.5:  # Only relevant elements
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
                    
                    # Small delay between interactions
                    time.sleep(1.5)
            
            # Final summary
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
                'summary': f"Full exploration: {total_interactions} interactions performed"
            }
            
        except Exception as e:
            logger.error(f"❌ Full exploration error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_form_interaction(self, prompt: str, session_id: str, 
                               detection: Dict[str, Any]) -> Dict[str, Any]:
        """Handles interactions with forms"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL required for form interaction'}
            
            navigation_result = create_interactive_navigation_session(
                session_id, url, goals=['form_interaction']
            )
            
            if not navigation_result['success']:
                return navigation_result
            
            elements_result = get_page_interactive_elements(session_id)
            form_results = []
            
            # Find forms and fields
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
                'note': 'Form interaction identified (data entry not implemented for security)'
            }
            
        except Exception as e:
            logger.error(f"❌ Form interaction error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_generic_interaction(self, prompt: str, session_id: str, 
                                  detection: Dict[str, Any]) -> Dict[str, Any]:
        """Handles generic interactions"""
        try:
            url = detection['extracted_params'].get('url')
            if not url:
                return {'success': False, 'error': 'URL required'}
            
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
            logger.error(f"❌ Generic interaction error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _extract_target_element(self, prompt: str) -> str:
        """Extracts the target element mentioned in the prompt"""
        prompt_lower = prompt.lower()
        
        # Patterns to identify target elements
        patterns = [
            r'click(?:ing)? on ["\']?([^"\']+)["\']?',
            r'press(?:ing)? on ["\']?([^"\']+)["\']?',
            r'select(?:ing)? ["\']?([^"\']+)["\']?',
            r'the tab ["\']?([^"\']+)["\']?',
            r'the button ["\']?([^"\']+)["\']?',
            r'the link ["\']?([^"\']+)["\']?'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, prompt_lower)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_url_from_prompt(self, prompt: str) -> Optional[str]:
        """Extracts the URL from the prompt"""
        import re
        url_pattern = r'https?://[^\s]+|www\.[^\s]+'
        match = re.search(url_pattern, prompt)
        return match.group(0) if match else None
    
    def _find_best_matching_element(self, elements: List[Dict], target_text: str) -> Optional[Dict]:
        """Finds the element that best matches the target text"""
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
        """Summarizes the content of a tab"""
        total_elements = elements_result.get('total_elements', 0)
        elements_by_type = elements_result.get('elements_by_type', {})
        
        summary_parts = [f"{total_elements} interactive elements"]
        
        for element_type, elements in elements_by_type.items():
            if elements:
                summary_parts.append(f"{len(elements)} {element_type}")
        
        return ", ".join(summary_parts)
    
    def _generate_interaction_summary(self, session_id: str) -> Dict[str, Any]:
        """Generates an interaction summary for a session"""
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
            logger.error(f"❌ Error generating summary: {e}")
            return {}
    
    def _format_interaction_response(self, result: Dict[str, Any], original_prompt: str) -> str:
        """Formats the response for the user"""
        if not result['success']:
            return f"❌ I could not perform the requested interaction: {result.get('error', 'Unknown error')}"
        
        response_parts = []
        
        if result.get('interaction_performed'):
            if result.get('tabs_explored', 0) > 0:
                response_parts.append(f"✅ I have explored {result['tabs_explored']} tabs on the site.")
                
                if 'tabs_content' in result:
                    response_parts.append("\n📋 Discovered tab content:")
                    for tab in result['tabs_content'][:5]:  # Limit to 5
                        response_parts.append(f"• {tab['tab_name']}: {tab['content_summary']}")
            
            elif result.get('element_interacted'):
                element = result['element_interacted']
                response_parts.append(f"✅ I clicked on '{element['text'][:50]}'")
                
                if result.get('page_changed'):
                    response_parts.append("📄 The page changed after this interaction.")
            
            elif result.get('exploration_complete'):
                total = result.get('total_interactions', 0)
                response_parts.append(f"✅ I performed a full exploration with {total} interactions.")
                
                if 'results' in result:
                    r = result['results']
                    response_parts.append(f"📊 Results: {r.get('tabs_explored', 0)} tabs, "
                                        f"{r.get('buttons_clicked', 0)} buttons, "
                                        f"{r.get('navigation_links_followed', 0)} navigation links")
        
        else:
            response_parts.append("🔍 I analyzed the interactive elements of the page.")
            
            if result.get('elements_discovered', 0) > 0:
                response_parts.append(f"📋 {result['elements_discovered']} interactive elements discovered.")
        
        # Add suggestions if available
        if 'interaction_summary' in result and result['interaction_summary'].get('top_recommendations'):
            response_parts.append("\n💡 Interaction suggestions:")
            for suggestion in result['interaction_summary']['top_recommendations'][:3]:
                response_parts.append(f"• {suggestion.get('description', 'Suggested action')}")
        
        return "\n".join(response_parts) if response_parts else "✅ Interaction successfully performed."
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """Returns interaction statistics"""
        return {
            'stats': self.interaction_stats,
            'navigator_stats': self.navigator.get_statistics() if self.navigator else {}
        }
    
    def cleanup_sessions(self, max_age_hours: int = 2):
        """Cleans up old sessions"""
        try:
            # To be implemented: automatic session cleanup
            logger.info(f"🧹 Cleaning up sessions older than {max_age_hours}h")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

# Global instance
_gemini_interactive_adapter = None

def get_gemini_interactive_adapter(gemini_api_instance=None):
    """Returns the global interactive adapter instance"""
    global _gemini_interactive_adapter
    if _gemini_interactive_adapter is None:
        _gemini_interactive_adapter = GeminiInteractiveWebAdapter(gemini_api_instance)
    return _gemini_interactive_adapter

def initialize_gemini_interactive_adapter(gemini_api_instance=None):
    """Initializes the Gemini interactive adapter"""
    adapter = get_gemini_interactive_adapter(gemini_api_instance)
    logger.info("🚀 Gemini Interactive Adapter initialized")
    return adapter

def handle_gemini_interactive_request(prompt: str, user_id: int, session_id: str = None):
    """Main entry point for Gemini interactive requests"""
    adapter = get_gemini_interactive_adapter()
    return adapter.handle_interactive_request(prompt, user_id, session_id)

def detect_interactive_need(prompt: str):
    """Detects if a prompt requires web interaction"""
    adapter = get_gemini_interactive_adapter()
    return adapter.detect_interactive_request(prompt)
