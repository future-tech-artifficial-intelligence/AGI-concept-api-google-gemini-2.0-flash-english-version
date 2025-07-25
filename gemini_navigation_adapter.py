"""
Advanced Web Navigation System Integration with the Google Gemini 2.0 Flash AI Adapter
This module connects the new navigation system with the existing Google Gemini 2.0 Flash AI adapter.
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

# Logging configuration
logger = logging.getLogger('GeminiAdapterIntegration')

class GeminiWebNavigationAdapter:
    """Adapter to integrate web navigation with the existing Google Gemini 2.0 Flash AI API"""
    
    def __init__(self, gemini_api_instance=None):
        self.gemini_api = gemini_api_instance
        self.navigation_enabled = True
        self.max_content_length = 8000  # Limit for Gemini
        
        # Initialize web integration
        initialize_gemini_web_integration()
        
        # Counters and statistics
        self.navigation_stats = {
            'total_requests': 0,
            'successful_navigations': 0,
            'content_extractions': 0,
            'searches_performed': 0
        }
        
        logger.info("🔗 Google Gemini 2.0 Flash AI Navigation Adapter initialized")
    
    def detect_navigation_request(self, prompt: str) -> Dict[str, Any]:
        """
        Detects if the prompt requires advanced web navigation
        
        Args:
            prompt: The user's prompt
            
        Returns:
            Dict containing the navigation type and parameters
        """
        prompt_lower = prompt.lower()
        
        # Keywords for deep navigation
        deep_navigation_keywords = [
            'explore the site', 'navigate through', 'browse the site', 'visit all pages',
            'full site analysis', 'deep navigation', 'explore in detail'
        ]
        
        # Keywords for specific extraction
        extraction_keywords = [
            'extract content from', 'analyze this page', 'retrieve information from',
            'content of this url', 'page details'
        ]
        
        # Keywords for search and navigation
        search_navigation_keywords = [
            'search and navigate', 'find and explore', 'search and analyze',
            'detailed search', 'complete information on'
        ]
        
        # Keywords for user journey
        user_journey_keywords = [
            'user journey', 'user experience', 'user navigation',
            'as a user', 'simulate a purchase', 'purchase process'
        ]
        
        detection_result = {
            'requires_navigation': False,
            'navigation_type': None,
            'confidence': 0.0,
            'extracted_params': {}
        }
        
        # Detect navigation type
        if any(keyword in prompt_lower for keyword in deep_navigation_keywords):
            detection_result.update({
                'requires_navigation': True,
                'navigation_type': 'deep_navigation',
                'confidence': 0.9
            })
            
            # Extract URL if present
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
        
        # Detect general web search requests that could benefit from navigation
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
        Handles a web navigation request
        
        Args:
            prompt: The user's prompt
            user_id: User ID
            session_id: Session ID
            
        Returns:
            Navigation result formatted for Google Gemini 2.0 Flash AI
        """
        if not self.navigation_enabled:
            return {
                'success': False,
                'error': 'Web navigation disabled',
                'fallback_required': True
            }
        
        self.navigation_stats['total_requests'] += 1
        
        try:
            # Detect navigation type
            detection = self.detect_navigation_request(prompt)
            
            if not detection['requires_navigation']:
                return {
                    'success': False,
                    'error': 'Navigation not detected',
                    'fallback_required': True
                }
            
            logger.info(f"🎯 Navigation detected: {detection['navigation_type']} (confidence: {detection['confidence']})")
            
            # Process according to navigation type
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
                    'error': f'Unsupported navigation type: {detection["navigation_type"]}',
                    'fallback_required': True
                }
            
            # Format for Google Gemini 2.0 Flash AI
            if result.get('success', False):
                self.navigation_stats['successful_navigations'] += 1
                gemini_response = self._format_for_gemini(result, detection['navigation_type'], prompt)
                
                logger.info(f"✅ Navigation successful: {detection['navigation_type']}")
                return gemini_response
            else:
                logger.warning(f"⚠️ Navigation failed: {result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': result.get('error', 'Navigation error'),
                    'fallback_required': True
                }
                
        except Exception as e:
            logger.error(f"❌ Error during navigation: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'fallback_required': True
            }
    
    def _handle_search_and_navigate(self, detection: Dict, prompt: str, user_id: int) -> Dict[str, Any]:
        """Handles a search and navigate request"""
        query = detection['extracted_params'].get('query', prompt)
        user_context = f"User {user_id} - Advanced Search"
        
        logger.info(f"🔍 Searching and navigating: {query}")
        
        result = search_web_for_gemini(query, user_context)
        self.navigation_stats['searches_performed'] += 1
        
        return result
    
    def _handle_content_extraction(self, detection: Dict, prompt: str) -> Dict[str, Any]:
        """Handles a content extraction request"""
        url = detection['extracted_params'].get('url')
        
        if not url:
            return {
                'success': False,
                'error': 'URL not found in prompt'
            }
        
        logger.info(f"🎯 Content extraction: {url}")
        
        # Determine extraction requirements based on the prompt
        requirements = self._determine_extraction_requirements(prompt)
        
        result = extract_content_for_gemini(url, requirements)
        self.navigation_stats['content_extractions'] += 1
        
        return result
    
    def _handle_deep_navigation(self, detection: Dict, prompt: str) -> Dict[str, Any]:
        """Handles a deep navigation request"""
        start_url = detection['extracted_params'].get('start_url')
        
        if not start_url:
            return {
                'success': False,
                'error': 'Starting URL not found in prompt'
            }
        
        logger.info(f"🚀 Deep navigation: {start_url}")
        
        # Default parameters or extracted from the prompt
        max_depth = self._extract_number_from_prompt(prompt, 'depth', 3)
        max_pages = self._extract_number_from_prompt(prompt, 'pages', 10)
        
        nav_path = navigate_website_deep(start_url, max_depth, max_pages)
        
        # Convert to compatible format
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
        """Handles a user journey request"""
        start_url = detection['extracted_params'].get('start_url')
        user_intent = detection['extracted_params'].get('user_intent', 'explore')
        
        if not start_url:
            return {
                'success': False,
                'error': 'Starting URL not found in prompt'
            }
        
        logger.info(f"👤 User journey: {user_intent} from {start_url}")
        
        result = simulate_user_journey(start_url, user_intent)
        return result
    
    def _format_for_gemini(self, result: Dict[str, Any], navigation_type: str, 
                          original_prompt: str) -> Dict[str, Any]:
        """Formats the result for the Google Gemini 2.0 Flash AI API"""
        
        # Create a summary adapted to the navigation type
        if navigation_type == 'search_and_navigate':
            summary = self._create_search_summary(result)
            
        elif navigation_type == 'content_extraction':
            summary = self._create_extraction_summary(result)
            
        elif navigation_type == 'deep_navigation':
            summary = self._create_navigation_summary(result)
            
        elif navigation_type == 'user_journey':
            summary = self._create_journey_summary(result)
            
        else:
            summary = "Web navigation successfully performed."
        
        # Prepare content for Google Gemini 2.0 Flash AI
        gemini_content = {
            'web_navigation_summary': summary,
            'navigation_type': navigation_type,
            'data_extracted': True,
            'content_length': len(str(result)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add specific details based on type
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
        """Creates a search summary for Google Gemini 2.0 Flash AI"""
        if not result.get('success', False):
            return "❌ Web search failed."
        
        search_summary = result.get('search_summary', {})
        sites_navigated = search_summary.get('sites_navigated', 0)
        pages_visited = search_summary.get('total_pages_visited', 0)
        
        summary = f"🌐 **Web search successfully performed!**\n\n"
        summary += f"I navigated {sites_navigated} websites and analyzed {pages_visited} pages.\n\n"
        
        if 'content_synthesis' in result:
            summary += f"**Synthesis of information found:**\n{result['content_synthesis']}\n\n"
        
        if 'aggregated_keywords' in result and result['aggregated_keywords']:
            keywords = ', '.join(result['aggregated_keywords'][:10])
            summary += f"**Identified keywords:** {keywords}\n\n"
        
        summary += "Detailed information has been integrated into my knowledge base."
        
        return summary
    
    def _create_extraction_summary(self, result: Dict[str, Any]) -> str:
        """Creates an extraction summary for Google Gemini 2.0 Flash AI"""
        if not result.get('success', False):
            return f"❌ Failed to extract content from URL: {result.get('error', 'Unknown error')}"
        
        summary = f"📄 **Content successfully extracted!**\n\n"
        summary += f"**Title:** {result.get('title', 'Not specified')}\n"
        summary += f"**URL:** {result.get('url', 'Not specified')}\n"
        summary += f"**Language:** {result.get('language', 'Not detected')}\n"
        summary += f"**Quality Score:** {result.get('content_quality_score', 0):.1f}/10\n\n"
        
        if 'summary' in result:
            summary += f"**Summary:**\n{result['summary']}\n\n"
        
        if 'keywords' in result and result['keywords']:
            keywords = ', '.join(result['keywords'][:8])
            summary += f"**Keywords:** {keywords}\n\n"
        
        return summary
    
    def _create_navigation_summary(self, result: Dict[str, Any]) -> str:
        """Creates a navigation summary for Google Gemini 2.0 Flash AI"""
        if not result.get('success', False):
            return "❌ Deep navigation failed."
        
        nav_summary = result.get('navigation_summary', {})
        pages_visited = nav_summary.get('pages_visited', 0)
        depth = nav_summary.get('navigation_depth', 0)
        
        summary = f"🚀 **Deep navigation performed!**\n\n"
        summary += f"I explored {pages_visited} pages with a navigation depth of {depth} levels.\n\n"
        
        if 'visited_pages' in result and result['visited_pages']:
            summary += "**Most relevant pages:**\n"
            top_pages = sorted(result['visited_pages'], 
                             key=lambda x: x['content_quality_score'], reverse=True)[:3]
            
            for i, page in enumerate(top_pages, 1):
                summary += f"{i}. **{page['title']}** (Score: {page['content_quality_score']:.1f})\n"
                summary += f"   📄 {page['summary'][:150]}...\n\n"
        
        return summary
    
    def _create_journey_summary(self, result: Dict[str, Any]) -> str:
        """Creates a user journey summary for Google Gemini 2.0 Flash AI"""
        if not result.get('success', False):
            return f"❌ User journey simulation failed: {result.get('error', 'Unknown error')}"
        
        pages_visited = result.get('pages_visited', 0)
        user_intent = result.get('user_intent', 'explore')
        
        intent_names = {
            'buy': 'purchase',
            'learn': 'learning',
            'contact': 'contact',
            'explore': 'exploration'
        }
        
        intent_text = intent_names.get(user_intent, user_intent)
        
        summary = f"👤 **User journey successfully simulated!**\n\n"
        summary += f"I simulated a **{intent_text}** journey across {pages_visited} pages.\n\n"
        
        if 'journey_analysis' in result:
            analysis = result['journey_analysis']
            satisfaction = analysis.get('intent_satisfaction', 0) * 100
            summary += f"**Intent Satisfaction:** {satisfaction:.1f}%\n\n"
        
        if 'key_pages' in result and result['key_pages']:
            summary += "**Key pages identified:**\n"
            for i, page in enumerate(result['key_pages'][:3], 1):
                summary += f"{i}. **{page['title']}**\n"
                summary += f"   📄 {page['summary'][:100]}...\n\n"
        
        return summary
    
    # Utility methods
    def _extract_url_from_prompt(self, prompt: str) -> Optional[str]:
        """Extracts a URL from the prompt"""
        import re
        url_pattern = r'https?://[^\s<>"{\|}\\^`\[\]]+'
        urls = re.findall(url_pattern, prompt)
        return urls[0] if urls else None
    
    def _extract_search_query_from_prompt(self, prompt: str) -> Optional[str]:
        """Extracts a search query from the prompt"""
        # Patterns to extract the query
        patterns = [
            r'search\s+(?:and\s+navigate\s+)?["\']([^"\']+)["\']',
            r'find\s+(?:and\s+explore\s+)?["\']([^"\']+)["\']',
            r'analyze\s+(?:and\s+examine\s+)?["\']([^"\']+)["\']',
            r'search\s+(?:and\s+navigate\s+)?(?:on\s+)?(.+?)(?:\s+and\s|$)',
            r'find\s+(?:and\s+explore\s+)?(?:on\s+)?(.+?)(?:\s+and\s|$)'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_user_intent_from_prompt(self, prompt: str) -> str:
        """Extracts the user intent from the prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['buy', 'purchase', 'order', 'price']):
            return 'buy'
        elif any(word in prompt_lower for word in ['learn', 'training', 'course', 'tutorial']):
            return 'learn'
        elif any(word in prompt_lower for word in ['contact', 'support', 'help']):
            return 'contact'
        else:
            return 'explore'
    
    def _extract_number_from_prompt(self, prompt: str, context: str, default: int) -> int:
        """Extracts a number from the prompt based on context"""
        import re
        
        # Patterns for different contexts
        if context == 'depth':
            patterns = [r'depth\s+(?:of\s+)?(\d+)', r'(\d+)\s+levels?']
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
        """Determines extraction requirements based on the prompt"""
        prompt_lower = prompt.lower()
        requirements = ['summary']  # Always include the summary
        
        if any(word in prompt_lower for word in ['detail', 'complete', 'all', 'entire']):
            requirements.extend(['details', 'structure'])
        
        if any(word in prompt_lower for word in ['links', 'link', 'url']):
            requirements.append('links')
        
        if any(word in prompt_lower for word in ['images', 'photos', 'illustrations']):
            requirements.append('images')
        
        if any(word in prompt_lower for word in ['navigation', 'menu', 'navigate']):
            requirements.append('navigation')
        
        if any(word in prompt_lower for word in ['metadata', 'properties']):
            requirements.append('metadata')
        
        return list(set(requirements))  # Remove duplicates
    
    def _is_general_web_search(self, prompt: str) -> bool:
        """Determines if it's a general web search that could benefit from navigation"""
        prompt_lower = prompt.lower()
        
        # Keywords indicating an information search
        search_indicators = [
            'what is', 'how to', 'why', 'where to find', 'information about',
            'explanation of', 'definition of', 'guide for', 'tutorial on'
        ]
        
        # Check if it's a question or information request
        if any(indicator in prompt_lower for indicator in search_indicators):
            return True
        
        # Check if it's a query that ends with a question mark
        if prompt.strip().endswith('?'):
            return True
        
        return False
    
    def get_navigation_stats(self) -> Dict[str, Any]:
        """Retrieves navigation statistics"""
        return {
            'navigation_stats': self.navigation_stats.copy(),
            'navigation_enabled': self.navigation_enabled,
            'max_content_length': self.max_content_length
        }
    
    def enable_navigation(self):
        """Enables web navigation"""
        self.navigation_enabled = True
        logger.info("🟢 Web navigation enabled")
    
    def disable_navigation(self):
        """Disables web navigation"""
        self.navigation_enabled = False
        logger.info("🔴 Web navigation disabled")

# Global instance
gemini_navigation_adapter = None

def initialize_gemini_navigation_adapter(gemini_api_instance=None):
    """Initializes the Google Gemini 2.0 Flash AI navigation adapter"""
    global gemini_navigation_adapter
    gemini_navigation_adapter = GeminiWebNavigationAdapter(gemini_api_instance)
    logger.info("🔗 Google Gemini 2.0 Flash AI Navigation Adapter initialized")

def handle_gemini_navigation_request(prompt: str, user_id: int = 1, session_id: str = None) -> Dict[str, Any]:
    """Public interface for Google Gemini 2.0 Flash AI navigation requests"""
    if not gemini_navigation_adapter:
        initialize_gemini_navigation_adapter()
    
    return gemini_navigation_adapter.handle_navigation_request(prompt, user_id, session_id)

def detect_navigation_need(prompt: str) -> Dict[str, Any]:
    """Public interface for navigation detection"""
    if not gemini_navigation_adapter:
        initialize_gemini_navigation_adapter()
    
    return gemini_navigation_adapter.detect_navigation_request(prompt)

if __name__ == "__main__":
    print("=== Google Gemini 2.0 Flash AI Navigation Adapter Test ===")
    
    # Initialize
    initialize_gemini_navigation_adapter()
    
    # Detection tests
    test_prompts = [
        "Search and navigate artificial intelligence",
        "Extract content from https://example.com",
        "Explore the site https://wikipedia.org in depth",
        "Simulate a purchase journey on https://shop.example.com",
        "What is machine learning?"
    ]
    
    print("🧪 Navigation detection tests:")
    for prompt in test_prompts:
        detection = detect_navigation_need(prompt)
        print(f"  📝 '{prompt}'")
        print(f"     → Type: {detection['navigation_type']}, Confidence: {detection['confidence']}")
        print(f"     → Parameters: {detection['extracted_params']}")
        print()
