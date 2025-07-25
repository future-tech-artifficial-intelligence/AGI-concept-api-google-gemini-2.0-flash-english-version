"""
Advanced Web Navigation System Integration with Searx and artificial intelligence Google Gemini 2.0 Flash AI
This module connects the advanced navigator with the Google Gemini 2.0 Flash AI API and Searx
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

# Logger configuration
logger = logging.getLogger('GeminiWebIntegration')

class GeminiWebNavigationIntegration:
    """Web navigation integration for the Google Gemini 2.0 Flash AI API"""
    
    def __init__(self, searx_interface=None):
        self.navigator = AdvancedWebNavigator()
        self.searx_interface = searx_interface
        
        # Configuration
        self.max_search_results = 5
        self.max_navigation_depth = 3
        self.max_pages_per_site = 8
        self.content_quality_threshold = 3.0
        
        # Recent search cache
        self.search_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        # Directory for Google Gemini 2.0 Flash AI reports
        self.reports_dir = Path("data/gemini_web_reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Google Gemini 2.0 Flash AI-Navigation Integration initialized")
    
    def search_and_navigate_for_gemini(self, query: str, user_context: str = "") -> Dict[str, Any]:
        """
        Performs a comprehensive search and navigation for Google Gemini 2.0 Flash AI
        
        Args:
            query: Search query
            user_context: User context to customize the search
            
        Returns:
            Dictionary with structured content for Google Gemini 2.0 Flash AI
        """
        search_id = f"search_{int(time.time())}"
        logger.info(f"🔍 Google Gemini 2.0 Flash AI search: {query} (ID: {search_id})")
        
        # Check cache
        cache_key = f"{query}_{hash(user_context)}"
        if cache_key in self.search_cache:
            cached_result, cached_time = self.search_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                logger.info("📋 Result retrieved from cache")
                return cached_result
        
        try:
            # Phase 1: Search with Searx
            search_results = self._perform_searx_search(query)
            
            if not search_results:
                logger.warning("⚠️ No search results")
                return self._create_empty_result(query, "No results found")
            
            # Phase 2: Navigate through results
            navigation_results = []
            total_content_extracted = 0
            
            for i, search_result in enumerate(search_results[:self.max_search_results]):
                try:
                    logger.info(f"🚀 Navigating site {i+1}: {search_result['url']}")
                    
                    # Deep navigation
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
                        
                        logger.info(f"✅ Site navigated: {len(nav_path.visited_pages)} pages, {nav_path.total_content_extracted} characters")
                    
                except Exception as e:
                    logger.error(f"❌ Error navigating site {search_result['url']}: {str(e)}")
                    continue
            
            # Phase 3: Synthesis for Google Gemini 2.0 Flash AI
            gemini_report = self._create_gemini_report(
                query=query,
                user_context=user_context,
                search_results=search_results,
                navigation_results=navigation_results,
                search_id=search_id
            )
            
            # Cache the result
            self.search_cache[cache_key] = (gemini_report, datetime.now())
            
            # Save the report
            self._save_gemini_report(gemini_report, search_id)
            
            logger.info(f"🎯 Search completed: {len(navigation_results)} sites navigated, {total_content_extracted} characters extracted")
            
            return gemini_report
            
        except Exception as e:
            logger.error(f"❌ Error in Google Gemini 2.0 Flash AI search: {str(e)}")
            return self._create_error_result(query, str(e))
    
    def extract_specific_content(self, url: str, content_requirements: List[str]) -> Dict[str, Any]:
        """
        Extracts specific content from a URL according to requirements
        
        Args:
            url: URL to analyze
            content_requirements: List of required content types
                                ['summary', 'details', 'links', 'images', 'structure']
        """
        logger.info(f"🎯 Specific extraction: {url}")
        
        try:
            # Content extraction
            page_content = self.navigator.extract_page_content(url)
            
            if not page_content.success:
                return {
                    'success': False,
                    'error': page_content.error_message,
                    'url': url
                }
            
            # Prepare response according to requirements
            extracted_content = {
                'success': True,
                'url': url,
                'title': page_content.title,
                'extraction_timestamp': page_content.extraction_timestamp.isoformat(),
                'content_quality_score': page_content.content_quality_score,
                'language': page_content.language
            }
            
            # Add content according to requirements
            if 'summary' in content_requirements:
                extracted_content['summary'] = page_content.summary
            
            if 'details' in content_requirements:
                extracted_content['main_content'] = page_content.main_content
                extracted_content['cleaned_text'] = page_content.cleaned_text[:2000]  # Limit for Google Gemini 2.0 Flash AI
            
            if 'links' in content_requirements:
                extracted_content['links'] = page_content.links[:20]  # Top 20 links
            
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
            logger.error(f"❌ Specific extraction error {url}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def navigate_user_journey(self, start_url: str, user_intent: str) -> Dict[str, Any]:
        """
        Simulates a user journey on a site based on their intent
        
        Args:
            start_url: Starting URL
            user_intent: User's intent ('buy', 'learn', 'contact', 'explore')
        """
        logger.info(f"👤 User journey: {user_intent} from {start_url}")
        
        try:
            # Configuration according to intent
            intent_config = {
                'buy': {
                    'keywords': ['price', 'buy', 'order', 'cart', 'product'],
                    'max_depth': 4,
                    'max_pages': 15
                },
                'learn': {
                    'keywords': ['guide', 'tutorial', 'training', 'course', 'learn'],
                    'max_depth': 3,
                    'max_pages': 10
                },
                'contact': {
                    'keywords': ['contact', 'support', 'help', 'phone', 'email'],
                    'max_depth': 2,
                    'max_pages': 8
                },
                'explore': {
                    'keywords': ['see', 'discover', 'more', 'detail', 'information'],
                    'max_depth': 3,
                    'max_pages': 12
                }
            }
            
            config = intent_config.get(user_intent, intent_config['explore'])
            
            # Navigation with intent filter
            def intent_filter(page_content: WebPageContent) -> bool:
                text_lower = page_content.cleaned_text.lower()
                title_lower = page_content.title.lower()
                
                # Check for presence of intent keywords
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
            
            # Analyze the journey
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
            logger.error(f"❌ User journey error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'start_url': start_url,
                'user_intent': user_intent
            }
    
    def _perform_searx_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs a search with Searx"""
        if not self.searx_interface:
            logger.warning("⚠️ Searx interface not available, using simulated results")
            return self._simulate_search_results(query)
        
        try:
            # Use the Searx interface
            search_results = self.searx_interface.search(query, categories=['general'], max_results=10)
            
            # Convert to expected format
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
            logger.error(f"❌ Searx search error: {str(e)}")
            return self._simulate_search_results(query)
    
    def _simulate_search_results(self, query: str) -> List[Dict[str, Any]]:
        """Simulates search results (fallback)"""
        # Simulated results based on the query
        base_results = [
            {
                'title': f'Result 1 for {query}',
                'url': 'https://en.wikipedia.org/wiki/Artificial_intelligence',
                'content': f'Detailed information about {query}',
                'engine': 'wikipedia',
                'score': 0.9
            },
            {
                'title': f'Complete Guide on {query}',
                'url': 'https://www.nature.com/articles/d41586-023-02202-0',
                'content': f'Guide and explanation of {query}',
                'engine': 'nature-journal',
                'score': 0.8
            }
        ]
        
        return base_results
    
    def _quality_content_filter(self, page_content: WebPageContent) -> bool:
        """Filters pages based on their quality"""
        return (page_content.content_quality_score >= self.content_quality_threshold and
                len(page_content.cleaned_text) > 200 and
                page_content.title != "Untitled Page")
    
    def _create_gemini_report(self, query: str, user_context: str, 
                            search_results: List[Dict], 
                            navigation_results: List[Dict],
                            search_id: str) -> Dict[str, Any]:
        """Creates a structured report for Google Gemini 2.0 Flash AI"""
        
        # Extract best content
        best_content = []
        all_keywords = set()
        total_pages = 0
        
        for nav_result in navigation_results:
            nav_path = nav_result['navigation_path']
            total_pages += len(nav_path.visited_pages)
            
            for page in nav_path.visited_pages:
                if page.content_quality_score >= 4.0:  # Only the best content
                    best_content.append({
                        'url': page.url,
                        'title': page.title,
                        'summary': page.summary,
                        'main_content': page.main_content[:1000],  # Limit for Google Gemini 2.0 Flash AI
                        'keywords': page.keywords,
                        'quality_score': page.content_quality_score,
                        'language': page.language
                    })
                    all_keywords.update(page.keywords)
        
        # Create an intelligent synthesis
        content_synthesis = self._synthesize_content(best_content)
        
        # Final report
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
            'best_content': best_content[:5],  # Top 5 contents
            'aggregated_keywords': list(all_keywords)[:20],  # Top 20 keywords
            'navigation_insights': self._generate_navigation_insights(navigation_results),
            'recommended_actions': self._generate_recommendations(query, best_content),
            'success': True
        }
    
    def _synthesize_content(self, content_list: List[Dict]) -> str:
        """Synthesizes extracted content"""
        if not content_list:
            return "No quality content found."
        
        # Combine summaries
        all_summaries = [content['summary'] for content in content_list if content['summary']]
        
        # Extract key information
        key_info = []
        for content in content_list:
            key_info.append(f"• {content['title']}: {content['summary'][:150]}...")
        
        synthesis = f"Synthesis based on {len(content_list)} quality pages:\n\n"
        synthesis += "\n".join(key_info[:5])  # Top 5
        
        return synthesis
    
    def _generate_navigation_insights(self, navigation_results: List[Dict]) -> List[str]:
        """Generates navigation insights"""
        insights = []
        
        for nav_result in navigation_results:
            nav_path = nav_result['navigation_path']
            site_domain = urlparse(nav_result['search_result']['url']).netloc
            
            insights.append(f"Site {site_domain}: {len(nav_path.visited_pages)} pages explored, "
                          f"depth {nav_path.navigation_depth}")
        
        return insights
    
    def _generate_recommendations(self, query: str, content_list: List[Dict]) -> List[str]:
        """Generates recommendations based on content"""
        recommendations = []
        
        if not content_list:
            recommendations.append("Try searching with different keywords")
            return recommendations
        
        # Quality-based recommendations
        high_quality_count = sum(1 for c in content_list if c['quality_score'] >= 7.0)
        if high_quality_count > 0:
            recommendations.append(f"{high_quality_count} very high-quality sources identified")
        
        # Language recommendations
        languages = set(c['language'] for c in content_list if c['language'])
        if 'fr' in languages and 'en' in languages:
            recommendations.append("Content available in French and English")
        
        return recommendations
    
    def _analyze_user_journey(self, nav_path: NavigationPath, intent: str, keywords: List[str]) -> Dict[str, Any]:
        """Analyzes the user journey"""
        analysis = {
            'intent_satisfaction': 0.0,
            'journey_efficiency': 0.0,
            'content_relevance': 0.0,
            'key_findings': []
        }
        
        if not nav_path.visited_pages:
            return analysis
        
        # Calculate intent satisfaction
        intent_pages = 0
        for page in nav_path.visited_pages:
            text_lower = page.cleaned_text.lower()
            if any(keyword in text_lower for keyword in keywords):
                intent_pages += 1
        
        analysis['intent_satisfaction'] = intent_pages / len(nav_path.visited_pages)
        
        # Calculate journey efficiency
        avg_quality = sum(p.content_quality_score for p in nav_path.visited_pages) / len(nav_path.visited_pages)
        analysis['journey_efficiency'] = min(avg_quality / 10.0, 1.0)
        
        # Content relevance
        total_content = sum(len(p.cleaned_text) for p in nav_path.visited_pages)
        analysis['content_relevance'] = min(total_content / 10000, 1.0)  # Normalize
        
        # Key findings
        analysis['key_findings'] = [
            f"Journey of {len(nav_path.visited_pages)} pages",
            f"Navigation depth: {nav_path.navigation_depth}",
            f"Average quality score: {avg_quality:.1f}/10"
        ]
        
        return analysis
    
    def _extract_key_pages(self, nav_path: NavigationPath, keywords: List[str]) -> List[Dict[str, Any]]:
        """Extracts key pages from the journey"""
        key_pages = []
        
        for page in nav_path.visited_pages:
            # Score based on quality + keyword relevance
            keyword_score = sum(1 for keyword in keywords 
                              if keyword in page.cleaned_text.lower())
            
            total_score = page.content_quality_score + keyword_score
            
            if total_score >= 5.0:  # Threshold for key pages
                key_pages.append({
                    'url': page.url,
                    'title': page.title,
                    'summary': page.summary,
                    'score': total_score,
                    'keywords_found': keyword_score
                })
        
        # Sort by score and return the top
        key_pages.sort(key=lambda x: x['score'], reverse=True)
        return key_pages[:5]
    
    def _create_empty_result(self, query: str, reason: str) -> Dict[str, Any]:
        """Creates an empty result"""
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
            'content_synthesis': f"No results found for: {query}",
            'best_content': [],
            'aggregated_keywords': [],
            'navigation_insights': [],
            'recommended_actions': ["Try with different keywords", "Check spelling"]
        }
    
    def _create_error_result(self, query: str, error: str) -> Dict[str, Any]:
        """Creates an error result"""
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
            'content_synthesis': f"Error during search: {error}",
            'best_content': [],
            'aggregated_keywords': [],
            'navigation_insights': [],
            'recommended_actions': ["Try again later", "Check connection"]
        }
    
    def _save_gemini_report(self, report: Dict[str, Any], search_id: str):
        """Saves the report for Google Gemini 2.0 Flash AI"""
        try:
            filename = f"gemini_report_{search_id}.json"
            filepath = self.reports_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"📊 Google Gemini 2.0 Flash AI report saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")

# Global instance
gemini_web_integration = None

def initialize_gemini_web_integration(searx_interface=None):
    """Initializes the Google Gemini 2.0 Flash AI-Web Integration"""
    global gemini_web_integration
    gemini_web_integration = GeminiWebNavigationIntegration(searx_interface)
    logger.info("🚀 Google Gemini 2.0 Flash AI-Web Integration initialized")

def search_web_for_gemini(query: str, user_context: str = "") -> Dict[str, Any]:
    """Public interface for Google Gemini 2.0 Flash AI"""
    if not gemini_web_integration:
        initialize_gemini_web_integration()
    
    return gemini_web_integration.search_and_navigate_for_gemini(query, user_context)

def extract_content_for_gemini(url: str, requirements: List[str] = None) -> Dict[str, Any]:
    """Public interface for specific extraction"""
    if not gemini_web_integration:
        initialize_gemini_web_integration()
    
    if requirements is None:
        requirements = ['summary', 'details', 'links']
    
    return gemini_web_integration.extract_specific_content(url, requirements)

def simulate_user_journey(start_url: str, intent: str) -> Dict[str, Any]:
    """Public interface for user journey"""
    if not gemini_web_integration:
        initialize_gemini_web_integration()
    
    return gemini_web_integration.navigate_user_journey(start_url, intent)

if __name__ == "__main__":
    print("=== Google Gemini 2.0 Flash AI-Navigation Integration Test ===")
    
    # Initialize
    initialize_gemini_web_integration()
    
    # Search test
    test_query = "artificial intelligence machine learning"
    print(f"Search test: {test_query}")
    
    result = search_web_for_gemini(test_query, "user interested in AI")
    
    if result['success']:
        print(f"✅ Search successful: {result['search_summary']['total_pages_visited']} pages visited")
        print(f"✅ Synthesis: {result['content_synthesis'][:200]}...")
        print(f"✅ Keywords: {result['aggregated_keywords'][:10]}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
