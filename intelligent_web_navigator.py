"""
Simplified Intelligent Web Navigation System
This module manages autonomous navigation for artificial intelligence Google Gemini 2.0 Flash  on websites.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)

@dataclass
class NavigationSession:
    """Simplified web navigation session"""
    session_id: str
    visited_urls: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_default_factory=datetime.now)

class SimpleWebNavigator:
    """Simplified autonomous web navigator"""

    def __init__(self, scraper_instance):
        self.scraper = scraper_instance
        self.active_sessions: Dict[str, NavigationSession] = {}

        logger.info("Simplified web navigator initialized")

    def create_navigation_session(self, session_id: str) -> NavigationSession:
        """Creates a navigation session"""
        session = NavigationSession(session_id=session_id)
        self.active_sessions[session_id] = session
        return session

    def navigate_autonomously(self, start_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Simplified autonomous navigation"""
        session_id = f"nav_{int(time.time())}"
        session = self.create_navigation_session(session_id)

        results = []
        urls_to_visit = [start_url]

        logger.info(f"Autonomous navigation from {start_url}")

        while urls_to_visit and len(results) < (max_pages * 10):  # 10x more pages allowed
            url = urls_to_visit.pop(0)

            if url in session.visited_urls:
                continue

            # Scrape the page
            scraping_result = self.scraper.scrape_url(url)
            if scraping_result.success:
                session.visited_urls.add(url)

                # Analyze the page
                page_analysis = self._analyze_page_content(scraping_result)

                results.append({
                    'url': url,
                    'analysis': page_analysis,
                    'scraping_result': scraping_result
                })

                # Add some interesting links
                interesting_links = self._select_interesting_links(
                    scraping_result.links, session.visited_urls
                )
                urls_to_visit.extend(interesting_links[:3])

            # Minimal delay between pages
            time.sleep(0.05)  # 50ms only

        logger.info(f"Navigation completed: {len(results)} pages visited")
        return results

    def _analyze_page_content(self, scraping_result) -> Dict[str, Any]:
        """Simplified page content analysis"""
        content_lower = scraping_result.content.lower()

        # Determine content type
        content_type = "general"
        if any(term in content_lower for term in ['course', 'training', 'tutorial']):
            content_type = "educational"
        elif any(term in content_lower for term in ['news']):
            content_type = "news"
        elif any(term in content_lower for term in ['documentation', 'api']):
            content_type = "technical"

        # Calculate content quality
        quality_score = self._calculate_content_quality(scraping_result)

        return {
            'content_type': content_type,
            'quality_score': quality_score,
            'title': scraping_result.title,
            'content_length': len(scraping_result.content),
            'links_count': len(scraping_result.links),
            'has_useful_content': quality_score >= 5
        }

    def _select_interesting_links(self, links: List[str], visited: Set[str]) -> List[str]:
        """Selects the most interesting links"""
        interesting = []

        for link in links:
            if link in visited:
                continue

            # Simple selection criteria
            link_lower = link.lower()

            # Prioritize educational/informative links
            if any(term in link_lower for term in [
                'course', 'guide', 'tutorial', 'documentation',
                'article', 'blog', 'training'
            ]):
                interesting.append(link)
            elif len(interesting) < 10:  # Add other links if not enough
                interesting.append(link)

        return interesting[:5]  # Limit to 5 links

    def _calculate_content_quality(self, result) -> int:
        """Calculates a simple quality score"""
        score = 0

        # Content length
        if len(result.content) > 1000:
            score += 3
        elif len(result.content) > 500:
            score += 2

        # Title presence
        if result.title and len(result.title) > 10:
            score += 2

        # Number of links
        if len(result.links) > 5:
            score += 2

        # Quality indicators in content
        content_lower = result.content.lower()
        quality_indicators = ['training', 'course', 'guide', 'explanation', 'principle']
        score += sum(1 for indicator in quality_indicators if indicator in content_lower)

        return min(score, 10)

# Global instance (will be initialized with the scraper)
simple_navigator = None

def set_scraper_instance(scraper):
    """Configures the scraper instance"""
    global simple_navigator
    simple_navigator = SimpleWebNavigator(scraper)

def navigate_autonomously(start_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
    """Public interface for autonomous navigation"""
    if simple_navigator:
        return simple_navigator.navigate_autonomously(start_url, max_pages)
    return []

if __name__ == "__main__":
    print("=== Simplified Navigation System Test ===")
    print("Module loaded successfully")
