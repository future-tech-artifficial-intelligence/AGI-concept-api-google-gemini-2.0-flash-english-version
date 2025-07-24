"""
Système de Navigation Web Intelligente Simplifiée
Ce module gère la navigation autonome sur les sites web.
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
    """Session de navigation web simplifiée"""
    session_id: str
    visited_urls: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)

class SimpleWebNavigator:
    """Navigateur web autonome simplifié"""

    def __init__(self, scraper_instance):
        self.scraper = scraper_instance
        self.active_sessions: Dict[str, NavigationSession] = {}

        logger.info("Navigateur web simplifié initialisé")

    def create_navigation_session(self, session_id: str) -> NavigationSession:
        """Crée une session de navigation"""
        session = NavigationSession(session_id=session_id)
        self.active_sessions[session_id] = session
        return session

    def navigate_autonomously(self, start_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Navigation autonome simplifiée"""
        session_id = f"nav_{int(time.time())}"
        session = self.create_navigation_session(session_id)

        results = []
        urls_to_visit = [start_url]

        logger.info(f"Navigation autonome depuis {start_url}")

        while urls_to_visit and len(results) < (max_pages * 10):  # 10x plus de pages autorisées
            url = urls_to_visit.pop(0)

            if url in session.visited_urls:
                continue

            # Scraper la page
            scraping_result = self.scraper.scrape_url(url)
            if scraping_result.success:
                session.visited_urls.add(url)

                # Analyser la page
                page_analysis = self._analyze_page_content(scraping_result)

                results.append({
                    'url': url,
                    'analysis': page_analysis,
                    'scraping_result': scraping_result
                })

                # Ajouter quelques liens intéressants
                interesting_links = self._select_interesting_links(
                    scraping_result.links, session.visited_urls
                )
                urls_to_visit.extend(interesting_links[:3])

            # Délai minimal entre les pages
            time.sleep(0.05)  # 50ms seulement

        logger.info(f"Navigation terminée: {len(results)} pages visitées")
        return results

    def _analyze_page_content(self, scraping_result) -> Dict[str, Any]:
        """Analyse simplifiée du contenu de page"""
        content_lower = scraping_result.content.lower()

        # Déterminer le type de contenu
        content_type = "general"
        if any(term in content_lower for term in ['cours', 'formation', 'tutorial']):
            content_type = "educational"
        elif any(term in content_lower for term in ['actualité', 'news']):
            content_type = "news"
        elif any(term in content_lower for term in ['documentation', 'api']):
            content_type = "technical"

        # Calculer la qualité du contenu
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
        """Sélectionne les liens les plus intéressants"""
        interesting = []

        for link in links:
            if link in visited:
                continue

            # Critères de sélection simple
            link_lower = link.lower()

            # Privilégier les liens éducatifs/informatifs
            if any(term in link_lower for term in [
                'cours', 'guide', 'tutorial', 'documentation',
                'article', 'blog', 'formation'
            ]):
                interesting.append(link)
            elif len(interesting) < 10:  # Ajouter d'autres liens si pas assez
                interesting.append(link)

        return interesting[:5]  # Limiter à 5 liens

    def _calculate_content_quality(self, result) -> int:
        """Calcule un score de qualité simple"""
        score = 0

        # Longueur du contenu
        if len(result.content) > 1000:
            score += 3
        elif len(result.content) > 500:
            score += 2

        # Présence de titre
        if result.title and len(result.title) > 10:
            score += 2

        # Nombre de liens
        if len(result.links) > 5:
            score += 2

        # Indicateurs de qualité dans le contenu
        content_lower = result.content.lower()
        quality_indicators = ['formation', 'cours', 'guide', 'explication', 'principe']
        score += sum(1 for indicator in quality_indicators if indicator in content_lower)

        return min(score, 10)

# Instance globale (sera initialisée avec le scraper)
simple_navigator = None

def set_scraper_instance(scraper):
    """Configure l'instance de scraper"""
    global simple_navigator
    simple_navigator = SimpleWebNavigator(scraper)

def navigate_autonomously(start_url: str, max_pages: int = 5) -> List[Dict[str, Any]]:
    """Interface publique pour la navigation autonome"""
    if simple_navigator:
        return simple_navigator.navigate_autonomously(start_url, max_pages)
    return []

if __name__ == "__main__":
    print("=== Test du Système de Navigation Simplifiée ===")
    print("Module chargé avec succès")