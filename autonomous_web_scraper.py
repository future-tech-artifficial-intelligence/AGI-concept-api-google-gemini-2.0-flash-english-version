"""
**Universal Autonomous Web Scraping System for Artificial Intelligence**
This module allowed artificial intelligence to autonomously access the Internet to obtain real links from any website. The extracted data was saved in text files. **
This system is no longer used; now artificial intelligence API GOOGLE GEMINI 2.0 FLASH uses the Searx search engine for greater precision.**
"""

import requests
import asyncio
import aiohttp
import time
import json
import hashlib
import logging
import re
import ssl
from urllib.parse import urljoin, urlparse, parse_qs, quote, unquote
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Result of a scraping operation"""
    url: str
    content: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    links: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""
    response_time: float = 0.0
    content_type: str = ""
    status_code: int = 200

class UniversalWebScraper:
    """Universal web scraping system for AI"""

    def __init__(self):
        self.session = requests.Session()

        # Directories for saving data
        self.data_dir = Path("data/autonomous_web_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.extracted_text_dir = self.data_dir / "extracted_text"
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)

        self.real_links_dir = self.data_dir / "real_links"
        self.real_links_dir.mkdir(parents=True, exist_ok=True)

        # Default headers configuration
        self.default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8', # Changed to en-US primary
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        self.session.headers.update(self.default_headers)

        # History of visited URLs
        self.visited_urls: Set[str] = set()

        # Universal configuration
        self.universal_config = {
            "max_concurrent_requests": 20,
            "default_delay": 0.2,
            "max_pages_per_search": 50,
            "auto_save_interval": 5
        }

        # Search engines and supported sites
        self.search_engines = {
            "duckduckgo": "https://duckduckgo.com/?q={query}",
            "bing": "https://www.bing.com/search?q={query}",
            "yandex": "https://yandex.com/search/?text={query}"
        }

        # Specialized sites for different search types
        self.specialized_sites = {
            "real_estate": [ # Changed from "immobilier"
                "leboncoin.fr",
                "seloger.com",
                "pap.fr",
                "logic-immo.com",
                "bienici.com"
            ],
            "ecommerce": [
                "amazon.fr",
                "cdiscount.com",
                "fnac.com",
                "darty.com",
                "boulanger.com"
            ],
            "news": [ # Changed from "actualites"
                "lemonde.fr",
                "lefigaro.fr",
                "liberation.fr",
                "franceinfo.fr",
                "20minutes.fr"
            ],
            "jobs": [ # Changed from "emploi"
                "pole-emploi.fr",
                "indeed.fr",
                "monster.fr",
                "apec.fr",
                "linkedin.com"
            ],
            "education": [ # Changed from "formation"
                "coursera.org",
                "udemy.com",
                "openclassrooms.com",
                "fun-mooc.fr",
                "edx.org"
            ]
        }

        # Session counter
        self.session_counter = 0

        logger.info("Universal Web Scraping System initialized")

    def search_real_links_universal(self, query: str, max_results: int = 20, 
                                  site_category: str = None) -> List[Dict[str, Any]]:
        """Universal search for real links on all types of sites"""

        self.session_counter += 1
        session_id = f"universal_search_{self.session_counter}_{int(time.time())}"

        logger.info(f"🔍 Universal search for: '{query}' (Session: {session_id})")

        all_real_links = []

        try:
            # 1. Search via search engines
            search_results = self._search_via_engines(query)

            # 2. Extract and validate links
            for result in search_results:
                if result.success:
                    extracted_links = self._extract_all_links(result.content, result.url)

                    for link_info in extracted_links:
                        if self._is_real_valid_link(link_info['url'], query):
                            # Enrich link information
                            enriched_link = self._enrich_link_info(link_info, query)
                            if enriched_link:
                                all_real_links.append(enriched_link)

                                if len(all_real_links) >= max_results:
                                    break

                if len(all_real_links) >= max_results:
                    break

            # 3. Search on specialized sites if category specified
            if site_category and site_category in self.specialized_sites:
                specialized_links = self._search_specialized_sites(query, site_category)
                all_real_links.extend(specialized_links[:max_results//2])

            # 4. Save found links
            if all_real_links:
                self._save_real_links(all_real_links, session_id, query)

            logger.info(f"✅ {len(all_real_links)} real links found for '{query}'")

        except Exception as e:
            logger.error(f"Error during universal search: {str(e)}")

        return all_real_links[:max_results]

    def _search_via_engines(self, query: str) -> List[ScrapingResult]:
        """Search via multiple search engines"""
        results = []

        encoded_query = quote(query)

        for engine_name, engine_url in self.search_engines.items():
            try:
                search_url = engine_url.format(query=encoded_query)
                logger.info(f"Searching on {engine_name}: {search_url}")

                result = self._scrape_url(search_url)
                if result.success:
                    results.append(result)

                time.sleep(self.universal_config["default_delay"])

            except Exception as e:
                logger.error(f"Error with {engine_name}: {str(e)}")
                continue

        return results

    def _extract_all_links(self, html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """Extracts all links from an HTML page"""
        links = []

        # Patterns for different link types
        link_patterns = [
            r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]*)</a>',
            r'href=["\']([^"\']+)["\']',
            r'url\s*\(\s*["\']([^"\']+)["\']',
            r'src=["\']([^"\']+)["\']'
        ]

        for pattern in link_patterns:
            matches = re.finditer(pattern, html_content, re.IGNORECASE)

            for match in matches:
                if len(match.groups()) >= 2:
                    url = match.group(1)
                    text = match.group(2) if len(match.groups()) > 1 else ""
                else:
                    url = match.group(1)
                    text = ""

                # Build absolute URL
                absolute_url = urljoin(base_url, url)

                if self._is_valid_url_format(absolute_url):
                    links.append({
                        'url': absolute_url,
                        'text': text.strip(),
                        'source_page': base_url,
                        'found_at': datetime.now().isoformat()
                    })

        # Remove duplicates
        unique_links = []
        seen_urls = set()

        for link in links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])

        return unique_links[:100]  # Limit to 100 links per page

    def _is_valid_url_format(self, url: str) -> bool:
        """Checks if the URL has a valid format"""
        try:
            parsed = urlparse(url)

            # Basic checks
            if not parsed.scheme or not parsed.netloc:
                return False

            # Exclude non-web URLs
            invalid_schemes = ['javascript', 'mailto', 'tel', 'ftp', 'file']
            if parsed.scheme.lower() in invalid_schemes:
                return False

            # Exclude fragments and anchors
            if url.startswith('#'):
                return False

            return True

        except Exception:
            return False

    def _is_real_valid_link(self, url: str, query: str) -> bool:
        """Checks if the link is real and relevant"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            # Exclude search engines themselves
            search_domains = ['google.', 'bing.', 'duckduckgo.', 'yandex.']
            if any(search in domain for search in search_domains):
                return False

            # Exclude too short or suspicious URLs
            if len(url) < 20:
                return False

            # Include relevant domains
            query_lower = query.lower()

            # Check relevance in URL or domain
            if any(word in url.lower() for word in query_lower.split() if len(word) > 3):
                return True

            # Trusted domains
            trusted_domains = [
                '.fr', '.com', '.org', '.net', '.edu', '.gov',
                'wikipedia.', 'github.', 'stackoverflow.', 'reddit.'
            ]

            if any(trusted in domain for trusted in trusted_domains):
                return True

            return False

        except Exception:
            return False

    def _enrich_link_info(self, link_info: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Enriches link information by visiting the page"""
        try:
            url = link_info['url']

            # Avoid revisiting URLs
            if url in self.visited_urls:
                return None

            result = self._scrape_url(url)

            if result.success and len(result.content) > 200:
                self.visited_urls.add(url)

                # Extract additional information
                domain = urlparse(url).netloc
                content_preview = result.content[:500] + "..." if len(result.content) > 500 else result.content

                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(result.content, result.title, query)

                return {
                    'url': url,
                    'title': result.title or link_info.get('text', 'No title'), # Changed from 'Sans titre'
                    'domain': domain,
                    'content_preview': content_preview,
                    'relevance_score': relevance_score,
                    'content_length': len(result.content),
                    'links_count': len(result.links),
                    'source_page': link_info.get('source_page', ''),
                    'found_at': link_info.get('found_at', datetime.now().isoformat()),
                    'query': query,
                    'type': self._classify_link_type(url, result.content)
                }

            return None

        except Exception as e:
            logger.error(f"Error while enriching {link_info.get('url', 'unknown URL')}: {str(e)}") # Changed from 'URL inconnue'
            return None

    def _calculate_relevance_score(self, content: str, title: str, query: str) -> int:
        """Calculates a relevance score for a link"""
        score = 0
        content_lower = content.lower()
        title_lower = title.lower() if title else ""
        query_words = query.lower().split()

        # Query words in title (high score)
        for word in query_words:
            if len(word) > 2 and word in title_lower:
                score += 5

        # Query words in content
        for word in query_words:
            if len(word) > 2:
                count = content_lower.count(word)
                score += min(count, 3)  # Max 3 points per word

        # Content quality
        if len(content) > 1000:
            score += 2
        if len(title) > 10:
            score += 1

        return min(score, 20)  # Maximum score of 20

    def _classify_link_type(self, url: str, content: str) -> str:
        """Classifies the link type"""
        domain = urlparse(url).netloc.lower()
        content_lower = content.lower()

        # Classification by domain
        if any(term in domain for term in ['youtube.', 'vimeo.', 'dailymotion.']):
            return "video"
        elif any(term in domain for term in ['github.', 'gitlab.', 'bitbucket.']):
            return "code"
        elif any(term in domain for term in ['amazon.', 'ebay.', 'cdiscount.']):
            return "ecommerce"
        elif any(term in domain for term in ['wikipedia.', 'wikimedia.']):
            return "encyclopedia"
        elif any(term in domain for term in ['leboncoin.', 'seloger.', 'pap.']):
            return "classified_ads"

        # Classification by content
        if any(term in content_lower for term in ['cours', 'formation', 'tutorial', 'course', 'training']): # Added English terms
            return "educational"
        elif any(term in content_lower for term in ['actualité', 'news', 'article']): # Maintained French as it's a French code originally, but 'news' is good
            return "news"
        elif any(term in content_lower for term in ['prix', 'acheter', 'vendre', 'price', 'buy', 'sell']): # Added English terms
            return "commercial"

        return "general"

    def _search_specialized_sites(self, query: str, category: str) -> List[Dict[str, Any]]:
        """Search on specialized sites"""
        specialized_links = []

        if category not in self.specialized_sites:
            return specialized_links

        sites = self.specialized_sites[category]

        for site in sites[:3]:  # Limit to 3 sites per category
            try:
                # Build search URL for the site
                search_url = f"https://www.{site}/recherche?q={quote(query)}" # 'recherche' is French for 'search' but part of URL path, so kept as is. Could be site-specific

                result = self._scrape_url(search_url)

                if result.success:
                    site_links = self._extract_all_links(result.content, search_url)

                    for link_info in site_links[:10]:  # Max 10 links per site
                        if site in link_info['url']:
                            enriched_link = self._enrich_link_info(link_info, query)
                            if enriched_link:
                                specialized_links.append(enriched_link)

                time.sleep(self.universal_config["default_delay"])

            except Exception as e:
                logger.error(f"Error on {site}: {str(e)}")
                continue

        return specialized_links

    def _scrape_url(self, url: str) -> ScrapingResult:
        """Scrapes a specific URL"""
        start_time = time.time()

        try:
            response = self.session.get(
                url, 
                timeout=15,
                allow_redirects=True,
                verify=False
            )

            response.raise_for_status()

            # Analyze content
            content = response.text
            extracted_data = self._extract_content_from_html(content, url)

            return ScrapingResult(
                url=url,
                content=extracted_data['text'],
                title=extracted_data['title'],
                links=extracted_data['links'],
                images=extracted_data['images'],
                timestamp=datetime.now(),
                success=True,
                response_time=time.time() - start_time,
                status_code=response.status_code
            )

        except Exception as e:
            logger.error(f"Error while scraping {url}: {str(e)}")
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=str(e),
                response_time=time.time() - start_time
            )

    def _extract_content_from_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extracts content from an HTML page"""

        # Title extraction
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        # Remove scripts and styles
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Extract text by removing HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text).strip()

        # Extract links
        links = []
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
        for match in re.finditer(link_pattern, html_content, re.IGNORECASE):
            href = match.group(1)
            absolute_url = urljoin(url, href)
            if self._is_valid_url_format(absolute_url):
                links.append(absolute_url)

        # Extract images
        images = []
        img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        for match in re.finditer(img_pattern, html_content, re.IGNORECASE):
            src = match.group(1)
            absolute_url = urljoin(url, src)
            if self._is_valid_url_format(absolute_url):
                images.append(absolute_url)

        return {
            'text': text,
            'title': title,
            'links': list(set(links)),
            'images': list(set(images))
        }

    def _save_real_links(self, links: List[Dict[str, Any]], session_id: str, query: str):
        """Saves the real links found"""
        try:
            # Main file with all links
            links_file = self.real_links_dir / f"{session_id}_real_links.txt"

            with open(links_file, 'w', encoding='utf-8') as f:
                f.write(f"REAL LINKS FOUND - UNIVERSAL SEARCH\n") # Changed from French
                f.write(f"=" * 60 + "\n")
                f.write(f"Query: {query}\n") # Changed from 'Requête'
                f.write(f"Session: {session_id}\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
                f.write(f"Number of links: {len(links)}\n") # Changed from 'Nombre de liens'
                f.write(f"=" * 60 + "\n\n")

                for i, link in enumerate(links, 1):
                    f.write(f"LINK {i}/{len(links)}\n") # Changed from 'LIEN'
                    f.write(f"URL: {link['url']}\n")
                    f.write(f"Title: {link.get('title', 'No title')}\n") # Changed from 'Titre', 'Sans titre'
                    f.write(f"Domain: {link.get('domain', 'Unknown')}\n") # Changed from 'Domaine', 'Inconnu'
                    f.write(f"Type: {link.get('type', 'general')}\n")
                    f.write(f"Relevance score: {link.get('relevance_score', 0)}/20\n") # Changed from 'Score de pertinence'
                    f.write(f"Content length: {link.get('content_length', 0)} characters\n") # Changed from 'Longueur du contenu', 'caractères'
                    f.write(f"Found on: {link.get('found_at', 'Unknown date')}\n") # Changed from 'Trouvé le', 'Date inconnue'

                    if link.get('content_preview'):
                        f.write(f"Content preview:\n{link['content_preview']}\n") # Changed from 'Aperçu du contenu'

                    f.write("-" * 40 + "\n\n")

            # JSON file for automatic processing
            json_file = self.real_links_dir / f"{session_id}_real_links.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'links_count': len(links),
                    'links': links
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"Links saved to {links_file} and {json_file}") # Changed from 'Liens sauvegardés'

        except Exception as e:
            logger.error(f"Error during saving: {str(e)}") # Changed from 'Erreur lors de la sauvegarde'

# Global instance
universal_web_scraper = UniversalWebScraper()

# Public interface functions
def search_real_links_from_any_site(query: str, max_results: int = 20, 
                                   site_category: str = None) -> List[Dict[str, Any]]:
    """Public interface to search for real links on all types of sites"""
    return universal_web_scraper.search_real_links_universal(query, max_results, site_category)

def get_supported_site_categories() -> Dict[str, List[str]]:
    """Returns the supported site categories"""
    return universal_web_scraper.specialized_sites

if __name__ == "__main__":
    print("=== Universal Real Link Search System Test ===") # Changed from French

    # Test with different query types
    test_queries = [
        "apartment lille", # Changed from 'appartement lille'
        "python programming course", # Changed from 'cours python programmation'
        "artificial intelligence news", # Changed from 'actualités intelligence artificielle'
        "web developer job" # Changed from 'emploi développeur web'
    ]

    for query in test_queries:
        print(f"\n--- Test for: '{query}' ---") # Changed from 'Test pour'

        links = search_real_links_from_any_site(query, max_results=5)

        if links:
            print(f"✓ {len(links)} real links found:") # Changed from 'liens réels trouvés'
            for i, link in enumerate(links, 1):
                print(f"  {i}. {link['title']}")
                print(f"     URL: {link['url']}")
                print(f"     Type: {link.get('type', 'general')}")
                print(f"     Score: {link.get('relevance_score', 0)}/20")
                print()
        else:
            print("✗ No link found") # Changed from 'Aucun lien trouvé'

    print("\n=== Supported Site Categories ===") # Changed from French
    categories = get_supported_site_categories()
    for category, sites in categories.items():
        print(f"{category}: {', '.join(sites[:3])}...")
