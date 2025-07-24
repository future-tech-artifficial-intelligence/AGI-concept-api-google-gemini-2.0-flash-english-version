"""
Système de Web Scraping Autonome Universel pour l'Intelligence Artificielle
Ce module permet à l'IA d'accéder à Internet de manière autonome pour obtenir de vrais liens
depuis n'importe quel site web. Les données extraites sont sauvegardées dans des fichiers texte.
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

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Résultat d'une opération de scraping"""
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
    """Système de scraping web universel pour l'IA"""

    def __init__(self):
        self.session = requests.Session()

        # Répertoires pour sauvegarder les données
        self.data_dir = Path("data/autonomous_web_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.extracted_text_dir = self.data_dir / "extracted_text"
        self.extracted_text_dir.mkdir(parents=True, exist_ok=True)

        self.real_links_dir = self.data_dir / "real_links"
        self.real_links_dir.mkdir(parents=True, exist_ok=True)

        # Configuration des headers par défaut
        self.default_headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

        self.session.headers.update(self.default_headers)

        # Historique des URLs visitées
        self.visited_urls: Set[str] = set()

        # Configuration universelle
        self.universal_config = {
            "max_concurrent_requests": 20,
            "default_delay": 0.2,
            "max_pages_per_search": 50,
            "auto_save_interval": 5
        }

        # Moteurs de recherche et sites supportés
        self.search_engines = {
            "duckduckgo": "https://duckduckgo.com/?q={query}",
            "bing": "https://www.bing.com/search?q={query}",
            "yandex": "https://yandex.com/search/?text={query}"
        }

        # Sites spécialisés pour différents types de recherche
        self.specialized_sites = {
            "immobilier": [
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
            "actualites": [
                "lemonde.fr",
                "lefigaro.fr",
                "liberation.fr",
                "franceinfo.fr",
                "20minutes.fr"
            ],
            "emploi": [
                "pole-emploi.fr",
                "indeed.fr",
                "monster.fr",
                "apec.fr",
                "linkedin.com"
            ],
            "formation": [
                "coursera.org",
                "udemy.com",
                "openclassrooms.com",
                "fun-mooc.fr",
                "edx.org"
            ]
        }

        # Compteur de sessions
        self.session_counter = 0

        logger.info("Système de Web Scraping Universel initialisé")

    def search_real_links_universal(self, query: str, max_results: int = 20, 
                                  site_category: str = None) -> List[Dict[str, Any]]:
        """Recherche universelle de vrais liens sur tous types de sites"""

        self.session_counter += 1
        session_id = f"universal_search_{self.session_counter}_{int(time.time())}"

        logger.info(f"🔍 Recherche universelle pour: '{query}' (Session: {session_id})")

        all_real_links = []

        try:
            # 1. Recherche via moteurs de recherche
            search_results = self._search_via_engines(query)

            # 2. Extraction et validation des liens
            for result in search_results:
                if result.success:
                    extracted_links = self._extract_all_links(result.content, result.url)

                    for link_info in extracted_links:
                        if self._is_real_valid_link(link_info['url'], query):
                            # Enrichir les informations du lien
                            enriched_link = self._enrich_link_info(link_info, query)
                            if enriched_link:
                                all_real_links.append(enriched_link)

                                if len(all_real_links) >= max_results:
                                    break

                if len(all_real_links) >= max_results:
                    break

            # 3. Recherche sur sites spécialisés si catégorie spécifiée
            if site_category and site_category in self.specialized_sites:
                specialized_links = self._search_specialized_sites(query, site_category)
                all_real_links.extend(specialized_links[:max_results//2])

            # 4. Sauvegarder les liens trouvés
            if all_real_links:
                self._save_real_links(all_real_links, session_id, query)

            logger.info(f"✅ {len(all_real_links)} liens réels trouvés pour '{query}'")

        except Exception as e:
            logger.error(f"Erreur lors de la recherche universelle: {str(e)}")

        return all_real_links[:max_results]

    def _search_via_engines(self, query: str) -> List[ScrapingResult]:
        """Recherche via plusieurs moteurs de recherche"""
        results = []

        encoded_query = quote(query)

        for engine_name, engine_url in self.search_engines.items():
            try:
                search_url = engine_url.format(query=encoded_query)
                logger.info(f"Recherche sur {engine_name}: {search_url}")

                result = self._scrape_url(search_url)
                if result.success:
                    results.append(result)

                time.sleep(self.universal_config["default_delay"])

            except Exception as e:
                logger.error(f"Erreur avec {engine_name}: {str(e)}")
                continue

        return results

    def _extract_all_links(self, html_content: str, base_url: str) -> List[Dict[str, Any]]:
        """Extrait tous les liens d'une page HTML"""
        links = []

        # Patterns pour différents types de liens
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

                # Construire l'URL absolue
                absolute_url = urljoin(base_url, url)

                if self._is_valid_url_format(absolute_url):
                    links.append({
                        'url': absolute_url,
                        'text': text.strip(),
                        'source_page': base_url,
                        'found_at': datetime.now().isoformat()
                    })

        # Supprimer les doublons
        unique_links = []
        seen_urls = set()

        for link in links:
            if link['url'] not in seen_urls:
                unique_links.append(link)
                seen_urls.add(link['url'])

        return unique_links[:100]  # Limiter à 100 liens par page

    def _is_valid_url_format(self, url: str) -> bool:
        """Vérifie si l'URL a un format valide"""
        try:
            parsed = urlparse(url)

            # Vérifications de base
            if not parsed.scheme or not parsed.netloc:
                return False

            # Exclure les URLs non web
            invalid_schemes = ['javascript', 'mailto', 'tel', 'ftp', 'file']
            if parsed.scheme.lower() in invalid_schemes:
                return False

            # Exclure les fragments et ancres
            if url.startswith('#'):
                return False

            return True

        except Exception:
            return False

    def _is_real_valid_link(self, url: str, query: str) -> bool:
        """Vérifie si le lien est réel et pertinent"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()

            # Exclure les moteurs de recherche eux-mêmes
            search_domains = ['google.', 'bing.', 'duckduckgo.', 'yandex.']
            if any(search in domain for search in search_domains):
                return False

            # Exclure les URLs trop courtes ou suspectes
            if len(url) < 20:
                return False

            # Inclure les domaines pertinents
            query_lower = query.lower()

            # Vérifier la pertinence dans l'URL ou le domaine
            if any(word in url.lower() for word in query_lower.split() if len(word) > 3):
                return True

            # Domaines de confiance
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
        """Enrichit les informations d'un lien en visitant la page"""
        try:
            url = link_info['url']

            # Eviter de revisiter les URLs
            if url in self.visited_urls:
                return None

            result = self._scrape_url(url)

            if result.success and len(result.content) > 200:
                self.visited_urls.add(url)

                # Extraire des informations supplémentaires
                domain = urlparse(url).netloc
                content_preview = result.content[:500] + "..." if len(result.content) > 500 else result.content

                # Calculer un score de pertinence
                relevance_score = self._calculate_relevance_score(result.content, result.title, query)

                return {
                    'url': url,
                    'title': result.title or link_info.get('text', 'Sans titre'),
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
            logger.error(f"Erreur lors de l'enrichissement de {link_info.get('url', 'URL inconnue')}: {str(e)}")
            return None

    def _calculate_relevance_score(self, content: str, title: str, query: str) -> int:
        """Calcule un score de pertinence pour un lien"""
        score = 0
        content_lower = content.lower()
        title_lower = title.lower() if title else ""
        query_words = query.lower().split()

        # Mots de la requête dans le titre (score élevé)
        for word in query_words:
            if len(word) > 2 and word in title_lower:
                score += 5

        # Mots de la requête dans le contenu
        for word in query_words:
            if len(word) > 2:
                count = content_lower.count(word)
                score += min(count, 3)  # Max 3 points par mot

        # Qualité du contenu
        if len(content) > 1000:
            score += 2
        if len(title) > 10:
            score += 1

        return min(score, 20)  # Score maximum de 20

    def _classify_link_type(self, url: str, content: str) -> str:
        """Classifie le type de lien"""
        domain = urlparse(url).netloc.lower()
        content_lower = content.lower()

        # Classification par domaine
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

        # Classification par contenu
        if any(term in content_lower for term in ['cours', 'formation', 'tutorial']):
            return "educational"
        elif any(term in content_lower for term in ['actualité', 'news', 'article']):
            return "news"
        elif any(term in content_lower for term in ['prix', 'acheter', 'vendre']):
            return "commercial"

        return "general"

    def _search_specialized_sites(self, query: str, category: str) -> List[Dict[str, Any]]:
        """Recherche sur des sites spécialisés"""
        specialized_links = []

        if category not in self.specialized_sites:
            return specialized_links

        sites = self.specialized_sites[category]

        for site in sites[:3]:  # Limiter à 3 sites par catégorie
            try:
                # Construire l'URL de recherche pour le site
                search_url = f"https://www.{site}/recherche?q={quote(query)}"

                result = self._scrape_url(search_url)

                if result.success:
                    site_links = self._extract_all_links(result.content, search_url)

                    for link_info in site_links[:10]:  # 10 liens max par site
                        if site in link_info['url']:
                            enriched_link = self._enrich_link_info(link_info, query)
                            if enriched_link:
                                specialized_links.append(enriched_link)

                time.sleep(self.universal_config["default_delay"])

            except Exception as e:
                logger.error(f"Erreur sur {site}: {str(e)}")
                continue

        return specialized_links

    def _scrape_url(self, url: str) -> ScrapingResult:
        """Scrape une URL spécifique"""
        start_time = time.time()

        try:
            response = self.session.get(
                url, 
                timeout=15,
                allow_redirects=True,
                verify=False
            )

            response.raise_for_status()

            # Analyser le contenu
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
            logger.error(f"Erreur lors du scraping de {url}: {str(e)}")
            return ScrapingResult(
                url=url,
                content="",
                success=False,
                error_message=str(e),
                response_time=time.time() - start_time
            )

    def _extract_content_from_html(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extrait le contenu d'une page HTML"""

        # Extraction du titre
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        title = title_match.group(1).strip() if title_match else ""

        # Supprimer les scripts et styles
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)

        # Extraire le texte en supprimant les balises HTML
        text = re.sub(r'<[^>]+>', ' ', html_content)
        text = re.sub(r'\s+', ' ', text).strip()

        # Extraire les liens
        links = []
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
        for match in re.finditer(link_pattern, html_content, re.IGNORECASE):
            href = match.group(1)
            absolute_url = urljoin(url, href)
            if self._is_valid_url_format(absolute_url):
                links.append(absolute_url)

        # Extraire les images
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
        """Sauvegarde les liens réels trouvés"""
        try:
            # Fichier principal avec tous les liens
            links_file = self.real_links_dir / f"{session_id}_real_links.txt"

            with open(links_file, 'w', encoding='utf-8') as f:
                f.write(f"LIENS RÉELS TROUVÉS - RECHERCHE UNIVERSELLE\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"Requête: {query}\n")
                f.write(f"Session: {session_id}\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
                f.write(f"Nombre de liens: {len(links)}\n")
                f.write(f"=" * 60 + "\n\n")

                for i, link in enumerate(links, 1):
                    f.write(f"LIEN {i}/{len(links)}\n")
                    f.write(f"URL: {link['url']}\n")
                    f.write(f"Titre: {link.get('title', 'Sans titre')}\n")
                    f.write(f"Domaine: {link.get('domain', 'Inconnu')}\n")
                    f.write(f"Type: {link.get('type', 'general')}\n")
                    f.write(f"Score de pertinence: {link.get('relevance_score', 0)}/20\n")
                    f.write(f"Longueur du contenu: {link.get('content_length', 0)} caractères\n")
                    f.write(f"Trouvé le: {link.get('found_at', 'Date inconnue')}\n")

                    if link.get('content_preview'):
                        f.write(f"Aperçu du contenu:\n{link['content_preview']}\n")

                    f.write("-" * 40 + "\n\n")

            # Fichier JSON pour traitement automatique
            json_file = self.real_links_dir / f"{session_id}_real_links.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': query,
                    'session_id': session_id,
                    'timestamp': datetime.now().isoformat(),
                    'links_count': len(links),
                    'links': links
                }, f, ensure_ascii=False, indent=2)

            logger.info(f"Liens sauvegardés dans {links_file} et {json_file}")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")

# Instance globale
universal_web_scraper = UniversalWebScraper()

# Fonctions d'interface publique
def search_real_links_from_any_site(query: str, max_results: int = 20, 
                                   site_category: str = None) -> List[Dict[str, Any]]:
    """Interface publique pour rechercher des liens réels sur tous types de sites"""
    return universal_web_scraper.search_real_links_universal(query, max_results, site_category)

def get_supported_site_categories() -> Dict[str, List[str]]:
    """Retourne les catégories de sites supportées"""
    return universal_web_scraper.specialized_sites

if __name__ == "__main__":
    print("=== Test du Système de Recherche Universelle de Liens Réels ===")

    # Test avec différents types de requêtes
    test_queries = [
        "appartement lille",
        "cours python programmation",
        "actualités intelligence artificielle",
        "emploi développeur web"
    ]

    for query in test_queries:
        print(f"\n--- Test pour: '{query}' ---")

        links = search_real_links_from_any_site(query, max_results=5)

        if links:
            print(f"✓ {len(links)} liens réels trouvés:")
            for i, link in enumerate(links, 1):
                print(f"  {i}. {link['title']}")
                print(f"     URL: {link['url']}")
                print(f"     Type: {link.get('type', 'general')}")
                print(f"     Score: {link.get('relevance_score', 0)}/20")
                print()
        else:
            print("✗ Aucun lien trouvé")

    print("\n=== Catégories de sites supportées ===")
    categories = get_supported_site_categories()
    for category, sites in categories.items():
        print(f"{category}: {', '.join(sites[:3])}...")