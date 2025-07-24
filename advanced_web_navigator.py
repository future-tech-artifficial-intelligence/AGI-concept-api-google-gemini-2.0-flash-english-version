"""
Système de Navigation Web Avancé pour l'API Gemini
Ce module permet à l'API Gemini de naviguer à l'intérieur des sites web
et d'extraire du contenu détaillé, pas seulement les liens.
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
from bs4 import BeautifulSoup, Comment
import nltk
from collections import defaultdict

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AdvancedWebNavigator')

@dataclass
class WebPageContent:
    """Contenu structuré d'une page web"""
    url: str
    title: str
    content: str
    cleaned_text: str
    summary: str
    main_content: str
    metadata: Dict[str, Any]
    links: List[Dict[str, str]]
    images: List[Dict[str, str]]
    navigation_elements: List[Dict[str, str]]
    content_sections: List[Dict[str, str]]
    keywords: List[str]
    language: str
    content_quality_score: float
    extraction_timestamp: datetime
    success: bool = True
    error_message: str = ""

@dataclass
class NavigationPath:
    """Chemin de navigation dans un site"""
    start_url: str
    visited_pages: List[WebPageContent]
    navigation_depth: int
    total_content_extracted: int
    navigation_strategy: str
    session_id: str
    created_at: datetime

class AdvancedContentExtractor:
    """Extracteur de contenu web avancé"""
    
    def __init__(self):
        self.content_selectors = {
            'main_content': [
                'main', 'article', '[role="main"]', '.content', '.main-content',
                '.article-content', '.post-content', '.entry-content', 
                '#content', '#main-content', '.page-content'
            ],
            'navigation': [
                'nav', '.nav', '.navigation', '.menu', '.main-nav',
                '[role="navigation"]', '.navbar', '.site-nav'
            ],
            'sidebar': [
                'aside', '.sidebar', '.side-nav', '.secondary'
            ],
            'footer': [
                'footer', '.footer', '[role="contentinfo"]'
            ]
        }
        
        self.noise_selectors = [
            'script', 'style', 'noscript', 'iframe', 'embed', 'object',
            '.advertisement', '.ads', '.sponsor', '.popup', '.modal',
            '[class*="ad-"]', '[id*="ad-"]', '.cookie-banner'
        ]
    
    def extract_page_content(self, html: str, url: str) -> WebPageContent:
        """Extrait le contenu structuré d'une page web"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Supprimer le bruit
            for selector in self.noise_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extraire les métadonnées
            metadata = self._extract_metadata(soup)
            
            # Extraire le titre
            title = self._extract_title(soup)
            
            # Extraire le contenu principal
            main_content = self._extract_main_content(soup)
            
            # Extraire tout le texte
            all_text = soup.get_text(separator=' ', strip=True)
            cleaned_text = self._clean_text(all_text)
            
            # Créer un résumé
            summary = self._create_summary(cleaned_text)
            
            # Extraire les liens
            links = self._extract_links(soup, url)
            
            # Extraire les images
            images = self._extract_images(soup, url)
            
            # Extraire les éléments de navigation
            nav_elements = self._extract_navigation_elements(soup, url)
            
            # Extraire les sections de contenu
            content_sections = self._extract_content_sections(soup)
            
            # Extraire les mots-clés
            keywords = self._extract_keywords(cleaned_text)
            
            # Détecter la langue
            language = self._detect_language(cleaned_text)
            
            # Calculer le score de qualité
            quality_score = self._calculate_content_quality(cleaned_text, title, links)
            
            return WebPageContent(
                url=url,
                title=title,
                content=all_text,
                cleaned_text=cleaned_text,
                summary=summary,
                main_content=main_content,
                metadata=metadata,
                links=links,
                images=images,
                navigation_elements=nav_elements,
                content_sections=content_sections,
                keywords=keywords,
                language=language,
                content_quality_score=quality_score,
                extraction_timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du contenu de {url}: {str(e)}")
            return WebPageContent(
                url=url,
                title="",
                content="",
                cleaned_text="",
                summary="",
                main_content="",
                metadata={},
                links=[],
                images=[],
                navigation_elements=[],
                content_sections=[],
                keywords=[],
                language="",
                content_quality_score=0.0,
                extraction_timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extrait les métadonnées de la page"""
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property') or meta.get('http-equiv')
            content = meta.get('content')
            if name and content:
                metadata[name] = content
        
        # Schema.org data
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                schema_data = json.loads(script.string)
                metadata['schema_org'] = schema_data
            except:
                pass
        
        return metadata
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extrait le titre de la page"""
        # Essayer plusieurs sources pour le titre
        title_sources = [
            lambda: soup.find('title').get_text().strip() if soup.find('title') else '',
            lambda: soup.find('h1').get_text().strip() if soup.find('h1') else '',
            lambda: soup.find('meta', property='og:title')['content'] if soup.find('meta', property='og:title') else '',
            lambda: soup.find('meta', name='twitter:title')['content'] if soup.find('meta', name='twitter:title') else ''
        ]
        
        for source in title_sources:
            try:
                title = source()
                if title:
                    return title
            except:
                continue
        
        return "Page sans titre"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extrait le contenu principal de la page"""
        # Essayer les sélecteurs de contenu principal
        for selector in self.content_selectors['main_content']:
            elements = soup.select(selector)
            if elements:
                # Prendre le plus grand élément
                main_element = max(elements, key=lambda x: len(x.get_text()))
                return main_element.get_text(separator=' ', strip=True)
        
        # Fallback: supprimer navigation, sidebar, footer et prendre le reste
        content_soup = BeautifulSoup(str(soup), 'html.parser')
        
        for selector_type in ['navigation', 'sidebar', 'footer']:
            for selector in self.content_selectors[selector_type]:
                for element in content_soup.select(selector):
                    element.decompose()
        
        return content_soup.get_text(separator=' ', strip=True)
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte extrait"""
        if not text:
            return ""
        
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Supprimer les caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Supprimer les lignes vides multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _create_summary(self, text: str, max_sentences: int = 3) -> str:
        """Crée un résumé du contenu"""
        if not text or len(text) < 100:
            return text
        
        # Diviser en phrases
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return text[:500] + "..." if len(text) > 500 else text
        
        # Prendre les premières phrases qui contiennent des informations importantes
        important_sentences = []
        for sentence in sentences[:10]:  # Examiner les 10 premières phrases
            if any(keyword in sentence.lower() for keyword in 
                   ['important', 'principal', 'essentiel', 'clé', 'majeur', 'définition']):
                important_sentences.append(sentence)
        
        # Si pas assez de phrases importantes, prendre les premières
        if len(important_sentences) < max_sentences:
            summary_sentences = sentences[:max_sentences]
        else:
            summary_sentences = important_sentences[:max_sentences]
        
        return '. '.join(summary_sentences) + '.'
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extrait tous les liens de la page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Nettoyer l'URL
            parsed = urlparse(full_url)
            if parsed.scheme in ['http', 'https']:
                links.append({
                    'url': full_url,
                    'text': link.get_text().strip(),
                    'title': link.get('title', ''),
                    'rel': link.get('rel', []),
                    'target': link.get('target', '')
                })
        
        return links
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extrait toutes les images de la page"""
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            full_url = urljoin(base_url, src)
            
            images.append({
                'url': full_url,
                'alt': img.get('alt', ''),
                'title': img.get('title', ''),
                'width': img.get('width', ''),
                'height': img.get('height', '')
            })
        
        return images
    
    def _extract_navigation_elements(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extrait les éléments de navigation"""
        nav_elements = []
        
        for selector in self.content_selectors['navigation']:
            for nav in soup.select(selector):
                for link in nav.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(base_url, href)
                    
                    nav_elements.append({
                        'url': full_url,
                        'text': link.get_text().strip(),
                        'type': 'navigation'
                    })
        
        return nav_elements
    
    def _extract_content_sections(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extrait les sections de contenu"""
        sections = []
        
        # Sections avec headers
        for i in range(1, 7):  # h1 à h6
            for header in soup.find_all(f'h{i}'):
                section_content = ""
                
                # Trouver le contenu après le header
                next_element = header.next_sibling
                while next_element and next_element.name not in [f'h{j}' for j in range(1, i+1)]:
                    if hasattr(next_element, 'get_text'):
                        section_content += next_element.get_text(separator=' ', strip=True) + " "
                    next_element = next_element.next_sibling
                
                if section_content.strip():
                    sections.append({
                        'title': header.get_text().strip(),
                        'content': section_content.strip(),
                        'level': f'h{i}'
                    })
        
        return sections
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extrait les mots-clés du texte"""
        if not text:
            return []
        
        # Mots vides en français et anglais
        stop_words = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'en', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'the', 'be',
            'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on',
            'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from'
        }
        
        # Extraire les mots
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
        
        # Compter les fréquences
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
        
        # Retourner les mots les plus fréquents
        return [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]]
    
    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte"""
        if not text:
            return "unknown"
        
        # Mots indicateurs français
        french_indicators = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'est', 'sont', 'avec', 'dans', 'pour', 'sur', 'par']
        english_indicators = ['the', 'and', 'or', 'is', 'are', 'with', 'in', 'for', 'on', 'by', 'at', 'to', 'of']
        
        text_lower = text.lower()
        french_count = sum(1 for word in french_indicators if f' {word} ' in text_lower)
        english_count = sum(1 for word in english_indicators if f' {word} ' in text_lower)
        
        if french_count > english_count:
            return "fr"
        elif english_count > french_count:
            return "en"
        else:
            return "unknown"
    
    def _calculate_content_quality(self, text: str, title: str, links: List[Dict]) -> float:
        """Calcule un score de qualité du contenu"""
        score = 0.0
        
        # Longueur du contenu
        if len(text) > 1000:
            score += 3.0
        elif len(text) > 500:
            score += 2.0
        elif len(text) > 100:
            score += 1.0
        
        # Qualité du titre
        if title and len(title) > 10:
            score += 1.0
        
        # Nombre de liens
        if len(links) > 10:
            score += 2.0
        elif len(links) > 5:
            score += 1.0
        
        # Ratio texte/liens (éviter le spam)
        if len(links) > 0:
            text_link_ratio = len(text) / len(links)
            if text_link_ratio > 100:
                score += 1.0
        
        # Présence de structure (sections, headers)
        if re.search(r'\n\s*\n', text):  # Paragraphes
            score += 1.0
        
        return min(score, 10.0)

class AdvancedWebNavigator:
    """Navigateur web avancé pour l'API Gemini"""
    
    def __init__(self):
        self.session = requests.Session()
        self.content_extractor = AdvancedContentExtractor()
        
        # Configuration des headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Répertoires de stockage
        self.data_dir = Path("data/advanced_web_navigation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.content_cache_dir = self.data_dir / "content_cache"
        self.content_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.navigation_logs_dir = self.data_dir / "navigation_logs"
        self.navigation_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache en mémoire
        self.content_cache: Dict[str, WebPageContent] = {}
        self.url_blacklist: Set[str] = set()
        
        logger.info("✅ Navigateur web avancé initialisé")
    
    def navigate_deep(self, start_url: str, 
                     max_depth: int = 3, 
                     max_pages: int = 10,
                     navigation_strategy: str = 'breadth_first',
                     content_filter: Optional[callable] = None) -> NavigationPath:
        """
        Navigation en profondeur dans un site web
        
        Args:
            start_url: URL de départ
            max_depth: Profondeur maximale de navigation
            max_pages: Nombre maximum de pages à visiter
            navigation_strategy: 'breadth_first', 'depth_first', 'quality_first'
            content_filter: Fonction de filtrage du contenu
        """
        session_id = f"nav_{int(time.time())}"
        logger.info(f"🚀 Début de navigation profonde: {start_url} (session: {session_id})")
        
        navigation_path = NavigationPath(
            start_url=start_url,
            visited_pages=[],
            navigation_depth=0,
            total_content_extracted=0,
            navigation_strategy=navigation_strategy,
            session_id=session_id,
            created_at=datetime.now()
        )
        
        # Queue de navigation avec priorité
        navigation_queue = [(start_url, 0)]  # (url, depth)
        visited_urls = set()
        
        while navigation_queue and len(navigation_path.visited_pages) < max_pages:
            # Sélectionner la prochaine URL selon la stratégie
            if navigation_strategy == 'quality_first':
                # Trier par qualité potentielle (à implémenter)
                navigation_queue.sort(key=lambda x: self._estimate_url_quality(x[0]), reverse=True)
            
            current_url, current_depth = navigation_queue.pop(0)
            
            if current_url in visited_urls or current_depth > max_depth:
                continue
            
            logger.info(f"📄 Navigation vers: {current_url} (profondeur: {current_depth})")
            
            # Extraire le contenu de la page
            page_content = self.extract_page_content(current_url)
            
            if page_content.success:
                visited_urls.add(current_url)
                
                # Appliquer le filtre de contenu si fourni
                if content_filter is None or content_filter(page_content):
                    navigation_path.visited_pages.append(page_content)
                    navigation_path.total_content_extracted += len(page_content.cleaned_text)
                    navigation_path.navigation_depth = max(navigation_path.navigation_depth, current_depth)
                    
                    # Sauvegarder le contenu
                    self._save_page_content(page_content, session_id)
                    
                    logger.info(f"✅ Contenu extrait: {len(page_content.cleaned_text)} caractères")
                    
                    # Ajouter les liens intéressants à la queue
                    if current_depth < max_depth:
                        interesting_links = self._select_navigation_links(page_content, visited_urls)
                        for link_url in interesting_links[:5]:  # Max 5 liens par page
                            navigation_queue.append((link_url, current_depth + 1))
                else:
                    logger.info("❌ Contenu filtré, page ignorée")
            else:
                logger.warning(f"⚠️ Échec de l'extraction: {page_content.error_message}")
            
            # Délai entre les requêtes
            time.sleep(0.5)
        
        # Sauvegarder le chemin de navigation
        self._save_navigation_path(navigation_path)
        
        logger.info(f"🏁 Navigation terminée: {len(navigation_path.visited_pages)} pages visitées")
        return navigation_path
    
    def extract_page_content(self, url: str) -> WebPageContent:
        """Extrait le contenu détaillé d'une page web"""
        # Vérifier le cache
        if url in self.content_cache:
            logger.info(f"📋 Contenu récupéré du cache: {url}")
            return self.content_cache[url]
        
        try:
            logger.info(f"🔍 Extraction du contenu: {url}")
            
            response = self.session.get(url, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            # Extraire le contenu
            page_content = self.content_extractor.extract_page_content(response.text, url)
            
            # Mettre en cache
            self.content_cache[url] = page_content
            
            return page_content
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction de {url}: {str(e)}")
            return WebPageContent(
                url=url,
                title="",
                content="",
                cleaned_text="",
                summary="",
                main_content="",
                metadata={},
                links=[],
                images=[],
                navigation_elements=[],
                content_sections=[],
                keywords=[],
                language="",
                content_quality_score=0.0,
                extraction_timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
    
    def search_and_navigate(self, query: str, search_engine: str = "duckduckgo") -> List[NavigationPath]:
        """Recherche et navigate dans les résultats"""
        logger.info(f"🔎 Recherche et navigation pour: {query}")
        
        # TODO: Intégrer avec Searx pour la recherche
        # Pour l'instant, on simule des résultats
        search_results = self._simulate_search_results(query)
        
        navigation_paths = []
        for result_url in search_results[:3]:  # Naviguer dans les 3 premiers résultats
            try:
                nav_path = self.navigate_deep(
                    start_url=result_url,
                    max_depth=2,
                    max_pages=5,
                    navigation_strategy='quality_first'
                )
                navigation_paths.append(nav_path)
            except Exception as e:
                logger.error(f"Erreur lors de la navigation de {result_url}: {str(e)}")
        
        return navigation_paths
    
    def _select_navigation_links(self, page_content: WebPageContent, visited_urls: Set[str]) -> List[str]:
        """Sélectionne les liens les plus intéressants pour la navigation"""
        interesting_links = []
        
        base_domain = urlparse(page_content.url).netloc
        
        for link in page_content.links:
            link_url = link['url']
            link_text = link['text'].lower()
            
            # Ignorer les liens déjà visités
            if link_url in visited_urls:
                continue
            
            # Privilégier les liens du même domaine
            if urlparse(link_url).netloc != base_domain:
                continue
            
            # Ignorer certains types de liens
            if any(ignored in link_url.lower() for ignored in 
                   ['.pdf', '.jpg', '.png', '.gif', 'mailto:', 'javascript:', '#']):
                continue
            
            # Score du lien basé sur le texte
            link_score = 0
            
            # Mots-clés intéressants dans le texte du lien
            interesting_keywords = [
                'détail', 'plus', 'voir', 'lire', 'article', 'page', 'section',
                'chapitre', 'guide', 'tutoriel', 'formation', 'cours', 'about',
                'contact', 'service', 'produit', 'information'
            ]
            
            for keyword in interesting_keywords:
                if keyword in link_text:
                    link_score += 1
            
            # Éviter les liens de navigation générique
            generic_links = ['accueil', 'home', 'menu', 'recherche', 'search']
            if any(generic in link_text for generic in generic_links):
                link_score -= 2
            
            if link_score > 0:
                interesting_links.append(link_url)
        
        # Trier par score potentiel
        return interesting_links[:10]
    
    def _estimate_url_quality(self, url: str) -> float:
        """Estime la qualité potentielle d'une URL"""
        score = 0.0
        
        # Longueur de l'URL (ni trop courte ni trop longue)
        url_length = len(url)
        if 20 < url_length < 100:
            score += 1.0
        
        # Présence de mots-clés dans l'URL
        quality_keywords = ['article', 'guide', 'tutorial', 'about', 'detail', 'info']
        for keyword in quality_keywords:
            if keyword in url.lower():
                score += 0.5
        
        # Éviter les URLs de spam
        spam_indicators = ['ad', 'ads', 'popup', 'redirect', 'track']
        for indicator in spam_indicators:
            if indicator in url.lower():
                score -= 1.0
        
        return max(score, 0.0)
    
    def _simulate_search_results(self, query: str) -> List[str]:
        """Simule des résultats de recherche (à remplacer par Searx)"""
        # Simulation simple - à remplacer par l'intégration Searx
        return [
            "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
            "https://www.futura-sciences.com/tech/definitions/informatique-intelligence-artificielle-555/",
            "https://www.lebigdata.fr/intelligence-artificielle-definition"
        ]
    
    def _save_page_content(self, page_content: WebPageContent, session_id: str):
        """Sauvegarde le contenu d'une page"""
        try:
            # Créer un nom de fichier basé sur l'URL
            url_hash = hashlib.md5(page_content.url.encode()).hexdigest()[:12]
            filename = f"{session_id}_{url_hash}.json"
            
            filepath = self.content_cache_dir / filename
            
            # Sérialiser le contenu
            content_data = {
                'url': page_content.url,
                'title': page_content.title,
                'content': page_content.content,
                'cleaned_text': page_content.cleaned_text,
                'summary': page_content.summary,
                'main_content': page_content.main_content,
                'metadata': page_content.metadata,
                'links': page_content.links,
                'images': page_content.images,
                'navigation_elements': page_content.navigation_elements,
                'content_sections': page_content.content_sections,
                'keywords': page_content.keywords,
                'language': page_content.language,
                'content_quality_score': page_content.content_quality_score,
                'extraction_timestamp': page_content.extraction_timestamp.isoformat(),
                'success': page_content.success,
                'error_message': page_content.error_message
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Contenu sauvegardé: {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
    
    def _save_navigation_path(self, navigation_path: NavigationPath):
        """Sauvegarde le chemin de navigation"""
        try:
            filename = f"navigation_{navigation_path.session_id}.json"
            filepath = self.navigation_logs_dir / filename
            
            path_data = {
                'start_url': navigation_path.start_url,
                'navigation_depth': navigation_path.navigation_depth,
                'total_content_extracted': navigation_path.total_content_extracted,
                'navigation_strategy': navigation_path.navigation_strategy,
                'session_id': navigation_path.session_id,
                'created_at': navigation_path.created_at.isoformat(),
                'visited_pages_count': len(navigation_path.visited_pages),
                'visited_urls': [page.url for page in navigation_path.visited_pages]
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(path_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"🗺️ Chemin de navigation sauvegardé: {filepath}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du chemin: {str(e)}")

# Instance globale
advanced_navigator = AdvancedWebNavigator()

def navigate_website_deep(url: str, max_depth: int = 3, max_pages: int = 10) -> NavigationPath:
    """Interface publique pour la navigation profonde"""
    return advanced_navigator.navigate_deep(url, max_depth, max_pages)

def extract_website_content(url: str) -> WebPageContent:
    """Interface publique pour l'extraction de contenu"""
    return advanced_navigator.extract_page_content(url)

def search_and_navigate_websites(query: str) -> List[NavigationPath]:
    """Interface publique pour recherche et navigation"""
    return advanced_navigator.search_and_navigate(query)

if __name__ == "__main__":
    print("=== Test du Système de Navigation Web Avancé ===")
    
    # Test d'extraction de contenu
    test_url = "https://fr.wikipedia.org/wiki/Intelligence_artificielle"
    print(f"Test d'extraction: {test_url}")
    
    content = extract_website_content(test_url)
    if content.success:
        print(f"✅ Titre: {content.title}")
        print(f"✅ Contenu: {len(content.cleaned_text)} caractères")
        print(f"✅ Résumé: {content.summary[:200]}...")
        print(f"✅ Mots-clés: {content.keywords[:5]}")
        print(f"✅ Score de qualité: {content.content_quality_score}")
    else:
        print(f"❌ Erreur: {content.error_message}")
