"""
**Advanced Web Navigation System for artificial intelligence API GOOGLE GEMINI 2.0 FLASH**
This module allows artificial intelligence API GOOGLE GEMINI 2.0 FLASH to navigate within websites and extract detailed content, not just links.
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

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AdvancedWebNavigator')

@dataclass
class WebPageContent:
    """Structured content of a web page"""
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
    """Navigation path within a site"""
    start_url: str
    visited_pages: List[WebPageContent]
    navigation_depth: int
    total_content_extracted: int
    navigation_strategy: str
    session_id: str
    created_at: datetime

class AdvancedContentExtractor:
    """Advanced web content extractor"""
    
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
        """Extracts structured content from a web page"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove noise
            for selector in self.noise_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Extract all text
            all_text = soup.get_text(separator=' ', strip=True)
            cleaned_text = self._clean_text(all_text)
            
            # Create a summary
            summary = self._create_summary(cleaned_text)
            
            # Extract links
            links = self._extract_links(soup, url)
            
            # Extract images
            images = self._extract_images(soup, url)
            
            # Extract navigation elements
            nav_elements = self._extract_navigation_elements(soup, url)
            
            # Extract content sections
            content_sections = self._extract_content_sections(soup)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_text)
            
            # Detect language
            language = self._detect_language(cleaned_text)
            
            # Calculate quality score
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
            logger.error(f"Error during content extraction from {url}: {str(e)}")
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
        """Extracts page metadata"""
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
        """Extracts page title"""
        # Try multiple sources for the title
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
        
        return "Untitled page"
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extracts the main content of the page"""
        # Try main content selectors
        for selector in self.content_selectors['main_content']:
            elements = soup.select(selector)
            if elements:
                # Take the largest element
                main_element = max(elements, key=lambda x: len(x.get_text()))
                return main_element.get_text(separator=' ', strip=True)
        
        # Fallback: remove navigation, sidebar, footer and take the rest
        content_soup = BeautifulSoup(str(soup), 'html.parser')
        
        for selector_type in ['navigation', 'sidebar', 'footer']:
            for selector in self.content_selectors[selector_type]:
                for element in content_soup.select(selector):
                    element.decompose()
        
        return content_soup.get_text(separator=' ', strip=True)
    
    def _clean_text(self, text: str) -> str:
        """Cleans the extracted text"""
        if not text:
            return ""
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Remove multiple empty lines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _create_summary(self, text: str, max_sentences: int = 3) -> str:
        """Creates a summary of the content"""
        if not text or len(text) < 100:
            return text
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= max_sentences:
            return text[:500] + "..." if len(text) > 500 else text
        
        # Take the first sentences that contain important information
        important_sentences = []
        for sentence in sentences[:10]:  # Examine the first 10 sentences
            if any(keyword in sentence.lower() for keyword in 
                   ['important', 'principal', 'essential', 'key', 'major', 'definition']):
                important_sentences.append(sentence)
        
        # If not enough important sentences, take the first ones
        if len(important_sentences) < max_sentences:
            summary_sentences = sentences[:max_sentences]
        else:
            summary_sentences = important_sentences[:max_sentences]
        
        return '. '.join(summary_sentences) + '.'
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extracts all links from the page"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            # Clean the URL
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
        """Extracts all images from the page"""
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
        """Extracts navigation elements"""
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
        """Extracts content sections"""
        sections = []
        
        # Sections with headers
        for i in range(1, 7):  # h1 to h6
            for header in soup.find_all(f'h{i}'):
                section_content = ""
                
                # Find content after the header
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
        """Extracts keywords from the text"""
        if not text:
            return []
        
        # Stop words in French and English
        stop_words = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'en', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'the', 'be',
            'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on',
            'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', text.lower())
        
        # Count frequencies
        word_freq = defaultdict(int)
        for word in words:
            if word not in stop_words:
                word_freq[word] += 1
        
        # Return the most frequent words
        return [word for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]]
    
    def _detect_language(self, text: str) -> str:
        """Detects the language of the text"""
        if not text:
            return "unknown"
        
        # French indicator words
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
        """Calculates a content quality score"""
        score = 0.0
        
        # Content length
        if len(text) > 1000:
            score += 3.0
        elif len(text) > 500:
            score += 2.0
        elif len(text) > 100:
            score += 1.0
        
        # Title quality
        if title and len(title) > 10:
            score += 1.0
        
        # Number of links
        if len(links) > 10:
            score += 2.0
        elif len(links) > 5:
            score += 1.0
        
        # Text/link ratio (avoid spam)
        if len(links) > 0:
            text_link_ratio = len(text) / len(links)
            if text_link_ratio > 100:
                score += 1.0
        
        # Presence of structure (sections, headers)
        if re.search(r'\n\s*\n', text):  # Paragraphs
            score += 1.0
        
        return min(score, 10.0)

class AdvancedWebNavigator:
    """Advanced web navigator for artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
    
    def __init__(self):
        self.session = requests.Session()
        self.content_extractor = AdvancedContentExtractor()
        
        # Header configuration
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Storage directories
        self.data_dir = Path("data/advanced_web_navigation")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.content_cache_dir = self.data_dir / "content_cache"
        self.content_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.navigation_logs_dir = self.data_dir / "navigation_logs"
        self.navigation_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.content_cache: Dict[str, WebPageContent] = {}
        self.url_blacklist: Set[str] = set()
        
        logger.info("✅ Advanced web navigator initialized")
    
    def navigate_deep(self, start_url: str, 
                     max_depth: int = 3, 
                     max_pages: int = 10,
                     navigation_strategy: str = 'breadth_first',
                     content_filter: Optional[callable] = None) -> NavigationPath:
        """
        Deep navigation within a website
        
        Args:
            start_url: Starting URL
            max_depth: Maximum navigation depth
            max_pages: Maximum number of pages to visit
            navigation_strategy: 'breadth_first', 'depth_first', 'quality_first'
            content_filter: Content filtering function
        """
        session_id = f"nav_{int(time.time())}"
        logger.info(f"🚀 Starting deep navigation: {start_url} (session: {session_id})")
        
        navigation_path = NavigationPath(
            start_url=start_url,
            visited_pages=[],
            navigation_depth=0,
            total_content_extracted=0,
            navigation_strategy=navigation_strategy,
            session_id=session_id,
            created_at=datetime.now()
        )
        
        # Navigation queue with priority
        navigation_queue = [(start_url, 0)]  # (url, depth)
        visited_urls = set()
        
        while navigation_queue and len(navigation_path.visited_pages) < max_pages:
            # Select the next URL according to the strategy
            if navigation_strategy == 'quality_first':
                # Sort by potential quality (to be implemented)
                navigation_queue.sort(key=lambda x: self._estimate_url_quality(x[0]), reverse=True)
            
            current_url, current_depth = navigation_queue.pop(0)
            
            if current_url in visited_urls or current_depth > max_depth:
                continue
            
            logger.info(f"📄 Navigating to: {current_url} (depth: {current_depth})")
            
            # Extract page content
            page_content = self.extract_page_content(current_url)
            
            if page_content.success:
                visited_urls.add(current_url)
                
                # Apply content filter if provided
                if content_filter is None or content_filter(page_content):
                    navigation_path.visited_pages.append(page_content)
                    navigation_path.total_content_extracted += len(page_content.cleaned_text)
                    navigation_path.navigation_depth = max(navigation_path.navigation_depth, current_depth)
                    
                    # Save content
                    self._save_page_content(page_content, session_id)
                    
                    logger.info(f"✅ Content extracted: {len(page_content.cleaned_text)} characters")
                    
                    # Add interesting links to the queue
                    if current_depth < max_depth:
                        interesting_links = self._select_navigation_links(page_content, visited_urls)
                        for link_url in interesting_links[:5]:  # Max 5 links per page
                            navigation_queue.append((link_url, current_depth + 1))
                else:
                    logger.info("❌ Content filtered, page skipped")
            else:
                logger.warning(f"⚠️ Extraction failed: {page_content.error_message}")
            
            # Delay between requests
            time.sleep(0.5)
        
        # Save navigation path
        self._save_navigation_path(navigation_path)
        
        logger.info(f"🏁 Navigation finished: {len(navigation_path.visited_pages)} pages visited")
        return navigation_path
    
    def extract_page_content(self, url: str) -> WebPageContent:
        """Extracts detailed content from a web page"""
        # Check cache
        if url in self.content_cache:
            logger.info(f"📋 Content retrieved from cache: {url}")
            return self.content_cache[url]
        
        try:
            logger.info(f"🔍 Extracting content: {url}")
            
            response = self.session.get(url, timeout=10, allow_redirects=True)
            response.raise_for_status()
            
            # Extract content
            page_content = self.content_extractor.extract_page_content(response.text, url)
            
            # Cache content
            self.content_cache[url] = page_content
            
            return page_content
            
        except Exception as e:
            logger.error(f"❌ Error during extraction from {url}: {str(e)}")
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
        """Searches and navigates through results"""
        logger.info(f"🔎 Searching and navigating for: {query}")
        
        # TODO: Integrate with Searx for search
        # For now, simulating results
        search_results = self._simulate_search_results(query)
        
        navigation_paths = []
        for result_url in search_results[:3]:  # Navigate through the first 3 results
            try:
                nav_path = self.navigate_deep(
                    start_url=result_url,
                    max_depth=2,
                    max_pages=5,
                    navigation_strategy='quality_first'
                )
                navigation_paths.append(nav_path)
            except Exception as e:
                logger.error(f"Error during navigation of {result_url}: {str(e)}")
        
        return navigation_paths
    
    def _select_navigation_links(self, page_content: WebPageContent, visited_urls: Set[str]) -> List[str]:
        """Selects the most interesting links for navigation"""
        interesting_links = []
        
        base_domain = urlparse(page_content.url).netloc
        
        for link in page_content.links:
            link_url = link['url']
            link_text = link['text'].lower()
            
            # Ignore already visited links
            if link_url in visited_urls:
                continue
            
            # Prioritize links from the same domain
            if urlparse(link_url).netloc != base_domain:
                continue
            
            # Ignore certain types of links
            if any(ignored in link_url.lower() for ignored in 
                   ['.pdf', '.jpg', '.png', '.gif', 'mailto:', 'javascript:', '#']):
                continue
            
            # Link score based on text
            link_score = 0
            
            # Interesting keywords in link text
            interesting_keywords = [
                'detail', 'more', 'view', 'read', 'article', 'page', 'section',
                'chapter', 'guide', 'tutorial', 'training', 'course', 'about',
                'contact', 'service', 'product', 'information'
            ]
            
            for keyword in interesting_keywords:
                if keyword in link_text:
                    link_score += 1
            
            # Avoid generic navigation links
            generic_links = ['home', 'menu', 'search']
            if any(generic in link_text for generic in generic_links):
                link_score -= 2
            
            if link_score > 0:
                interesting_links.append(link_url)
        
        # Sort by potential score
        return interesting_links[:10]
    
    def _estimate_url_quality(self, url: str) -> float:
        """Estimates the potential quality of a URL"""
        score = 0.0
        
        # URL length (neither too short nor too long)
        url_length = len(url)
        if 20 < url_length < 100:
            score += 1.0
        
        # Presence of keywords in the URL
        quality_keywords = ['article', 'guide', 'tutorial', 'about', 'detail', 'info']
        for keyword in quality_keywords:
            if keyword in url.lower():
                score += 0.5
        
        # Avoid spam URLs
        spam_indicators = ['ad', 'ads', 'popup', 'redirect', 'track']
        for indicator in spam_indicators:
            if indicator in url.lower():
                score -= 1.0
        
        return max(score, 0.0)
    
    def _simulate_search_results(self, query: str) -> List[str]:
        """Simulates search results (to be replaced by Searx)"""
        # Simple simulation - to be replaced by Searx integration
        return [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://www.ibm.com/cloud/what-is-artificial-intelligence",
            "https://www.techtarget.com/whatis/definition/artificial-intelligence-AI"
        ]
    
    def _save_page_content(self, page_content: WebPageContent, session_id: str):
        """Saves page content"""
        try:
            # Create a filename based on the URL
            url_hash = hashlib.md5(page_content.url.encode()).hexdigest()[:12]
            filename = f"{session_id}_{url_hash}.json"
            
            filepath = self.content_cache_dir / filename
            
            # Serialize content
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
            
            logger.info(f"💾 Content saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error during saving: {str(e)}")
    
    def _save_navigation_path(self, navigation_path: NavigationPath):
        """Saves the navigation path"""
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
            
            logger.info(f"🗺️ Navigation path saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving path: {str(e)}")

# Global instance
advanced_navigator = AdvancedWebNavigator()

def navigate_website_deep(url: str, max_depth: int = 3, max_pages: int = 10) -> NavigationPath:
    """Public interface for deep navigation"""
    return advanced_navigator.navigate_deep(url, max_depth, max_pages)

def extract_website_content(url: str) -> WebPageContent:
    """Public interface for content extraction"""
    return advanced_navigator.extract_page_content(url)

def search_and_navigate_websites(query: str) -> List[NavigationPath]:
    """Public interface for search and navigation"""
    return advanced_navigator.search_and_navigate(query)

if __name__ == "__main__":
    print("=== Advanced Web Navigation System Test ===")
    
    # Content extraction test
    test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Extraction test: {test_url}")
    
    content = extract_website_content(test_url)
    if content.success:
        print(f"✅ Title: {content.title}")
        print(f"✅ Content: {len(content.cleaned_text)} characters")
        print(f"✅ Summary: {content.summary[:200]}...")
        print(f"✅ Keywords: {content.keywords[:5]}")
        print(f"✅ Quality score: {content.content_quality_score}")
    else:
        print(f"❌ Error: {content.error_message}")
