#!/usr/bin/env python3
"""
Searx Interface Module for artificial intelligence   API GOOGLE GEMINI 2.0 FLASH
Enables autonomous searches with HTML parsing
"""

import requests
import logging
import json
import time
from typing import Dict, List, Optional, Any
from bs4 import BeautifulSoup
import urllib.parse
from dataclasses import dataclass

logger = logging.getLogger('SearxInterface')

@dataclass
class SearchResult:
    """Structured search result"""
    title: str
    url: str
    content: str
    engine: str
    score: float = 0.0
    metadata: Dict[str, Any] = None

class SearxInterface:
    """Interface for communicating with Searx"""
    
    def __init__(self, searx_url: str = "http://localhost:8080"):
        self.searx_url = searx_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-SearchBot/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en,fr;q=0.5', # Changed to prioritize English
            'Connection': 'keep-alive'
        })
        # More robust timeout configuration
        self.session.mount('http://', requests.adapters.HTTPAdapter(
            max_retries=requests.adapters.Retry(
                total=2,
                connect=2,
                read=2,
                backoff_factor=0.3,
                status_forcelist=(500, 502, 504)
            )
        ))
        self.is_running = False
        
        # Smart port manager integration
        self.port_manager = None
        self._init_port_manager()
        
        # Smart port manager integration
        self.port_manager = None
        self._init_port_manager()
        
        # Visual capture system integration
        self.visual_capture = None
        self._init_visual_capture()
        
    def _init_port_manager(self):
        """Initializes the smart port manager"""
        try:
            from port_manager import PortManager
            self.port_manager = PortManager()
            
            # Check if a Searx URL is already configured
            current_url = self.port_manager.get_current_searx_url()
            if current_url:
                self.searx_url = current_url
                logger.info(f"✅ Searx URL detected: {current_url}")
            
            logger.info("✅ Smart port manager initialized")
        except ImportError:
            logger.warning("⚠️ Port manager module not available")
        except Exception as e:
            logger.error(f"❌ Error initializing port manager: {e}")
        
    def _init_visual_capture(self):
        """Initializes the visual capture system"""
        try:
            from searx_visual_capture import SearxVisualCapture
            self.visual_capture = SearxVisualCapture(self.searx_url)
            logger.info("✅ Visual capture system initialized")
        except ImportError:
            logger.warning("⚠️ Visual capture module not available")
        except Exception as e:
            logger.error(f"❌ Error initializing visual capture: {e}")
        
    def start_searx(self) -> bool:
        """Starts the Searx container with smart port management"""
        try:
            if self.port_manager:
                # Use the smart port manager
                logger.info("🚀 Smart Searx startup...")
                success, url = self.port_manager.start_searx_smart()
                
                if success:
                    self.searx_url = url
                    self.is_running = True
                    logger.info(f"✅ Searx started on: {url}")
                    
                    # Update the visual capture system
                    if self.visual_capture:
                        self.visual_capture.searx_url = url
                    
                    return True
                else:
                    logger.error("❌ Smart Searx startup failed")
                    return False
            else:
                # Classic startup method (fallback)
                return self._start_searx_classic()
                
        except Exception as e:
            logger.error(f"❌ Error during Searx startup: {e}")
            return False
    
    def _start_searx_classic(self) -> bool:
        """Classic startup method (fallback)"""
        try:
            import subprocess
            logger.info("Starting Searx with Docker (classic method)...")
            
            # Check if Docker is available
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker is not available")
                return False
            
            # Start the Searx container
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 'up', '-d'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("Searx started successfully")
                # Wait for the service to be ready
                time.sleep(10)
                return self.check_health()
            else:
                logger.error(f"Error during Searx startup: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error during Searx startup: {e}")
            return False
    
    def check_health(self) -> bool:
        """Checks if Searx is accessible with progressive timeouts"""
        timeouts = [5, 10, 15]  # Progressive timeouts
        
        for timeout in timeouts:
            try:
                logger.debug(f"Searx connection test with timeout {timeout}s...")
                response = self.session.get(f"{self.searx_url}/", timeout=timeout)
                self.is_running = response.status_code == 200
                
                if self.is_running:
                    logger.info("Searx is operational")
                    return True
                else:
                    logger.warning(f"Searx responds with code: {response.status_code}")
                    
            except requests.exceptions.ReadTimeout:
                logger.warning(f"Read timeout ({timeout}s) - Searx might be starting up...")
                continue
            except requests.exceptions.ConnectTimeout:
                logger.warning(f"Connection timeout ({timeout}s) - Searx is not yet ready...")
                continue
            except requests.exceptions.ConnectionError:
                logger.warning("Searx is not accessible - the service might not be started")
                break
            except Exception as e:
                logger.error(f"Error checking Searx health: {e}")
                break
        
        self.is_running = False
        return False
    
    def search(self, query: str, category: str = "general", 
               language: str = "en", max_results: int = 10, # Changed default language to 'en'
               retry_count: int = 2) -> List[SearchResult]:
        """Performs a search and parses HTML results with automatic retry"""
        
        for attempt in range(retry_count + 1):
            try:
                if not self.is_running and not self.check_health():
                    if attempt == 0:
                        logger.warning("Searx is not available, attempting to start...")
                        if self.start_searx():
                            time.sleep(5)  # Wait for Searx to be ready
                        else:
                            logger.error("Unable to start Searx")
                            return []
                    else:
                        logger.error("Searx is still not available after startup attempt")
                        return []
                
                # Search parameters
                params = {
                    'q': query,
                    'category_general': '1' if category == 'general' else '0',
                    'category_videos': '1' if category == 'videos' else '0',
                    'category_it': '1' if category == 'it' else '0',
                    'language': language,
                    'format': 'html',
                    'pageno': '1'
                }
                
                logger.info(f"Searx search: '{query}' (category: {category}){f' - Attempt {attempt + 1}' if attempt > 0 else ''}")
                
                # Perform search with adaptive timeout
                response = self.session.post(
                    f"{self.searx_url}/search",
                    data=params,
                    timeout=(10, 30)  # (connect_timeout, read_timeout)
                )
                
                if response.status_code != 200:
                    logger.error(f"Search error: {response.status_code}")
                    if attempt < retry_count:
                        logger.info(f"Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    return []
                
                # Parse HTML results
                results = self._parse_html_results(response.text, max_results)
                if results:
                    return results
                elif attempt < retry_count:
                    logger.warning("No results found, retrying...")
                    time.sleep(1)
                    continue
                else:
                    return []
                    
            except requests.exceptions.ReadTimeout:
                logger.error(f"Read timeout during Searx search (attempt {attempt + 1}/{retry_count + 1})")
                if attempt < retry_count:
                    time.sleep(3)
                    continue
                return []
            except requests.exceptions.ConnectTimeout:
                logger.error(f"Connection timeout during Searx search (attempt {attempt + 1}/{retry_count + 1})")
                if attempt < retry_count:
                    time.sleep(3)
                    continue
                return []
            except requests.exceptions.ConnectionError:
                logger.error(f"Unable to connect to Searx for search (attempt {attempt + 1}/{retry_count + 1})")
                if attempt < retry_count:
                    time.sleep(5)
                    continue
                return []
            except Exception as e:
                logger.error(f"Error during search (attempt {attempt + 1}/{retry_count + 1}): {e}")
                if attempt < retry_count:
                    time.sleep(2)
                    continue
                return []
        
        return []
    
    def _parse_html_results(self, html_content: str, max_results: int) -> List[SearchResult]:
        """Parses Searx HTML results"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all results
            result_divs = soup.find_all('article', class_='result')
            
            for i, result_div in enumerate(result_divs[:max_results]):
                try:
                    # Extract title
                    title_elem = result_div.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else "No title"
                    
                    # Extract URL
                    link_elem = result_div.find('a')
                    url = link_elem.get('href', '') if link_elem else ''
                    
                    # Clean and retrieve the real URL for videos
                    if url:
                        url = self._clean_video_url(url, title)
                    
                    # Extract content/description
                    content_elem = result_div.find('p', class_='content')
                    content = content_elem.get_text(strip=True) if content_elem else ''
                    
                    # Extract used search engine
                    engine_elem = result_div.find('span', class_='engine')
                    engine = engine_elem.get_text(strip=True) if engine_elem else 'unknown'
                    
                    # Create the result
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        content=content,
                        engine=engine,
                        score=1.0 - (i * 0.1),  # Decreasing score
                        metadata={
                            'position': i + 1,
                            'html_snippet': str(result_div)[:500]
                        }
                    )
                    
                    results.append(search_result)
                    
                except Exception as e:
                    logger.warning(f"Error parsing a result: {e}")
                    continue
            
            logger.info(f"Parsed {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return []

    def _clean_video_url(self, url: str, title: str) -> str:
        """Cleans and retrieves the real URL for videos"""
        try:
            # If the URL contains 'x', try to retrieve the real URL
            if 'xxxxxxxxxx' in url or 'xxx' in url:
                # For YouTube, try to find the ID from the title or other sources
                if 'youtube.com' in url or 'youtu.be' in url:
                    # Try to find a YouTube ID pattern in the original URL
                    # If not possible, generate a YouTube search URL
                    search_query = urllib.parse.quote(title)
                    return f"https://www.youtube.com/results?search_query={search_query}"
                
                # For Vimeo
                elif 'vimeo.com' in url:
                    search_query = urllib.parse.quote(title)
                    return f"https://vimeo.com/search?q={search_query}"
                
                # For Dailymotion
                elif 'dailymotion.com' in url:
                    search_query = urllib.parse.quote(title)
                    return f"https://www.dailymotion.com/search/{search_query}"
                
                # For other platforms, return a general search URL
                else:
                    logger.warning(f"Masked video URL detected: {url}")
                    return f"[Masked video URL - Title: {title}]"
            
            # If the URL seems correct, return it as is
            return url
            
        except Exception as e:
            logger.error(f"Error cleaning the URL: {e}")
            return url
    
    def search_with_filters(self, query: str, engines: List[str] = None,
                           time_range: str = None, safe_search: int = 0) -> List[SearchResult]:
        """Advanced search with filters"""
        
        try:
            params = {
                'q': query,
                'format': 'html',
                'safesearch': str(safe_search)
            }
            
            # Add specific engines
            if engines:
                for engine in engines:
                    params[f'engine_{engine}'] = '1'
            
            # Add time range
            if time_range:
                params['time_range'] = time_range
            
            response = self.session.post(
                f"{self.searx_url}/search",
                data=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return self._parse_html_results(response.text, 20)
            else:
                logger.error(f"Advanced search error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error during advanced search: {e}")
            return []
    
    def search_with_visual(self, query: str, category: str = "general", 
                          language: str = "en", max_results: int = 10) -> Dict[str, Any]: # Changed default language to 'en'
        """Performs a search with visual capture for AI"""
        
        # Classic text search
        text_results = self.search(query, category, language, max_results)
        
        # Visual capture if available
        visual_data = None
        if self.visual_capture:
            try:
                visual_data = self.visual_capture.capture_with_annotations(query, category)
                logger.info("✅ Visual capture successful")
            except Exception as e:
                logger.error(f"❌ Visual capture error: {e}")
        
        return {
            'query': query,
            'category': category,
            'text_results': text_results,
            'visual_data': visual_data,
            'has_visual': visual_data is not None and visual_data.get('success', False),
            'timestamp': time.time()
        }
    
    def get_visual_search_summary(self, search_result: Dict[str, Any]) -> str:
        """Generates a summary for AI including visual data"""
        
        summary = f"""🔍 **Searx Search with Visual Analysis**

**Query**: {search_result['query']}
**Category**: {search_result['category']}

"""
        
        # Textual results
        if search_result.get('text_results'):
            summary += f"**Textual results found**: {len(search_result['text_results'])}\n\n"
            
            for i, result in enumerate(search_result['text_results'][:3], 1):
                summary += f"**{i}. {result.title}**\n"
                summary += f"Source: {result.engine}\n"
                summary += f"URL: {result.url}\n"
                summary += f"Content: {result.content[:200]}{'...' if len(result.content) > 200 else ''}\n\n"
        
        # Visual data
        if search_result.get('has_visual'):
            visual_data = search_result['visual_data']
            summary += "**📸 Visual Analysis Available**\n"
            summary += f"Screenshot: {visual_data.get('screenshot_path', 'N/A')}\n"
            
            if visual_data.get('page_text_context'):
                summary += f"\n**Extracted visual context**:\n{visual_data['page_text_context'][:300]}...\n"
        else:
            summary += "**⚠️ Visual Analysis Not Available**\n"
        return summary # Added missing return statement
    
    def get_suggestions(self, query: str) -> List[str]:
        """Gets search suggestions"""
        try:
            params = {
                'q': query,
                'format': 'json'
            }
            
            response = self.session.get(
                f"{self.searx_url}/autocompleter",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('suggestions', [])
            else:
                return []
                
        except Exception as e:
            logger.warning(f"Unable to get suggestions: {e}")
            return []
    
    def cleanup_visual_data(self):
        """Cleans old visual data"""
        if self.visual_capture:
            try:
                self.visual_capture.cleanup_old_screenshots()
            except Exception as e:
                logger.error(f"Visual cleanup error: {e}")
    
    def close_visual_capture(self):
        """Closes the visual capture system"""
        if self.visual_capture:
            try:
                self.visual_capture.close()
                logger.info("Visual capture system closed")
            except Exception as e:
                logger.error(f"Capture close error: {e}")
    
    def stop_searx(self) -> bool:
        """Stops the Searx container"""
        try:
            if self.port_manager:
                # Use the smart manager to stop
                success = self.port_manager.stop_all_searx_containers()
                if success:
                    self.is_running = False
                    logger.info("✅ Searx stopped via smart manager")
                return success
            else:
                # Classic method
                return self._stop_searx_classic()
                
        except Exception as e:
            logger.error(f"Error during Searx shutdown: {e}")
            return False
    
    def _stop_searx_classic(self) -> bool:
        """Stops Searx with the classic method"""
        try:
            import subprocess
            
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 'down'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("Searx stopped successfully")
                self.is_running = False
                return True
            else:
                logger.error(f"Error during Searx shutdown: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error during Searx shutdown: {e}")
            return False

# Global instance
searx_interface = SearxInterface()

def get_searx_interface() -> SearxInterface:
    """Returns the Searx interface instance"""
    return searx_interface

if __name__ == "__main__":
    # Module test
    searx = SearxInterface()
    
    if searx.start_searx():
        # Search test
        results = searx.search("artificial intelligence", max_results=5)
        
        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Engine: {result.engine}")
            print(f"   Content: {result.content[:100]}...")
    else:
        print("Unable to start Searx")
