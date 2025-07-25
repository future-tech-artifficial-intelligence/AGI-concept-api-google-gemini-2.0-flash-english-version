"""
old specialized web scraping module for searching listings on Leboncoin                
Allows the artificial intelligence API GOOGLE GEMINI 2.0 FLASH to get real links to various ads 
now artificial intelligence API GOOGLE GEMINI 2.0 FLASH 
uses the searx search engine, it allows more precise searches instead of a simple HTML page extraction 
"""

import requests
import re
import time
import logging
from typing import List, Dict, Any
from urllib.parse import urlencode, quote
from datetime import datetime

logger = logging.getLogger(__name__)

class ListingSearcher:
    """Specialized listing search on a generic platform (e.g., Leboncoin-like)"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive'
        })
        
    def search_listings(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Searches for various listings in different locations"""
        
        logger.info("🔍 Searching for listings on the platform")
        
        # Placeholder search URLs (These would need to be replaced with actual search URLs for a specific platform)
        search_urls = [
            "https://www.example.com/search?q=item_type_A&location=city_X",
            "https://www.example.com/search?q=item_type_B&location=city_Y",
            "https://www.example.com/search?q=any_item&location=city_Z"
        ]
        
        all_listings = []
        
        for search_url in search_urls:
            try:
                logger.info(f"Searching on: {search_url}")
                
                response = self.session.get(search_url, timeout=10)
                response.raise_for_status()
                
                # Extract listing links
                listing_links = self._extract_listing_links(response.text)
                
                for link in listing_links[:5]:  # 5 per search URL max
                    listing_info = self._get_listing_info(link)
                    if listing_info:
                        all_listings.append(listing_info)
                        
                        if len(all_listings) >= max_results:
                            break
                
                time.sleep(1)  # Respect site
                
                if len(all_listings) >= max_results:
                    break
                    
            except Exception as e:
                logger.error(f"Error during search on {search_url}: {str(e)}")
                continue
        
        logger.info(f"✅ {len(all_listings)} listings found")
        return all_listings
    
    def _extract_listing_links(self, html_content: str) -> List[str]:
        """Extracts links to listings"""
        links = []
        
        # Generic patterns for listing links (adjust based on target website structure)
        # These are highly dependent on the actual website HTML structure
        patterns = [
            r'href="(/ad/[^"]+)"',  # Example for relative paths like /ad/listing_id
            r'href="(https?://[^/]+/ad/[^"]+)"', # Example for full paths
            r'href="(https?://[^/]+/listings/[^"]+)"' # Another generic example
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                # This part would need platform-specific logic to construct full URLs
                # For generic example, assume it's already a full URL or construct based on a base URL
                if match.startswith('http'):
                    full_url = match
                else:
                    # This requires a base URL to be meaningful
                    full_url = f"https://www.example.com{match}" # Replace example.com with actual base URL
                
                if full_url not in links:
                    links.append(full_url)
        
        return links[:20]  # Limit to 20 links
    
    def _get_listing_info(self, listing_url: str) -> Dict[str, Any]:
        """Retrieves listing information"""
        
        try:
            response = self.session.get(listing_url, timeout=10)
            response.raise_for_status()
            
            content = response.text
            
            # Extract title (generic pattern)
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content)
            title = title_match.group(1).strip() if title_match else "Listing"
            
            # Extract price (generic patterns)
            price_patterns = [
                r'(\d+(?:\s?\d+)*)\s*€', # Matches "1 234 €" or "1234€"
                r'Price\s*:\s*(\d+(?:\s?\d+)*)\s*€',
                r'\$\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', # Matches "$1,234.56"
                r'€\s*(\d+(?:\s?\d+)*)' # Matches "€ 1234"
            ]
            
            price = "Price not specified"
            for pattern in price_patterns:
                price_match = re.search(pattern, content)
                if price_match:
                    price = price_match.group(0).strip() # Capture the whole match including currency
                    break
            
            # Extract location (generic patterns)
            location_patterns = [
                r'<span[^>]*class="location-info"[^>]*>([^<]+)</span>', # Common class for location
                r'Location\s*:\s*([^<\n]+)',
                r'Adresse\s*:\s*([^<\n]+)' # French equivalent
            ]
            
            location = "Location not specified"
            for pattern in location_patterns:
                location_match = re.search(pattern, content)
                if location_match:
                    location = location_match.group(1).strip()
                    break
            
            return {
                "url": listing_url,
                "title": title,
                "price": price,
                "location": location,
                "found_at": datetime.now().isoformat(),
                "source": "generic_platform_scraper" # Generic source
            }
            
        except Exception as e:
            logger.error(f"Error extracting info for {listing_url}: {str(e)}")
            return None

# Global instance
listing_searcher = ListingSearcher()

def search_real_listings(max_results: int = 10) -> List[Dict[str, Any]]:
    """Public interface for searching real listings"""
    return listing_searcher.search_listings(max_results)

if __name__ == "__main__":
    print("=== Generic Listing Search Test ===")
    listings = search_real_listings(5)
    
    for i, item in enumerate(listings, 1):
        print(f"\n{i}. {item['title']}")
        print(f"   Price: {item['price']}")
        print(f"   Location: {item['location']}")
        print(f"   URL: {item['url']}")
