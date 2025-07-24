
"""
Module spécialisé pour la recherche d'appartements sur Leboncoin
Permet à l'IA d'obtenir de vrais liens vers des annonces d'appartements
"""

import requests
import re
import time
import logging
from typing import List, Dict, Any
from urllib.parse import urlencode, quote
from datetime import datetime

logger = logging.getLogger(__name__)

class LeboncoinSearcher:
    """Recherche spécialisée d'appartements sur Leboncoin"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive'
        })
        
    def search_apartments_hauts_de_france(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """Recherche des appartements dans les Hauts-de-France"""
        
        logger.info("🔍 Recherche d'appartements dans les Hauts-de-France sur Leboncoin")
        
        # URLs de recherche pour différentes villes des Hauts-de-France
        search_urls = [
            "https://www.leboncoin.fr/recherche?category=9&regions=6&real_estate_type=2",  # Appartements Hauts-de-France
            "https://www.leboncoin.fr/recherche?category=9&locations=Lille_59000__45.48324_2.93576_5565&real_estate_type=2",  # Lille
            "https://www.leboncoin.fr/recherche?category=9&locations=Amiens_80000__49.89427_2.29576_5565&real_estate_type=2",  # Amiens
            "https://www.leboncoin.fr/recherche?category=9&locations=Roubaix_59100__50.69421_3.17456_5565&real_estate_type=2",  # Roubaix
            "https://www.leboncoin.fr/recherche?category=9&locations=Tourcoing_59200__50.72429_3.15789_5565&real_estate_type=2"  # Tourcoing
        ]
        
        all_apartments = []
        
        for search_url in search_urls:
            try:
                logger.info(f"Recherche sur: {search_url}")
                
                response = self.session.get(search_url, timeout=10)
                response.raise_for_status()
                
                # Extraire les liens d'annonces
                apartment_links = self._extract_apartment_links(response.text)
                
                for link in apartment_links[:5]:  # 5 par ville max
                    apartment_info = self._get_apartment_info(link)
                    if apartment_info:
                        all_apartments.append(apartment_info)
                        
                        if len(all_apartments) >= max_results:
                            break
                
                time.sleep(1)  # Respect du site
                
                if len(all_apartments) >= max_results:
                    break
                    
            except Exception as e:
                logger.error(f"Erreur lors de la recherche sur {search_url}: {str(e)}")
                continue
        
        logger.info(f"✅ {len(all_apartments)} appartements trouvés")
        return all_apartments
    
    def _extract_apartment_links(self, html_content: str) -> List[str]:
        """Extrait les liens vers les annonces d'appartements"""
        links = []
        
        # Patterns pour les liens d'annonces Leboncoin
        patterns = [
            r'href="(/ad/[^"]+)"',
            r'href="(https://www\.leboncoin\.fr/ad/[^"]+)"'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content, re.IGNORECASE)
            for match in matches:
                if match.startswith('/'):
                    full_url = f"https://www.leboncoin.fr{match}"
                else:
                    full_url = match
                
                if full_url not in links:
                    links.append(full_url)
        
        return links[:20]  # Limiter à 20 liens
    
    def _get_apartment_info(self, apartment_url: str) -> Dict[str, Any]:
        """Récupère les informations d'un appartement"""
        
        try:
            response = self.session.get(apartment_url, timeout=10)
            response.raise_for_status()
            
            content = response.text
            
            # Extraire le titre
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content)
            title = title_match.group(1).strip() if title_match else "Appartement"
            
            # Extraire le prix
            price_patterns = [
                r'(\d+(?:\s?\d+)*)\s*€',
                r'Prix\s*:\s*(\d+(?:\s?\d+)*)\s*€'
            ]
            
            price = "Prix non spécifié"
            for pattern in price_patterns:
                price_match = re.search(pattern, content)
                if price_match:
                    price = f"{price_match.group(1)} €"
                    break
            
            # Extraire la localisation
            location_patterns = [
                r'<span[^>]*data-qa-id="adview_location_informations"[^>]*>([^<]+)</span>',
                r'Ville\s*:\s*([^<\n]+)'
            ]
            
            location = "Localisation non spécifiée"
            for pattern in location_patterns:
                location_match = re.search(pattern, content)
                if location_match:
                    location = location_match.group(1).strip()
                    break
            
            return {
                "url": apartment_url,
                "title": title,
                "price": price,
                "location": location,
                "found_at": datetime.now().isoformat(),
                "source": "leboncoin"
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des infos pour {apartment_url}: {str(e)}")
            return None

# Instance globale
leboncoin_searcher = LeboncoinSearcher()

def search_real_apartments_hauts_de_france(max_results: int = 10) -> List[Dict[str, Any]]:
    """Interface publique pour rechercher de vrais appartements"""
    return leboncoin_searcher.search_apartments_hauts_de_france(max_results)

if __name__ == "__main__":
    print("=== Test de recherche d'appartements Leboncoin ===")
    apartments = search_real_apartments_hauts_de_france(5)
    
    for i, apt in enumerate(apartments, 1):
        print(f"\n{i}. {apt['title']}")
        print(f"   Prix: {apt['price']}")
        print(f"   Lieu: {apt['location']}")
        print(f"   URL: {apt['url']}")
