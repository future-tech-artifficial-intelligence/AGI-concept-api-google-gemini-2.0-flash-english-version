#!/usr/bin/env python3
"""
Module d'interface Searx pour l'IA
Permet des recherches autonomes avec parsing HTML
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
    """Résultat de recherche structuré"""
    title: str
    url: str
    content: str
    engine: str
    score: float = 0.0
    metadata: Dict[str, Any] = None

class SearxInterface:
    """Interface pour communiquer avec Searx"""
    
    def __init__(self, searx_url: str = "http://localhost:8080"):
        self.searx_url = searx_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AI-SearchBot/1.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fr,en;q=0.5',
            'Connection': 'keep-alive'
        })
        # Configuration plus robuste pour les timeouts
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
        
        # Intégration du gestionnaire de ports intelligent
        self.port_manager = None
        self._init_port_manager()
        
        # Intégration du gestionnaire de ports intelligent
        self.port_manager = None
        self._init_port_manager()
        
        # Intégration du système de capture visuelle
        self.visual_capture = None
        self._init_visual_capture()
        
    def _init_port_manager(self):
        """Initialise le gestionnaire de ports intelligent"""
        try:
            from port_manager import PortManager
            self.port_manager = PortManager()
            
            # Vérifier si une URL Searx est déjà configurée
            current_url = self.port_manager.get_current_searx_url()
            if current_url:
                self.searx_url = current_url
                logger.info(f"✅ URL Searx détectée: {current_url}")
            
            logger.info("✅ Gestionnaire de ports intelligent initialisé")
        except ImportError:
            logger.warning("⚠️ Module gestionnaire de ports non disponible")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation gestionnaire de ports: {e}")
        
    def _init_visual_capture(self):
        """Initialise le système de capture visuelle"""
        try:
            from searx_visual_capture import SearxVisualCapture
            self.visual_capture = SearxVisualCapture(self.searx_url)
            logger.info("✅ Système de capture visuelle initialisé")
        except ImportError:
            logger.warning("⚠️ Module de capture visuelle non disponible")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation capture visuelle: {e}")
        
    def start_searx(self) -> bool:
        """Démarre le conteneur Searx avec gestion intelligente des ports"""
        try:
            if self.port_manager:
                # Utiliser le gestionnaire intelligent de ports
                logger.info("🚀 Démarrage intelligent de Searx...")
                success, url = self.port_manager.start_searx_smart()
                
                if success:
                    self.searx_url = url
                    self.is_running = True
                    logger.info(f"✅ Searx démarré sur: {url}")
                    
                    # Mettre à jour le système de capture visuelle
                    if self.visual_capture:
                        self.visual_capture.searx_url = url
                    
                    return True
                else:
                    logger.error("❌ Échec démarrage intelligent de Searx")
                    return False
            else:
                # Méthode de démarrage classique (fallback)
                return self._start_searx_classic()
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du démarrage de Searx: {e}")
            return False
    
    def _start_searx_classic(self) -> bool:
        """Méthode de démarrage classique (fallback)"""
        try:
            import subprocess
            logger.info("Démarrage de Searx avec Docker (méthode classique)...")
            
            # Vérifier si Docker est disponible
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker n'est pas disponible")
                return False
            
            # Démarrer le conteneur Searx
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 'up', '-d'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("Searx démarré avec succès")
                # Attendre que le service soit prêt
                time.sleep(10)
                return self.check_health()
            else:
                logger.error(f"Erreur lors du démarrage de Searx: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de Searx: {e}")
            return False
    
    def check_health(self) -> bool:
        """Vérifie si Searx est accessible avec timeouts progressifs"""
        timeouts = [5, 10, 15]  # Timeouts progressifs
        
        for timeout in timeouts:
            try:
                logger.debug(f"Test de connexion Searx avec timeout {timeout}s...")
                response = self.session.get(f"{self.searx_url}/", timeout=timeout)
                self.is_running = response.status_code == 200
                
                if self.is_running:
                    logger.info("Searx est opérationnel")
                    return True
                else:
                    logger.warning(f"Searx répond avec le code: {response.status_code}")
                    
            except requests.exceptions.ReadTimeout:
                logger.warning(f"Timeout de lecture ({timeout}s) - Searx peut être en cours de démarrage...")
                continue
            except requests.exceptions.ConnectTimeout:
                logger.warning(f"Timeout de connexion ({timeout}s) - Searx n'est pas encore prêt...")
                continue
            except requests.exceptions.ConnectionError:
                logger.warning("Searx n'est pas accessible - le service n'est peut-être pas démarré")
                break
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de Searx: {e}")
                break
        
        self.is_running = False
        return False
    
    def search(self, query: str, category: str = "general", 
               language: str = "fr", max_results: int = 10, 
               retry_count: int = 2) -> List[SearchResult]:
        """Effectue une recherche et parse les résultats HTML avec retry automatique"""
        
        for attempt in range(retry_count + 1):
            try:
                if not self.is_running and not self.check_health():
                    if attempt == 0:
                        logger.warning("Searx n'est pas disponible, tentative de démarrage...")
                        if self.start_searx():
                            time.sleep(5)  # Attendre que Searx soit prêt
                        else:
                            logger.error("Impossible de démarrer Searx")
                            return []
                    else:
                        logger.error("Searx n'est toujours pas disponible après tentative de démarrage")
                        return []
                
                # Paramètres de recherche
                params = {
                    'q': query,
                    'category_general': '1' if category == 'general' else '0',
                    'category_videos': '1' if category == 'videos' else '0',
                    'category_it': '1' if category == 'it' else '0',
                    'language': language,
                    'format': 'html',
                    'pageno': '1'
                }
                
                logger.info(f"Recherche Searx: '{query}' (catégorie: {category}){f' - Tentative {attempt + 1}' if attempt > 0 else ''}")
                
                # Effectuer la recherche avec timeout adaptatif
                response = self.session.post(
                    f"{self.searx_url}/search",
                    data=params,
                    timeout=(10, 30)  # (connect_timeout, read_timeout)
                )
                
                if response.status_code != 200:
                    logger.error(f"Erreur de recherche: {response.status_code}")
                    if attempt < retry_count:
                        logger.info(f"Nouvelle tentative dans 2 secondes...")
                        time.sleep(2)
                        continue
                    return []
                
                # Parser les résultats HTML
                results = self._parse_html_results(response.text, max_results)
                if results:
                    return results
                elif attempt < retry_count:
                    logger.warning("Aucun résultat trouvé, nouvelle tentative...")
                    time.sleep(1)
                    continue
                else:
                    return []
                    
            except requests.exceptions.ReadTimeout:
                logger.error(f"Timeout de lecture lors de la recherche Searx (tentative {attempt + 1}/{retry_count + 1})")
                if attempt < retry_count:
                    time.sleep(3)
                    continue
                return []
            except requests.exceptions.ConnectTimeout:
                logger.error(f"Timeout de connexion lors de la recherche Searx (tentative {attempt + 1}/{retry_count + 1})")
                if attempt < retry_count:
                    time.sleep(3)
                    continue
                return []
            except requests.exceptions.ConnectionError:
                logger.error(f"Impossible de se connecter à Searx pour la recherche (tentative {attempt + 1}/{retry_count + 1})")
                if attempt < retry_count:
                    time.sleep(5)
                    continue
                return []
            except Exception as e:
                logger.error(f"Erreur lors de la recherche (tentative {attempt + 1}/{retry_count + 1}): {e}")
                if attempt < retry_count:
                    time.sleep(2)
                    continue
                return []
        
        return []
    
    def _parse_html_results(self, html_content: str, max_results: int) -> List[SearchResult]:
        """Parse les résultats HTML de Searx"""
        results = []
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Trouver tous les résultats
            result_divs = soup.find_all('article', class_='result')
            
            for i, result_div in enumerate(result_divs[:max_results]):
                try:
                    # Extraire le titre
                    title_elem = result_div.find('h3')
                    title = title_elem.get_text(strip=True) if title_elem else "Sans titre"
                    
                    # Extraire l'URL
                    link_elem = result_div.find('a')
                    url = link_elem.get('href', '') if link_elem else ''
                    
                    # Nettoyer et récupérer la vraie URL pour les vidéos
                    if url:
                        url = self._clean_video_url(url, title)
                    
                    # Extraire le contenu/description
                    content_elem = result_div.find('p', class_='content')
                    content = content_elem.get_text(strip=True) if content_elem else ''
                    
                    # Extraire le moteur de recherche utilisé
                    engine_elem = result_div.find('span', class_='engine')
                    engine = engine_elem.get_text(strip=True) if engine_elem else 'unknown'
                    
                    # Créer le résultat
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        content=content,
                        engine=engine,
                        score=1.0 - (i * 0.1),  # Score décroissant
                        metadata={
                            'position': i + 1,
                            'html_snippet': str(result_div)[:500]
                        }
                    )
                    
                    results.append(search_result)
                    
                except Exception as e:
                    logger.warning(f"Erreur lors du parsing d'un résultat: {e}")
                    continue
            
            logger.info(f"Parsed {len(results)} résultats de recherche")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du parsing HTML: {e}")
            return []

    def _clean_video_url(self, url: str, title: str) -> str:
        """Nettoie et récupère la vraie URL pour les vidéos"""
        try:
            # Si l'URL contient des 'x', essayer de récupérer la vraie URL
            if 'xxxxxxxxxx' in url or 'xxx' in url:
                # Pour YouTube, essayer de retrouver l'ID depuis le titre ou d'autres sources
                if 'youtube.com' in url or 'youtu.be' in url:
                    # Essayer de trouver un pattern d'ID YouTube dans l'URL originale
                    # Si on ne peut pas, on génère une URL de recherche YouTube
                    search_query = urllib.parse.quote(title)
                    return f"https://www.youtube.com/results?search_query={search_query}"
                
                # Pour Vimeo
                elif 'vimeo.com' in url:
                    search_query = urllib.parse.quote(title)
                    return f"https://vimeo.com/search?q={search_query}"
                
                # Pour Dailymotion
                elif 'dailymotion.com' in url:
                    search_query = urllib.parse.quote(title)
                    return f"https://www.dailymotion.com/search/{search_query}"
                
                # Pour d'autres plateformes, retourner une URL de recherche générale
                else:
                    logger.warning(f"URL vidéo masquée détectée: {url}")
                    return f"[URL vidéo masquée - Titre: {title}]"
            
            # Si l'URL semble correcte, la retourner telle quelle
            return url
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage de l'URL: {e}")
            return url
    
    def search_with_filters(self, query: str, engines: List[str] = None,
                           time_range: str = None, safe_search: int = 0) -> List[SearchResult]:
        """Recherche avancée avec filtres"""
        
        try:
            params = {
                'q': query,
                'format': 'html',
                'safesearch': str(safe_search)
            }
            
            # Ajouter les moteurs spécifiques
            if engines:
                for engine in engines:
                    params[f'engine_{engine}'] = '1'
            
            # Ajouter la plage de temps
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
                logger.error(f"Erreur de recherche avancée: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Erreur lors de la recherche avancée: {e}")
            return []
    
    def search_with_visual(self, query: str, category: str = "general", 
                          language: str = "fr", max_results: int = 10) -> Dict[str, Any]:
        """Effectue une recherche avec capture visuelle pour l'IA"""
        
        # Recherche textuelle classique
        text_results = self.search(query, category, language, max_results)
        
        # Capture visuelle si disponible
        visual_data = None
        if self.visual_capture:
            try:
                visual_data = self.visual_capture.capture_with_annotations(query, category)
                logger.info("✅ Capture visuelle réussie")
            except Exception as e:
                logger.error(f"❌ Erreur capture visuelle: {e}")
        
        return {
            'query': query,
            'category': category,
            'text_results': text_results,
            'visual_data': visual_data,
            'has_visual': visual_data is not None and visual_data.get('success', False),
            'timestamp': time.time()
        }
    
    def get_visual_search_summary(self, search_result: Dict[str, Any]) -> str:
        """Génère un résumé pour l'IA incluant les données visuelles"""
        
        summary = f"""🔍 **Recherche Searx avec analyse visuelle**

**Requête**: {search_result['query']}
**Catégorie**: {search_result['category']}

"""
        
        # Résultats textuels
        if search_result.get('text_results'):
            summary += f"**Résultats textuels trouvés**: {len(search_result['text_results'])}\n\n"
            
            for i, result in enumerate(search_result['text_results'][:3], 1):
                summary += f"**{i}. {result.title}**\n"
                summary += f"Source: {result.engine}\n"
                summary += f"URL: {result.url}\n"
                summary += f"Contenu: {result.content[:200]}{'...' if len(result.content) > 200 else ''}\n\n"
        
        # Données visuelles
        if search_result.get('has_visual'):
            visual_data = search_result['visual_data']
            summary += "**📸 Analyse visuelle disponible**\n"
            summary += f"Capture d'écran: {visual_data.get('screenshot_path', 'N/A')}\n"
            
            if visual_data.get('page_text_context'):
                summary += f"\n**Contexte visuel extrait**:\n{visual_data['page_text_context'][:300]}...\n"
        else:
            summary += "**⚠️ Analyse visuelle non disponible**\n"
        
    def get_suggestions(self, query: str) -> List[str]:
        """Obtient des suggestions de recherche"""
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
            logger.warning(f"Impossible d'obtenir les suggestions: {e}")
            return []
    
    def cleanup_visual_data(self):
        """Nettoie les données visuelles anciennes"""
        if self.visual_capture:
            try:
                self.visual_capture.cleanup_old_screenshots()
            except Exception as e:
                logger.error(f"Erreur nettoyage visuel: {e}")
    
    def close_visual_capture(self):
        """Ferme le système de capture visuelle"""
        if self.visual_capture:
            try:
                self.visual_capture.close()
                logger.info("Système de capture visuelle fermé")
            except Exception as e:
                logger.error(f"Erreur fermeture capture: {e}")
    
    def stop_searx(self) -> bool:
        """Arrête le conteneur Searx"""
        try:
            if self.port_manager:
                # Utiliser le gestionnaire intelligent pour arrêter
                success = self.port_manager.stop_all_searx_containers()
                if success:
                    self.is_running = False
                    logger.info("✅ Searx arrêté via gestionnaire intelligent")
                return success
            else:
                # Méthode classique
                return self._stop_searx_classic()
                
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt de Searx: {e}")
            return False
    
    def _stop_searx_classic(self) -> bool:
        """Arrête Searx avec la méthode classique"""
        try:
            import subprocess
            
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 'down'
            ], capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("Searx arrêté avec succès")
                self.is_running = False
                return True
            else:
                logger.error(f"Erreur lors de l'arrêt de Searx: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt de Searx: {e}")
            return False

# Instance globale
searx_interface = SearxInterface()

def get_searx_interface() -> SearxInterface:
    """Retourne l'instance de l'interface Searx"""
    return searx_interface

if __name__ == "__main__":
    # Test du module
    searx = SearxInterface()
    
    if searx.start_searx():
        # Test de recherche
        results = searx.search("intelligence artificielle", max_results=5)
        
        print(f"Trouvé {len(results)} résultats:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Moteur: {result.engine}")
            print(f"   Contenu: {result.content[:100]}...")
    else:
        print("Impossible de démarrer Searx")
