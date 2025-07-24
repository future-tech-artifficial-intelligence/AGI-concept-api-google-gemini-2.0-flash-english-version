#!/usr/bin/env python3
"""
Module de capture visuelle Searx pour l'IA
Permet à l'API Gemini de voir les résultats de recherche visuellement
"""

import os
import time
import logging
import base64
import io
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import requests

logger = logging.getLogger('SearxVisualCapture')

class SearxVisualCapture:
    """Système de capture visuelle pour Searx"""
    
    def __init__(self, searx_url: str = "http://localhost:8080"):
        self.searx_url = searx_url.rstrip('/')
        self.screenshots_dir = "searx_screenshots"
        self.webdriver = None
        self.driver_initialized = False
        
        # Créer le répertoire de captures
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    def _initialize_webdriver(self) -> bool:
        """Initialise le WebDriver Chrome/Edge en mode headless"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.edge.options import Options as EdgeOptions
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            # Essayer Chrome d'abord
            try:
                chrome_options = ChromeOptions()
                chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-plugins')
                chrome_options.add_argument('--disable-images')  # Optimisation
                
                self.webdriver = webdriver.Chrome(options=chrome_options)
                logger.info("✅ WebDriver Chrome initialisé")
                
            except Exception as chrome_error:
                logger.warning(f"Chrome non disponible: {chrome_error}")
                
                # Essayer Edge comme alternative
                try:
                    edge_options = EdgeOptions()
                    edge_options.add_argument('--headless')
                    edge_options.add_argument('--no-sandbox')
                    edge_options.add_argument('--disable-dev-shm-usage')
                    edge_options.add_argument('--disable-gpu')
                    edge_options.add_argument('--window-size=1920,1080')
                    
                    self.webdriver = webdriver.Edge(options=edge_options)
                    logger.info("✅ WebDriver Edge initialisé")
                    
                except Exception as edge_error:
                    logger.error(f"Aucun WebDriver disponible. Chrome: {chrome_error}, Edge: {edge_error}")
                    return False
            
            self.driver_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Selenium non installé: {e}")
            logger.error("Installez avec: pip install selenium")
            return False
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du WebDriver: {e}")
            return False
    
    def capture_search_results(self, query: str, category: str = "general") -> Optional[Dict[str, Any]]:
        """Capture visuellement les résultats de recherche Searx"""
        
        if not self.driver_initialized and not self._initialize_webdriver():
            logger.error("Impossible d'initialiser le WebDriver")
            return None
        
        try:
            # URL de recherche Searx
            search_url = f"{self.searx_url}/search"
            params = {
                'q': query,
                'category_general': '1' if category == 'general' else '0',
                'category_videos': '1' if category == 'videos' else '0',
                'category_it': '1' if category == 'it' else '0',
                'format': 'html'
            }
            
            # Construire l'URL complète
            param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{search_url}?{param_string}"
            
            logger.info(f"📸 Capture visuelle: '{query}' ({category})")
            
            # Naviguer vers la page de résultats
            self.webdriver.get(full_url)
            
            # Attendre que les résultats se chargent
            time.sleep(3)
            
            # Prendre une capture d'écran
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"searx_search_{timestamp}_{query[:20].replace(' ', '_')}.png"
            filepath = os.path.join(self.screenshots_dir, filename)
            
            # Capture d'écran de la page complète
            self.webdriver.save_screenshot(filepath)
            
            # Capturer aussi une version optimisée pour l'IA
            optimized_image = self._optimize_screenshot_for_ai(filepath)
            
            # Extraire le texte visible pour contexte
            page_text = self._extract_visible_text()
            
            result = {
                'query': query,
                'category': category,
                'screenshot_path': filepath,
                'optimized_image': optimized_image,
                'page_text_context': page_text,
                'timestamp': timestamp,
                'url': full_url,
                'success': True
            }
            
            logger.info(f"✅ Capture réussie: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de la capture: {e}")
            return {
                'query': query,
                'category': category,
                'error': str(e),
                'success': False
            }
    
    def _optimize_screenshot_for_ai(self, screenshot_path: str) -> Optional[str]:
        """Optimise la capture d'écran pour l'analyse IA"""
        try:
            # Ouvrir l'image
            image = Image.open(screenshot_path)
            
            # Redimensionner pour l'IA (optimisation)
            max_width = 1024
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Améliorer le contraste pour une meilleure lisibilité
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Convertir en base64 pour l'API
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation de l'image: {e}")
            return None
    
    def _extract_visible_text(self) -> str:
        """Extrait le texte visible de la page pour contexte"""
        try:
            from selenium.webdriver.common.by import By
            
            # Extraire le texte des résultats de recherche
            results_text = []
            
            # Chercher les éléments de résultats
            result_elements = self.webdriver.find_elements(By.CSS_SELECTOR, "article.result")
            
            for element in result_elements[:5]:  # Limiter aux 5 premiers résultats
                try:
                    # Titre
                    title_elem = element.find_element(By.TAG_NAME, "h3")
                    title = title_elem.text.strip()
                    
                    # URL
                    link_elem = element.find_element(By.TAG_NAME, "a")
                    url = link_elem.get_attribute("href")
                    
                    # Description
                    try:
                        desc_elem = element.find_element(By.CSS_SELECTOR, "p.content")
                        description = desc_elem.text.strip()
                    except:
                        description = "Pas de description"
                    
                    results_text.append(f"Titre: {title}\nURL: {url}\nDescription: {description}\n---")
                    
                except Exception as e:
                    logger.debug(f"Erreur extraction élément: {e}")
                    continue
            
            return "\n\n".join(results_text)
            
        except Exception as e:
            logger.error(f"Erreur extraction texte: {e}")
            return "Erreur lors de l'extraction du texte"
    
    def capture_with_annotations(self, query: str, category: str = "general") -> Optional[Dict[str, Any]]:
        """Capture avec annotations visuelles pour l'IA"""
        
        base_capture = self.capture_search_results(query, category)
        if not base_capture or not base_capture.get('success'):
            return base_capture
        
        try:
            # Ajouter des annotations visuelles
            annotated_image = self._add_visual_annotations(base_capture['screenshot_path'])
            
            if annotated_image:
                base_capture['annotated_image'] = annotated_image
                base_capture['has_annotations'] = True
            
            return base_capture
            
        except Exception as e:
            logger.error(f"Erreur lors de l'annotation: {e}")
            base_capture['annotation_error'] = str(e)
            return base_capture
    
    def _add_visual_annotations(self, screenshot_path: str) -> Optional[str]:
        """Ajoute des annotations visuelles à la capture"""
        try:
            # Ouvrir l'image
            image = Image.open(screenshot_path)
            draw = ImageDraw.Draw(image)
            
            # Tenter de charger une police
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Ajouter un titre informatif
            title_text = "🔍 Résultats de recherche Searx - Analyse IA"
            text_bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            # Fond pour le texte
            draw.rectangle([(10, 10), (text_width + 20, 40)], fill='black', outline='red', width=2)
            draw.text((15, 15), title_text, fill='white', font=font)
            
            # Ajouter un indicateur temporel
            timestamp_text = f"Capture: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            draw.text((15, 50), timestamp_text, fill='red', font=font)
            
            # Sauvegarder l'image annotée
            annotated_path = screenshot_path.replace('.png', '_annotated.png')
            image.save(annotated_path)
            
            # Convertir en base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'annotation: {e}")
            return None
    
    def cleanup_old_screenshots(self, max_age_hours: int = 24):
        """Nettoie les anciennes captures d'écran"""
        try:
            current_time = time.time()
            removed_count = 0
            
            for filename in os.listdir(self.screenshots_dir):
                filepath = os.path.join(self.screenshots_dir, filename)
                
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getctime(filepath)
                    if file_age > (max_age_hours * 3600):
                        os.remove(filepath)
                        removed_count += 1
            
            if removed_count > 0:
                logger.info(f"🧹 Nettoyage: {removed_count} captures supprimées")
                
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
    
    def close(self):
        """Ferme le WebDriver"""
        if self.webdriver:
            try:
                self.webdriver.quit()
                logger.info("WebDriver fermé")
            except Exception as e:
                logger.error(f"Erreur fermeture WebDriver: {e}")

# Instance globale
searx_visual_capture = SearxVisualCapture()

def get_searx_visual_capture() -> SearxVisualCapture:
    """Retourne l'instance de capture visuelle"""
    return searx_visual_capture

if __name__ == "__main__":
    # Test du module
    capture = SearxVisualCapture()
    
    try:
        result = capture.capture_with_annotations("intelligence artificielle", "general")
        
        if result and result.get('success'):
            print(f"✅ Capture réussie: {result['screenshot_path']}")
            print(f"📝 Contexte textuel: {result['page_text_context'][:200]}...")
            
            if result.get('has_annotations'):
                print("🎨 Annotations ajoutées")
        else:
            print(f"❌ Échec de la capture: {result.get('error', 'Erreur inconnue')}")
            
    finally:
        capture.close()
