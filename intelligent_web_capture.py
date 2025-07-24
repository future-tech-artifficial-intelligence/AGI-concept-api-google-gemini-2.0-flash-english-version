"""
Système de Capture Visuelle Intelligent pour Sites Web
Intégré avec Gemini Vision pour l'analyse en temps réel
"""

import os
import time
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('IntelligentWebCapture')

class IntelligentWebCapture:
    """Système de capture visuelle intelligent pour sites web"""
    
    def __init__(self, screenshots_dir: str = "intelligent_screenshots"):
        """
        Initialise le système de capture intelligent
        
        Args:
            screenshots_dir: Répertoire pour sauvegarder les captures
        """
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Répertoires organisés
        self.raw_screenshots_dir = self.screenshots_dir / "raw"
        self.optimized_screenshots_dir = self.screenshots_dir / "optimized"
        self.analysis_cache_dir = self.screenshots_dir / "analysis_cache"
        
        for dir_path in [self.raw_screenshots_dir, self.optimized_screenshots_dir, self.analysis_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.webdriver = None
        self.driver_initialized = False
        
        # Configuration de capture
        self.capture_config = {
            'window_size': (1920, 1080),
            'mobile_size': (375, 667),
            'tablet_size': (768, 1024),
            'wait_time': 3,  # Temps d'attente pour le chargement
            'scroll_pause': 1,  # Pause entre les scrolls
            'element_highlight': True  # Surligner les éléments importants
        }
        
        # Statistiques
        self.stats = {
            'captures_taken': 0,
            'successful_optimizations': 0,
            'failed_captures': 0,
            'total_processing_time': 0
        }
        
        logger.info("🎯 Système de Capture Visuelle Intelligent initialisé")
    
    def _initialize_webdriver(self) -> bool:
        """Initialise le WebDriver avec configuration optimisée pour l'IA"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.action_chains import ActionChains
            
            # Configuration Chrome optimisée pour capture IA
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--headless=new')  # Nouveau mode headless
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--window-size={self.capture_config["window_size"][0]},{self.capture_config["window_size"][1]}')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-images')  # Désactiver le chargement d'images pour plus de rapidité
            chrome_options.add_argument('--disable-javascript')  # Optionnel: désactiver JS pour captures statiques
            chrome_options.add_argument('--force-device-scale-factor=1')  # Échelle fixe
            chrome_options.add_argument('--high-dpi-support=1')
            chrome_options.add_argument('--disable-background-networking')
            chrome_options.add_argument('--disable-default-apps')
            chrome_options.add_argument('--disable-features=TranslateUI')
            
            # Préférences pour optimiser
            chrome_prefs = {
                'profile.default_content_setting_values': {
                    'notifications': 2,  # Bloquer notifications
                    'media_stream': 2,   # Bloquer média
                },
                'profile.default_content_settings.popups': 0,
                'profile.managed_default_content_settings.images': 2  # Bloquer images
            }
            chrome_options.add_experimental_option('prefs', chrome_prefs)
            
            # Créer le driver
            self.webdriver = webdriver.Chrome(options=chrome_options)
            self.webdriver.set_page_load_timeout(30)
            
            # Importer les modules Selenium pour utilisation
            self.By = By
            self.WebDriverWait = WebDriverWait
            self.EC = EC
            self.ActionChains = ActionChains
            
            self.driver_initialized = True
            logger.info("✅ WebDriver Chrome initialisé pour capture IA")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation WebDriver: {e}")
            return False
    
    def capture_website_intelligent(self, 
                                  url: str, 
                                  capture_type: str = "full_page",
                                  viewport: str = "desktop",
                                  analyze_elements: bool = True) -> Dict[str, Any]:
        """
        Capture intelligente d'un site web avec optimisation pour l'IA
        
        Args:
            url: URL à capturer
            capture_type: Type de capture (full_page, visible_area, element_focused)
            viewport: Taille d'écran (desktop, mobile, tablet)
            analyze_elements: Analyser les éléments pendant la capture
            
        Returns:
            Informations sur la capture et chemins des fichiers
        """
        start_time = datetime.now()
        
        try:
            if not self.driver_initialized and not self._initialize_webdriver():
                return {
                    'success': False,
                    'error': 'Impossible d\'initialiser WebDriver',
                    'captures': []
                }
            
            # Configuration du viewport
            viewport_sizes = {
                'desktop': self.capture_config['window_size'],
                'mobile': self.capture_config['mobile_size'], 
                'tablet': self.capture_config['tablet_size']
            }
            
            if viewport in viewport_sizes:
                size = viewport_sizes[viewport]
                self.webdriver.set_window_size(size[0], size[1])
                logger.info(f"📱 Viewport configuré: {viewport} ({size[0]}x{size[1]})")
            
            # Naviguer vers l'URL
            logger.info(f"🌐 Navigation vers: {url}")
            self.webdriver.get(url)
            
            # Attendre le chargement
            time.sleep(self.capture_config['wait_time'])
            
            # Générer nom de fichier unique
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"capture_{viewport}_{url_hash}_{timestamp}"
            
            captures = []
            
            if capture_type == "full_page":
                # Capture de la page complète avec scrolling intelligent
                captures.extend(self._capture_full_page_intelligent(base_filename, analyze_elements))
                
            elif capture_type == "visible_area":
                # Capture de la zone visible uniquement
                captures.extend(self._capture_visible_area(base_filename, analyze_elements))
                
            elif capture_type == "element_focused":
                # Capture focalisée sur les éléments importants
                captures.extend(self._capture_important_elements(base_filename))
            
            # Optimiser toutes les captures pour l'IA
            optimized_captures = []
            for capture in captures:
                optimized = self._optimize_for_ai_analysis(capture)
                if optimized:
                    optimized_captures.append(optimized)
            
            # Calculer le temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Mettre à jour les statistiques
            self.stats['captures_taken'] += len(captures)
            self.stats['successful_optimizations'] += len(optimized_captures)
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"✅ Capture intelligente réussie: {len(optimized_captures)} images en {processing_time:.2f}s")
            
            return {
                'success': True,
                'url': url,
                'capture_type': capture_type,
                'viewport': viewport,
                'captures': optimized_captures,
                'processing_time': processing_time,
                'timestamp': start_time.isoformat(),
                'total_captures': len(optimized_captures)
            }
            
        except Exception as e:
            self.stats['failed_captures'] += 1
            error_msg = f"Erreur capture intelligente {url}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'captures': []
            }
    
    def _capture_full_page_intelligent(self, base_filename: str, analyze_elements: bool) -> List[Dict[str, Any]]:
        """Capture intelligente de la page complète avec scrolling adaptatif"""
        captures = []
        
        try:
            # Obtenir la hauteur totale de la page
            total_height = self.webdriver.execute_script("return document.body.scrollHeight")
            viewport_height = self.webdriver.execute_script("return window.innerHeight")
            
            logger.info(f"📏 Page: {total_height}px, Viewport: {viewport_height}px")
            
            # Calculer le nombre de captures nécessaires
            scroll_positions = []
            current_position = 0
            
            while current_position < total_height:
                scroll_positions.append(current_position)
                current_position += viewport_height * 0.8  # 20% de chevauchement
            
            # S'assurer de capturer le bas de la page
            if scroll_positions[-1] < total_height - viewport_height:
                scroll_positions.append(total_height - viewport_height)
            
            # Prendre les captures à chaque position
            for i, position in enumerate(scroll_positions):
                # Scroller à la position
                self.webdriver.execute_script(f"window.scrollTo(0, {position});")
                time.sleep(self.capture_config['scroll_pause'])
                
                # Nom de fichier pour cette section
                section_filename = f"{base_filename}_section_{i+1:02d}.png"
                raw_path = self.raw_screenshots_dir / section_filename
                
                # Prendre la capture
                self.webdriver.save_screenshot(str(raw_path))
                
                # Analyser les éléments si demandé
                elements_info = {}
                if analyze_elements:
                    elements_info = self._analyze_visible_elements()
                
                captures.append({
                    'section': i + 1,
                    'total_sections': len(scroll_positions),
                    'raw_path': str(raw_path),
                    'scroll_position': position,
                    'elements_info': elements_info,
                    'filename': section_filename
                })
                
                logger.info(f"📸 Section {i+1}/{len(scroll_positions)} capturée")
            
            # Revenir en haut de la page
            self.webdriver.execute_script("window.scrollTo(0, 0);")
            
            return captures
            
        except Exception as e:
            logger.error(f"❌ Erreur capture page complète: {e}")
            return []
    
    def _capture_visible_area(self, base_filename: str, analyze_elements: bool) -> List[Dict[str, Any]]:
        """Capture de la zone visible actuelle"""
        try:
            filename = f"{base_filename}_visible.png"
            raw_path = self.raw_screenshots_dir / filename
            
            # Prendre la capture
            self.webdriver.save_screenshot(str(raw_path))
            
            # Analyser les éléments
            elements_info = {}
            if analyze_elements:
                elements_info = self._analyze_visible_elements()
            
            return [{
                'section': 1,
                'total_sections': 1,
                'raw_path': str(raw_path),
                'scroll_position': 0,
                'elements_info': elements_info,
                'filename': filename
            }]
            
        except Exception as e:
            logger.error(f"❌ Erreur capture zone visible: {e}")
            return []
    
    def _capture_important_elements(self, base_filename: str) -> List[Dict[str, Any]]:
        """Capture focalisée sur les éléments importants (headers, forms, CTA, etc.)"""
        captures = []
        
        try:
            # Sélecteurs d'éléments importants
            important_selectors = [
                'header, .header, #header',
                'nav, .nav, .navigation, #navigation',
                'main, .main, #main, .content, #content',
                'form, .form',
                '.cta, .call-to-action, .btn-primary, .button-primary',
                'footer, .footer, #footer'
            ]
            
            for i, selector in enumerate(important_selectors):
                try:
                    elements = self.webdriver.find_elements(self.By.CSS_SELECTOR, selector)
                    
                    if elements:
                        # Scroller vers le premier élément trouvé
                        self.webdriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elements[0])
                        time.sleep(1)
                        
                        # Prendre la capture
                        filename = f"{base_filename}_element_{i+1:02d}.png"
                        raw_path = self.raw_screenshots_dir / filename
                        
                        self.webdriver.save_screenshot(str(raw_path))
                        
                        captures.append({
                            'section': i + 1,
                            'total_sections': len(important_selectors),
                            'raw_path': str(raw_path),
                            'element_type': selector,
                            'elements_found': len(elements),
                            'filename': filename
                        })
                        
                        logger.info(f"🎯 Élément capturé: {selector} ({len(elements)} trouvés)")
                
                except Exception as e:
                    logger.warning(f"⚠️ Impossible de capturer {selector}: {e}")
                    continue
            
            return captures
            
        except Exception as e:
            logger.error(f"❌ Erreur capture éléments importants: {e}")
            return []
    
    def _analyze_visible_elements(self) -> Dict[str, Any]:
        """Analyse les éléments visibles sur la page actuelle"""
        try:
            # Compter différents types d'éléments
            elements_count = {
                'buttons': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'button, .btn, input[type="submit"], input[type="button"]')),
                'links': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'a[href]')),
                'forms': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'form')),
                'inputs': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'input, textarea, select')),
                'images': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'img')),
                'headings': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'h1, h2, h3, h4, h5, h6'))
            }
            
            # Obtenir le titre de la page
            page_title = self.webdriver.title
            
            # Obtenir l'URL actuelle
            current_url = self.webdriver.current_url
            
            return {
                'page_title': page_title,
                'current_url': current_url,
                'elements_count': elements_count,
                'viewport_size': self.webdriver.get_window_size(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse éléments: {e}")
            return {}
    
    def _optimize_for_ai_analysis(self, capture_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimise une capture pour l'analyse par l'IA"""
        try:
            raw_path = Path(capture_info['raw_path'])
            if not raw_path.exists():
                return None
            
            # Ouvrir l'image
            with Image.open(raw_path) as img:
                # Convertir en RGB si nécessaire
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Améliorer la qualité pour l'IA
                # 1. Améliorer le contraste
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                
                # 2. Améliorer la netteté
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
                
                # 3. Redimensionner si trop grande (optimisation pour Gemini)
                max_size = (1920, 1080)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Sauvegarder la version optimisée
                optimized_filename = f"opt_{raw_path.stem}.jpg"
                optimized_path = self.optimized_screenshots_dir / optimized_filename
                
                img.save(optimized_path, 'JPEG', quality=90, optimize=True)
                
                # Calculer les métadonnées
                file_size_raw = raw_path.stat().st_size
                file_size_optimized = optimized_path.stat().st_size
                compression_ratio = file_size_raw / file_size_optimized if file_size_optimized > 0 else 1
                
                # Mise à jour des informations de capture
                optimized_info = capture_info.copy()
                optimized_info.update({
                    'optimized_path': str(optimized_path),
                    'optimized_filename': optimized_filename,
                    'optimization': {
                        'file_size_raw': file_size_raw,
                        'file_size_optimized': file_size_optimized,
                        'compression_ratio': round(compression_ratio, 2),
                        'image_size': img.size,
                        'enhancements': ['contrast', 'sharpness', 'resize']
                    }
                })
                
                logger.info(f"✨ Image optimisée: {compression_ratio:.1f}x compression")
                return optimized_info
                
        except Exception as e:
            logger.error(f"❌ Erreur optimisation image: {e}")
            return None
    
    def close(self):
        """Ferme le WebDriver"""
        if self.webdriver:
            try:
                self.webdriver.quit()
                self.driver_initialized = False
                logger.info("🔚 WebDriver fermé")
            except Exception as e:
                logger.error(f"❌ Erreur fermeture WebDriver: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du système de capture"""
        avg_time = self.stats['total_processing_time'] / max(self.stats['captures_taken'], 1)
        
        return {
            'captures_taken': self.stats['captures_taken'],
            'successful_optimizations': self.stats['successful_optimizations'],
            'failed_captures': self.stats['failed_captures'],
            'success_rate': round(self.stats['successful_optimizations'] / max(self.stats['captures_taken'], 1) * 100, 2),
            'average_processing_time': round(avg_time, 2),
            'total_processing_time': round(self.stats['total_processing_time'], 2)
        }
    
    def __del__(self):
        """Destructeur pour nettoyer les ressources"""
        self.close()

# Instance globale
intelligent_capture = None

def initialize_intelligent_capture(screenshots_dir: str = "intelligent_screenshots") -> IntelligentWebCapture:
    """
    Initialise le système de capture intelligent global
    
    Args:
        screenshots_dir: Répertoire pour les captures
        
    Returns:
        Instance du système de capture
    """
    global intelligent_capture
    
    if intelligent_capture is None:
        intelligent_capture = IntelligentWebCapture(screenshots_dir)
        logger.info("🚀 Système de Capture Intelligent initialisé globalement")
    
    return intelligent_capture

def get_intelligent_capture() -> Optional[IntelligentWebCapture]:
    """
    Retourne l'instance globale du système de capture
    
    Returns:
        Instance ou None si non initialisé
    """
    global intelligent_capture
    return intelligent_capture

if __name__ == "__main__":
    # Test du système
    capture_system = initialize_intelligent_capture()
    print("🧪 Système de Capture Intelligent prêt pour les tests")
    print(f"📊 Statistiques: {capture_system.get_statistics()}")
