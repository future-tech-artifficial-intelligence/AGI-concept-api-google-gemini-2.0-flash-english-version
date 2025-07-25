"""
Intelligent Visual Capture System for Websites
Integrated with Google Gemini 2.0 Flash AI Vision for real-time analysis
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

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('IntelligentWebCapture')

class IntelligentWebCapture:
    """Intelligent visual capture system for websites"""
    
    def __init__(self, screenshots_dir: str = "intelligent_screenshots"):
        """
        Initializes the intelligent capture system
        
        Args:
            screenshots_dir: Directory to save captures
        """
        self.screenshots_dir = Path(screenshots_dir)
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Organized directories
        self.raw_screenshots_dir = self.screenshots_dir / "raw"
        self.optimized_screenshots_dir = self.screenshots_dir / "optimized"
        self.analysis_cache_dir = self.screenshots_dir / "analysis_cache"
        
        for dir_path in [self.raw_screenshots_dir, self.optimized_screenshots_dir, self.analysis_cache_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.webdriver = None
        self.driver_initialized = False
        
        # Capture configuration
        self.capture_config = {
            'window_size': (1920, 1080),
            'mobile_size': (375, 667),
            'tablet_size': (768, 1024),
            'wait_time': 3,  # Wait time for loading
            'scroll_pause': 1,  # Pause between scrolls
            'element_highlight': True  # Highlight important elements
        }
        
        # Statistics
        self.stats = {
            'captures_taken': 0,
            'successful_optimizations': 0,
            'failed_captures': 0,
            'total_processing_time': 0
        }
        
        logger.info("🎯 Intelligent Visual Capture System initialized")
    
    def _initialize_webdriver(self) -> bool:
        """Initializes the WebDriver with AI-optimized configuration"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.chrome.service import Service as ChromeService
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.action_chains import ActionChains
            
            # Chrome configuration optimized for AI capture
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--headless=new')  # New headless mode
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument(f'--window-size={self.capture_config["window_size"][0]},{self.capture_config["window_size"][1]}')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-images')  # Disable image loading for faster performance
            chrome_options.add_argument('--disable-javascript')  # Optional: disable JS for static captures
            chrome_options.add_argument('--force-device-scale-factor=1')  # Fixed scale
            chrome_options.add_argument('--high-dpi-support=1')
            chrome_options.add_argument('--disable-background-networking')
            chrome_options.add_argument('--disable-default-apps')
            chrome_options.add_argument('--disable-features=TranslateUI')
            
            # Preferences for optimization
            chrome_prefs = {
                'profile.default_content_setting_values': {
                    'notifications': 2,  # Block notifications
                    'media_stream': 2,   # Block media
                },
                'profile.default_content_settings.popups': 0,
                'profile.managed_default_content_settings.images': 2  # Block images
            }
            chrome_options.add_experimental_option('prefs', chrome_prefs)
            
            # Create the driver
            self.webdriver = webdriver.Chrome(options=chrome_options)
            self.webdriver.set_page_load_timeout(30)
            
            # Import Selenium modules for use
            self.By = By
            self.WebDriverWait = WebDriverWait
            self.EC = EC
            self.ActionChains = ActionChains
            
            self.driver_initialized = True
            logger.info("✅ Chrome WebDriver initialized for AI capture")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebDriver initialization error: {e}")
            return False
    
    def capture_website_intelligent(self, 
                                  url: str, 
                                  capture_type: str = "full_page",
                                  viewport: str = "desktop",
                                  analyze_elements: bool = True) -> Dict[str, Any]:
        """
        Intelligent website capture with AI optimization
        
        Args:
            url: URL to capture
            capture_type: Capture type (full_page, visible_area, element_focused)
            viewport: Screen size (desktop, mobile, tablet)
            analyze_elements: Analyze elements during capture
            
        Returns:
            Capture information and file paths
        """
        start_time = datetime.now()
        
        try:
            if not self.driver_initialized and not self._initialize_webdriver():
                return {
                    'success': False,
                    'error': 'Could not initialize WebDriver',
                    'captures': []
                }
            
            # Viewport configuration
            viewport_sizes = {
                'desktop': self.capture_config['window_size'],
                'mobile': self.capture_config['mobile_size'], 
                'tablet': self.capture_config['tablet_size']
            }
            
            if viewport in viewport_sizes:
                size = viewport_sizes[viewport]
                self.webdriver.set_window_size(size[0], size[1])
                logger.info(f"📱 Viewport configured: {viewport} ({size[0]}x{size[1]})")
            
            # Navigate to the URL
            logger.info(f"🌐 Navigating to: {url}")
            self.webdriver.get(url)
            
            # Wait for loading
            time.sleep(self.capture_config['wait_time'])
            
            # Generate unique filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"capture_{viewport}_{url_hash}_{timestamp}"
            
            captures = []
            
            if capture_type == "full_page":
                # Intelligent full page capture with adaptive scrolling
                captures.extend(self._capture_full_page_intelligent(base_filename, analyze_elements))
                
            elif capture_type == "visible_area":
                # Capture of visible area only
                captures.extend(self._capture_visible_area(base_filename, analyze_elements))
                
            elif capture_type == "element_focused":
                # Capture focused on important elements
                captures.extend(self._capture_important_elements(base_filename))
            
            # Optimize all captures for AI
            optimized_captures = []
            for capture in captures:
                optimized = self._optimize_for_ai_analysis(capture)
                if optimized:
                    optimized_captures.append(optimized)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.stats['captures_taken'] += len(captures)
            self.stats['successful_optimizations'] += len(optimized_captures)
            self.stats['total_processing_time'] += processing_time
            
            logger.info(f"✅ Intelligent capture successful: {len(optimized_captures)} images in {processing_time:.2f}s")
            
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
            error_msg = f"Intelligent capture error {url}: {str(e)}"
            logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'error': error_msg,
                'captures': []
            }
    
    def _capture_full_page_intelligent(self, base_filename: str, analyze_elements: bool) -> List[Dict[str, Any]]:
        """Intelligent full page capture with adaptive scrolling"""
        captures = []
        
        try:
            # Get total page height
            total_height = self.webdriver.execute_script("return document.body.scrollHeight")
            viewport_height = self.webdriver.execute_script("return window.innerHeight")
            
            logger.info(f"📏 Page: {total_height}px, Viewport: {viewport_height}px")
            
            # Calculate number of captures needed
            scroll_positions = []
            current_position = 0
            
            while current_position < total_height:
                scroll_positions.append(current_position)
                current_position += viewport_height * 0.8  # 20% overlap
            
            # Ensure bottom of page is captured
            if scroll_positions[-1] < total_height - viewport_height:
                scroll_positions.append(total_height - viewport_height)
            
            # Take captures at each position
            for i, position in enumerate(scroll_positions):
                # Scroll to position
                self.webdriver.execute_script(f"window.scrollTo(0, {position});")
                time.sleep(self.capture_config['scroll_pause'])
                
                # Filename for this section
                section_filename = f"{base_filename}_section_{i+1:02d}.png"
                raw_path = self.raw_screenshots_dir / section_filename
                
                # Take screenshot
                self.webdriver.save_screenshot(str(raw_path))
                
                # Analyze elements if requested
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
                
                logger.info(f"📸 Section {i+1}/{len(scroll_positions)} captured")
            
            # Scroll back to top of page
            self.webdriver.execute_script("window.scrollTo(0, 0);")
            
            return captures
            
        except Exception as e:
            logger.error(f"❌ Full page capture error: {e}")
            return []
    
    def _capture_visible_area(self, base_filename: str, analyze_elements: bool) -> List[Dict[str, Any]]:
        """Capture of current visible area"""
        try:
            filename = f"{base_filename}_visible.png"
            raw_path = self.raw_screenshots_dir / filename
            
            # Take screenshot
            self.webdriver.save_screenshot(str(raw_path))
            
            # Analyze elements
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
            logger.error(f"❌ Visible area capture error: {e}")
            return []
    
    def _capture_important_elements(self, base_filename: str) -> List[Dict[str, Any]]:
        """Capture focused on important elements (headers, forms, CTAs, etc.)"""
        captures = []
        
        try:
            # Important element selectors
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
                        # Scroll to the first found element
                        self.webdriver.execute_script("arguments[0].scrollIntoView({block: 'center'});", elements[0])
                        time.sleep(1)
                        
                        # Take screenshot
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
                        
                        logger.info(f"🎯 Element captured: {selector} ({len(elements)} found)")
                
                except Exception as e:
                    logger.warning(f"⚠️ Could not capture {selector}: {e}")
                    continue
            
            return captures
            
        except Exception as e:
            logger.error(f"❌ Important elements capture error: {e}")
            return []
    
    def _analyze_visible_elements(self) -> Dict[str, Any]:
        """Analyzes visible elements on the current page"""
        try:
            # Count different element types
            elements_count = {
                'buttons': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'button, .btn, input[type="submit"], input[type="button"]')),
                'links': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'a[href]')),
                'forms': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'form')),
                'inputs': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'input, textarea, select')),
                'images': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'img')),
                'headings': len(self.webdriver.find_elements(self.By.CSS_SELECTOR, 'h1, h2, h3, h4, h5, h6'))
            }
            
            # Get page title
            page_title = self.webdriver.title
            
            # Get current URL
            current_url = self.webdriver.current_url
            
            return {
                'page_title': page_title,
                'current_url': current_url,
                'elements_count': elements_count,
                'viewport_size': self.webdriver.get_window_size(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Elements analysis error: {e}")
            return {}
    
    def _optimize_for_ai_analysis(self, capture_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optimizes a capture for AI analysis"""
        try:
            raw_path = Path(capture_info['raw_path'])
            if not raw_path.exists():
                return None
            
            # Open the image
            with Image.open(raw_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Enhance quality for AI
                # 1. Enhance contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.2)
                
                # 2. Enhance sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.1)
                
                # 3. Resize if too large (optimization for Google Gemini 2.0 Flash AI)
                max_size = (1920, 1080)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Save optimized version
                optimized_filename = f"opt_{raw_path.stem}.jpg"
                optimized_path = self.optimized_screenshots_dir / optimized_filename
                
                img.save(optimized_path, 'JPEG', quality=90, optimize=True)
                
                # Calculate metadata
                file_size_raw = raw_path.stat().st_size
                file_size_optimized = optimized_path.stat().st_size
                compression_ratio = file_size_raw / file_size_optimized if file_size_optimized > 0 else 1
                
                # Updating capture information
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
                
                logger.info(f"✨ Image optimized: {compression_ratio:.1f}x compression")
                return optimized_info
                
        except Exception as e:
            logger.error(f"❌ Image optimization error: {e}")
            return None
    
    def close(self):
        """Closes the WebDriver"""
        if self.webdriver:
            try:
                self.webdriver.quit()
                self.driver_initialized = False
                logger.info("🔚 WebDriver closed")
            except Exception as e:
                logger.error(f"❌ WebDriver closing error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns capture system statistics"""
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
        """Destructor to clean up resources"""
        self.close()

# Global instance
intelligent_capture = None

def initialize_intelligent_capture(screenshots_dir: str = "intelligent_screenshots") -> IntelligentWebCapture:
    """
    Initializes the global intelligent capture system
    
    Args:
        screenshots_dir: Directory for captures
        
    Returns:
        Capture system instance
    """
    global intelligent_capture
    
    if intelligent_capture is None:
        intelligent_capture = IntelligentWebCapture(screenshots_dir)
        logger.info("🚀 Intelligent Capture System initialized globally")
    
    return intelligent_capture

def get_intelligent_capture() -> Optional[IntelligentWebCapture]:
    """
    Returns the global capture system instance
    
    Returns:
        Instance or None if not initialized
    """
    global intelligent_capture
    return intelligent_capture

if __name__ == "__main__":
    # System test
    capture_system = initialize_intelligent_capture()
    print("🧪 Intelligent Capture System ready for tests")
    print(f"📊 Statistics: {capture_system.get_statistics()}")
