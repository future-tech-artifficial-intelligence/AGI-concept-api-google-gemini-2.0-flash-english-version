#!/usr/bin/env python3
"""
Searx Visual Capture Module for AI
Enables the artificial intelligence API GOOGLE GEMINI 2.0 FLASH to visually interpret search results
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
    """Visual capture system for Searx"""
    
    def __init__(self, searx_url: str = "http://localhost:8080"):
        self.searx_url = searx_url.rstrip('/')
        self.screenshots_dir = "searx_screenshots"
        self.webdriver = None
        self.driver_initialized = False
        
        # Create screenshots directory
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
    def _initialize_webdriver(self) -> bool:
        """Initializes the Chrome/Edge WebDriver in headless mode"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.edge.options import Options as EdgeOptions
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            
            # Try Chrome first
            try:
                chrome_options = ChromeOptions()
                chrome_options.add_argument('--headless')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-plugins')
                chrome_options.add_argument('--disable-images')  # Optimization
                
                self.webdriver = webdriver.Chrome(options=chrome_options)
                logger.info("✅ Chrome WebDriver initialized")
                
            except Exception as chrome_error:
                logger.warning(f"Chrome not available: {chrome_error}")
                
                # Try Edge as an alternative
                try:
                    edge_options = EdgeOptions()
                    edge_options.add_argument('--headless')
                    edge_options.add_argument('--no-sandbox')
                    edge_options.add_argument('--disable-dev-shm-usage')
                    edge_options.add_argument('--disable-gpu')
                    edge_options.add_argument('--window-size=1920,1080')
                    
                    self.webdriver = webdriver.Edge(options=edge_options)
                    logger.info("✅ Edge WebDriver initialized")
                    
                except Exception as edge_error:
                    logger.error(f"No WebDriver available. Chrome: {chrome_error}, Edge: {edge_error}")
                    return False
            
            self.driver_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Selenium not installed: {e}")
            logger.error("Install with: pip install selenium")
            return False
        except Exception as e:
            logger.error(f"Error initializing WebDriver: {e}")
            return False
    
    def capture_search_results(self, query: str, category: str = "general") -> Optional[Dict[str, Any]]:
        """Visually captures Searx search results"""
        
        if not self.driver_initialized and not self._initialize_webdriver():
            logger.error("Could not initialize WebDriver")
            return None
        
        try:
            # Searx search URL
            search_url = f"{self.searx_url}/search"
            params = {
                'q': query,
                'category_general': '1' if category == 'general' else '0',
                'category_videos': '1' if category == 'videos' else '0',
                'category_it': '1' if category == 'it' else '0',
                'format': 'html'
            }
            
            # Build the full URL
            param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            full_url = f"{search_url}?{param_string}"
            
            logger.info(f"📸 Visual capture: '{query}' ({category})")
            
            # Navigate to results page
            self.webdriver.get(full_url)
            
            # Wait for results to load
            time.sleep(3)
            
            # Take a screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"searx_search_{timestamp}_{query[:20].replace(' ', '_')}.png"
            filepath = os.path.join(self.screenshots_dir, filename)
            
            # Full page screenshot
            self.webdriver.save_screenshot(filepath)
            
            # Also capture an AI-optimized version
            optimized_image = self._optimize_screenshot_for_ai(filepath)
            
            # Extract visible text for context
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
            
            logger.info(f"✅ Capture successful: {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error during capture: {e}")
            return {
                'query': query,
                'category': category,
                'error': str(e),
                'success': False
            }
    
    def _optimize_screenshot_for_ai(self, screenshot_path: str) -> Optional[str]:
        """Optimizes the screenshot for AI analysis"""
        try:
            # Open the image
            image = Image.open(screenshot_path)
            
            # Resize for AI (optimization)
            max_width = 1024
            if image.width > max_width:
                ratio = max_width / image.width
                new_height = int(image.height * ratio)
                image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
            
            # Enhance contrast for better readability
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            # Convert to base64 for the API
            buffer = io.BytesIO()
            image.save(buffer, format='PNG', optimize=True)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return None
    
    def _extract_visible_text(self) -> str:
        """Extracts visible text from the page for context"""
        try:
            from selenium.webdriver.common.by import By
            
            # Extract text from search results
            results_text = []
            
            # Look for result elements
            result_elements = self.webdriver.find_elements(By.CSS_SELECTOR, "article.result")
            
            for element in result_elements[:5]:  # Limit to the first 5 results
                try:
                    # Title
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
                        description = "No description"
                    
                    results_text.append(f"Title: {title}\nURL: {url}\nDescription: {description}\n---")
                    
                except Exception as e:
                    logger.debug(f"Error extracting element: {e}")
                    continue
            
            return "\n\n".join(results_text)
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return "Error during text extraction"
    
    def capture_with_annotations(self, query: str, category: str = "general") -> Optional[Dict[str, Any]]:
        """Captures with visual annotations for AI"""
        
        base_capture = self.capture_search_results(query, category)
        if not base_capture or not base_capture.get('success'):
            return base_capture
        
        try:
            # Add visual annotations
            annotated_image = self._add_visual_annotations(base_capture['screenshot_path'])
            
            if annotated_image:
                base_capture['annotated_image'] = annotated_image
                base_capture['has_annotations'] = True
            
            return base_capture
            
        except Exception as e:
            logger.error(f"Error during annotation: {e}")
            base_capture['annotation_error'] = str(e)
            return base_capture
    
    def _add_visual_annotations(self, screenshot_path: str) -> Optional[str]:
        """Adds visual annotations to the capture"""
        try:
            # Open the image
            image = Image.open(screenshot_path)
            draw = ImageDraw.Draw(image)
            
            # Attempt to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Add an informative title
            title_text = "🔍 Searx Search Results - AI Analysis"
            text_bbox = draw.textbbox((0, 0), title_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            
            # Background for text
            draw.rectangle([(10, 10), (text_width + 20, 40)], fill='black', outline='red', width=2)
            draw.text((15, 15), title_text, fill='white', font=font)
            
            # Add a time indicator
            timestamp_text = f"Capture: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            draw.text((15, 50), timestamp_text, fill='red', font=font)
            
            # Save the annotated image
            annotated_path = screenshot_path.replace('.png', '_annotated.png')
            image.save(annotated_path)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
            
        except Exception as e:
            logger.error(f"Error during annotation: {e}")
            return None
    
    def cleanup_old_screenshots(self, max_age_hours: int = 24):
        """Cleans up old screenshots"""
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
                logger.info(f"🧹 Cleanup: {removed_count} captures removed")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def close(self):
        """Closes the WebDriver"""
        if self.webdriver:
            try:
                self.webdriver.quit()
                logger.info("WebDriver closed")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {e}")

# Global instance
searx_visual_capture = SearxVisualCapture()

def get_searx_visual_capture() -> SearxVisualCapture:
    """Returns the visual capture instance"""
    return searx_visual_capture

if __name__ == "__main__":
    # Module test
    capture = SearxVisualCapture()
    
    try:
        result = capture.capture_with_annotations("intelligence artificielle", "general")
        
        if result and result.get('success'):
            print(f"✅ Capture successful: {result['screenshot_path']}")
            print(f"📝 Textual context: {result['page_text_context'][:200]}...")
            
            if result.get('has_annotations'):
                print("🎨 Annotations added")
        else:
            print(f"❌ Capture failed: {result.get('error', 'Unknown error')}")
            
    finally:
        capture.close()
