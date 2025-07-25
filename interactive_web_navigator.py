"""
Interactive Web Navigation System for the artificial intelligence Google Gemini 2.0 Flash  API
This module allows the Google Gemini 2.0 Flash AI API to interact with website elements
(clicking on tabs, buttons, links,
"""

import logging
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('InteractiveWebNavigator')

@dataclass
class InteractiveElement:
    """Represents an interactive element on a web page"""
    element_id: str
    element_type: str  # button, link, tab, form, input, etc.
    text: str
    xpath: str
    css_selector: str
    position: Dict[str, int]  # x, y, width, height
    is_visible: bool
    is_clickable: bool
    attributes: Dict[str, str]
    interaction_score: float  # Importance score for interaction
    
@dataclass
class InteractionResult:
    """Result of an interaction with an element"""
    success: bool
    element: Optional[InteractiveElement] # Added Optional type hint
    action_performed: str
    new_url: Optional[str]
    page_changed: bool
    error_message: str = ""
    execution_time: float = 0.0
    screenshot_path: Optional[str] = None

@dataclass
class NavigationSession:
    """Interactive navigation session"""
    session_id: str
    start_url: str
    current_url: str
    visited_urls: List[str]
    interactions_performed: List[InteractionResult]
    discovered_elements: List[InteractiveElement]
    navigation_depth: int
    session_start_time: datetime
    last_interaction_time: datetime
    goals: List[str]  # Navigation goals
    
class InteractiveElementAnalyzer:
    """Interactive element analyzer on a web page"""
    
    def __init__(self):
        # CSS selectors for different types of interactive elements
        self.element_selectors = {
            'buttons': [
                'button',
                'input[type="button"]',
                'input[type="submit"]',
                'input[type="reset"]',
                '[role="button"]',
                '.btn',
                '.button',
                'a.button'
            ],
            'links': [
                'a[href]',
                '[role="link"]'
            ],
            'tabs': [
                '[role="tab"]',
                '.tab',
                '.nav-tab',
                '.tab-button',
                '[data-tab]',
                'ul.tabs li',
                '.tabbed-navigation a'
            ],
            'forms': [
                'form'
            ],
            'inputs': [
                'input:not([type="hidden"])',
                'textarea',
                'select'
            ],
            'navigation': [
                'nav a',
                '.navigation a',
                '.menu a',
                '.navbar a',
                '[role="navigation"] a'
            ],
            'accordion': [
                '[role="button"][aria-expanded]',
                '.accordion-toggle',
                '.collapse-toggle'
            ],
            'dropdown': [
                '.dropdown-toggle',
                '[data-toggle="dropdown"]',
                'select'
            ]
        }
        
        # Keywords to identify element importance
        self.importance_keywords = {
            'high': ['next', 'continue', 'submit', 'login', 'register', 'buy', 'purchase', 'checkout', 'search'],
            'medium': ['more', 'details', 'info', 'about', 'contact', 'help', 'support'],
            'low': ['home', 'back', 'close', 'cancel']
        }
    
    def analyze_page_elements(self, webdriver) -> List[InteractiveElement]:
        """Analyzes all interactive elements on a page"""
        element_count = 0
        elements = []
        
        try:
            for element_type, selectors in self.element_selectors.items():
                for selector in selectors:
                    try:
                        web_elements = webdriver.find_elements('css selector', selector)
                        
                        for web_element in web_elements:
                            try:
                                # Check if the element is visible and interactive
                                if not web_element.is_displayed():
                                    continue
                                
                                element_count += 1
                                element_id = f"elem_{element_count}_{int(time.time() * 1000)}"
                                
                                # Extract element information
                                text = self._extract_element_text(web_element)
                                xpath = self._get_element_xpath(webdriver, web_element)
                                css_sel = self._generate_css_selector(web_element)
                                position = self._get_element_position(web_element)
                                attributes = self._extract_element_attributes(web_element)
                                is_clickable = self._is_element_clickable(web_element)
                                
                                # Calculate interaction score
                                interaction_score = self._calculate_interaction_score(
                                    text, attributes, element_type, position
                                )
                                
                                interactive_element = InteractiveElement(
                                    element_id=element_id,
                                    element_type=element_type,
                                    text=text,
                                    xpath=xpath,
                                    css_selector=css_sel,
                                    position=position,
                                    is_visible=True,
                                    is_clickable=is_clickable,
                                    attributes=attributes,
                                    interaction_score=interaction_score
                                )
                                
                                elements.append(interactive_element)
                                
                            except Exception as e:
                                logger.debug(f"Error analyzing individual element: {e}")
                                continue
                                
                    except Exception as e:
                        logger.debug(f"Selector error {selector}: {e}")
                        continue
            
            # Sort by interaction score (most important first)
            elements.sort(key=lambda x: x.interaction_score, reverse=True)
            
            logger.info(f"🔍 Analyzed {len(elements)} interactive elements")
            return elements
            
        except Exception as e:
            logger.error(f"❌ Error analyzing page elements: {e}")
            return []
    
    def _extract_element_text(self, element) -> str:
        """Extracts the text of an element"""
        try:
            # Try different methods to extract text
            text = element.text.strip()
            
            if not text:
                # Try attributes
                for attr in ['aria-label', 'title', 'alt', 'value', 'placeholder']:
                    attr_value = element.get_attribute(attr)
                    if attr_value:
                        text = attr_value.strip()
                        break
            
            if not text:
                # Try HTML content
                innerHTML = element.get_attribute('innerHTML')
                if innerHTML:
                    # Remove basic HTML tags
                    import re
                    text = re.sub(r'<[^>]+>', '', innerHTML).strip()
            
            return text[:200]  # Limit length
            
        except Exception:
            return ""
    
    def _get_element_xpath(self, webdriver, element) -> str:
        """Generates the XPath of an element"""
        try:
            return webdriver.execute_script("""
                function getXPath(element) {
                    if (element.id !== '') {
                        return '//*[@id="' + element.id + '"]';
                    }
                    if (element === document.body) {
                        return '/html/body';
                    }
                    
                    var ix = 0;
                    var siblings = element.parentNode.childNodes;
                    for (var i = 0; i < siblings.length; i++) {
                        var sibling = siblings[i];
                        if (sibling === element) {
                            return getXPath(element.parentNode) + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                        }
                        if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                            ix++;
                        }
                    }
                }
                return getXPath(arguments[0]);
            """, element)
        except Exception:
            return ""
    
    def _generate_css_selector(self, element) -> str:
        """Generates a CSS selector for the element"""
        try:
            # If the element has a unique ID
            element_id = element.get_attribute('id')
            if element_id:
                return f"#{element_id}"
            
            # If the element has classes
            classes = element.get_attribute('class')
            if classes:
                class_selector = '.' + '.'.join(classes.split())
                return f"{element.tag_name}{class_selector}"
            
            # Selector by tag and attributes
            tag_name = element.tag_name
            
            # Add distinctive attributes
            distinctive_attrs = ['name', 'type', 'role', 'data-tab']
            for attr in distinctive_attrs:
                attr_value = element.get_attribute(attr)
                if attr_value:
                    return f"{tag_name}[{attr}='{attr_value}']"
            
            return tag_name
            
        except Exception:
            return element.tag_name if hasattr(element, 'tag_name') else ""
    
    def _get_element_position(self, element) -> Dict[str, int]:
        """Gets the position and size of an element"""
        try:
            location = element.location
            size = element.size
            return {
                'x': location['x'],
                'y': location['y'],
                'width': size['width'],
                'height': size['height']
            }
        except Exception:
            return {'x': 0, 'y': 0, 'width': 0, 'height': 0}
    
    def _extract_element_attributes(self, element) -> Dict[str, str]:
        """Extracts important attributes of an element"""
        attributes = {}
        important_attributes = [
            'id', 'class', 'name', 'type', 'role', 'aria-label', 
            'title', 'href', 'onclick', 'data-tab', 'data-toggle'
        ]
        
        try:
            for attr in important_attributes:
                value = element.get_attribute(attr)
                if value:
                    attributes[attr] = value
        except Exception:
            pass
        
        return attributes
    
    def _is_element_clickable(self, element) -> bool:
        """Determines if an element is clickable"""
        try:
            # Check if the element is enabled and visible
            if not element.is_enabled() or not element.is_displayed():
                return False
            
            # Check tag and attributes
            tag_name = element.tag_name.lower()
            clickable_tags = ['a', 'button', 'input', 'select']
            
            if tag_name in clickable_tags:
                return True
            
            # Check role attributes and events
            role = element.get_attribute('role')
            onclick = element.get_attribute('onclick')
            
            if role in ['button', 'link', 'tab'] or onclick:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_interaction_score(self, text: str, attributes: Dict[str, str], 
                                   element_type: str, position: Dict[str, int]) -> float:
        """Calculates an importance score for element interaction"""
        score = 0.0
        
        # Base score by element type
        type_scores = {
            'buttons': 0.8,
            'tabs': 0.7,
            'links': 0.6,
            'navigation': 0.7,
            'forms': 0.5,
            'inputs': 0.4,
            'accordion': 0.6,
            'dropdown': 0.5
        }
        score += type_scores.get(element_type, 0.3)
        
        # Score based on text
        text_lower = text.lower()
        for importance, keywords in self.importance_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    if importance == 'high':
                        score += 0.3
                    elif importance == 'medium':
                        score += 0.2
                    else:
                        score += 0.1
                    break
        
        # Score based on position (higher elements are often more important)
        if position['y'] < 600:  # Above the fold
            score += 0.2
        
        # Score based on size
        area = position['width'] * position['height']
        if area > 10000:  # Large elements
            score += 0.1
        
        # Bonus for certain attributes
        if 'id' in attributes:
            score += 0.1
        if 'aria-label' in attributes:
            score += 0.1
        
        return min(score, 1.0)  # Limit to 1.0

class InteractiveWebNavigator:
    """Main interactive web navigator"""
    
    def __init__(self):
        self.element_analyzer = InteractiveElementAnalyzer()
        self.active_sessions: Dict[str, NavigationSession] = {}
        self.webdriver = None
        self.screenshots_dir = Path("interactive_screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.config = {
            'max_interactions_per_session': 50,
            'interaction_timeout': 30,
            'page_load_timeout': 15,
            'element_wait_timeout': 10,
            'screenshot_on_interaction': True
        }
        
        # Statistics
        self.stats = {
            'sessions_created': 0,
            'interactions_performed': 0,
            'successful_interactions': 0,
            'pages_navigated': 0,
            'elements_discovered': 0
        }
    
    def initialize_webdriver(self) -> bool:
        """Initializes the WebDriver for interaction"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.action_chains import ActionChains
            
            # Chrome configuration optimized for interaction
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--window-size=1920,1080')
            
            # Create the driver
            self.webdriver = webdriver.Chrome(options=chrome_options)
            self.webdriver.set_page_load_timeout(self.config['page_load_timeout'])
            self.webdriver.implicitly_wait(5)
            
            # Selenium modules for use
            self.By = By
            self.WebDriverWait = WebDriverWait
            self.EC = EC
            self.ActionChains = ActionChains
            
            logger.info("✅ WebDriver initialized for interactive navigation")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebDriver initialization error: {e}")
            return False
    
    def create_interactive_session(self, session_id: str, start_url: str, 
                                 navigation_goals: List[str] = None) -> NavigationSession:
        """Creates a new interactive navigation session"""
        session = NavigationSession(
            session_id=session_id,
            start_url=start_url,
            current_url=start_url,
            visited_urls=[],
            interactions_performed=[],
            discovered_elements=[],
            navigation_depth=0,
            session_start_time=datetime.now(),
            last_interaction_time=datetime.now(),
            goals=navigation_goals or []
        )
        
        self.active_sessions[session_id] = session
        self.stats['sessions_created'] += 1
        
        logger.info(f"🎯 Interactive session created: {session_id}")
        return session
    
    def navigate_to_url(self, session_id: str, url: str) -> Dict[str, Any]:
        """Navigates to a URL and analyzes interactive elements"""
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        try:
            if not self.webdriver:
                if not self.initialize_webdriver():
                    return {'success': False, 'error': 'Could not initialize WebDriver'}
            
            logger.info(f"🌐 Navigating to: {url}")
            self.webdriver.get(url)
            
            # Wait for full load
            time.sleep(3)
            
            # Update the session
            session.current_url = self.webdriver.current_url
            if url not in session.visited_urls:
                session.visited_urls.append(url)
                self.stats['pages_navigated'] += 1
            
            # Analyze interactive elements
            elements = self.element_analyzer.analyze_page_elements(self.webdriver)
            session.discovered_elements = elements
            self.stats['elements_discovered'] += len(elements)
            
            # Take a screenshot
            screenshot_path = None
            if self.config['screenshot_on_interaction']:
                screenshot_path = self._take_screenshot(f"navigation_{session_id}")
            
            return {
                'success': True,
                'current_url': session.current_url,
                'elements_found': len(elements),
                'interactive_elements': [
                    {
                        'id': elem.element_id,
                        'type': elem.element_type,
                        'text': elem.text,
                        'score': elem.interaction_score,
                        'clickable': elem.is_clickable
                    }
                    for elem in elements[:20]  # Top 20 elements
                ],
                'screenshot': screenshot_path
            }
            
        except Exception as e:
            logger.error(f"❌ Navigation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def interact_with_element(self, session_id: str, element_id: str, 
                            action: str = 'click') -> InteractionResult:
        """Interacts with a specific element"""
        if session_id not in self.active_sessions:
            return InteractionResult(
                success=False,
                element=None,
                action_performed=action,
                new_url=None,
                page_changed=False,
                error_message="Session not found"
            )
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        # Find the element in the session
        target_element = None
        for elem in session.discovered_elements:
            if elem.element_id == element_id:
                target_element = elem
                break
        
        if not target_element:
            return InteractionResult(
                success=False,
                element=None,
                action_performed=action,
                new_url=None,
                page_changed=False,
                error_message="Element not found"
            )
        
        try:
            current_url = self.webdriver.current_url
            
            # Locate the element on the page
            web_element = None
            
            # Try by CSS selector first
            if target_element.css_selector:
                try:
                    web_element = self.webdriver.find_element(self.By.CSS_SELECTOR, target_element.css_selector)
                except:
                    pass
            
            # Try by XPath if CSS failed
            if not web_element and target_element.xpath:
                try:
                    web_element = self.webdriver.find_element(self.By.XPATH, target_element.xpath)
                except:
                    pass
            
            if not web_element:
                return InteractionResult(
                    success=False,
                    element=target_element,
                    action_performed=action,
                    new_url=None,
                    page_changed=False,
                    error_message="Could not locate element on page"
                )
            
            # Scroll to the element if necessary
            self.webdriver.execute_script("arguments[0].scrollIntoView();", web_element)
            time.sleep(0.5)
            
            # Perform the action
            success = False
            if action == 'click':
                # Try normal click
                try:
                    web_element.click()
                    success = True
                except:
                    # Try JavaScript click if normal click fails
                    try:
                        self.webdriver.execute_script("arguments[0].click();", web_element)
                        success = True
                    except Exception as e:
                        logger.error(f"Click error: {e}")
            
            elif action == 'hover':
                try:
                    actions = self.ActionChains(self.webdriver)
                    actions.move_to_element(web_element).perform()
                    success = True
                except Exception as e:
                    logger.error(f"Hover error: {e}")
            
            # Wait for potential changes
            time.sleep(2)
            
            # Check if page has changed
            new_url = self.webdriver.current_url
            page_changed = (new_url != current_url)
            
            # Take a screenshot after interaction
            screenshot_path = None
            if self.config['screenshot_on_interaction']:
                screenshot_path = self._take_screenshot(f"interaction_{session_id}_{element_id}")
            
            # Create the result
            execution_time = time.time() - start_time
            result = InteractionResult(
                success=success,
                element=target_element,
                action_performed=action,
                new_url=new_url if page_changed else None,
                page_changed=page_changed,
                execution_time=execution_time,
                screenshot_path=screenshot_path
            )
            
            # Update the session
            session.interactions_performed.append(result)
            session.last_interaction_time = datetime.now()
            if page_changed:
                session.current_url = new_url
                if new_url not in session.visited_urls:
                    session.visited_urls.append(new_url)
            
            # Re-analyze elements if page has changed
            if page_changed:
                time.sleep(1)  # Wait for loading
                session.discovered_elements = self.element_analyzer.analyze_page_elements(self.webdriver)
            
            # Update statistics
            self.stats['interactions_performed'] += 1
            if success:
                self.stats['successful_interactions'] += 1
            
            logger.info(f"{'✅' if success else '❌'} Interaction {action} on {target_element.text[:30]} - "
                       f"Page changed: {page_changed}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Interaction error: {e}")
            return InteractionResult(
                success=False,
                element=target_element,
                action_performed=action,
                new_url=None,
                page_changed=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def get_interactive_elements_summary(self, session_id: str) -> Dict[str, Any]:
        """Returns a summary of discovered interactive elements"""
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        # Group elements by type
        elements_by_type = {}
        for element in session.discovered_elements:
            if element.element_type not in elements_by_type:
                elements_by_type[element.element_type] = []
            elements_by_type[element.element_type].append({
                'id': element.element_id,
                'text': element.text,
                'score': element.interaction_score,
                'clickable': element.is_clickable,
                'position': element.position
            })
        
        # Identify the most promising elements
        top_elements = sorted(session.discovered_elements, 
                            key=lambda x: x.interaction_score, reverse=True)[:10]
        
        return {
            'success': True,
            'session_id': session_id,
            'current_url': session.current_url,
            'total_elements': len(session.discovered_elements),
            'elements_by_type': elements_by_type,
            'top_interactive_elements': [
                {
                    'id': elem.element_id,
                    'type': elem.element_type,
                    'text': elem.text,
                    'score': elem.interaction_score,
                    'recommended_action': 'click' if elem.is_clickable else 'analyze'
                }
                for elem in top_elements
            ],
            'interaction_suggestions': self._generate_interaction_suggestions(session)
        }
    
    def _generate_interaction_suggestions(self, session: NavigationSession) -> List[Dict[str, Any]]:
        """Generates interaction suggestions based on goals"""
        suggestions = []
        
        # Identify available tabs
        tab_elements = [elem for elem in session.discovered_elements if elem.element_type == 'tabs']
        if tab_elements:
            suggestions.append({
                'type': 'explore_tabs',
                'description': f"Explore {len(tab_elements)} available tabs",
                'elements': [elem.element_id for elem in tab_elements[:5]]
            })
        
        # Identify forms
        form_elements = [elem for elem in session.discovered_elements if elem.element_type == 'forms']
        if form_elements:
            suggestions.append({
                'type': 'interact_forms',
                'description': f"Interact with {len(form_elements)} forms",
                'elements': [elem.element_id for elem in form_elements[:3]]
            })
        
        # Identify important navigation links
        nav_links = [elem for elem in session.discovered_elements 
                    if elem.element_type == 'navigation' and elem.interaction_score > 0.6]
        if nav_links:
            suggestions.append({
                'type': 'follow_navigation',
                'description': f"Follow {len(nav_links)} important navigation links",
                'elements': [elem.element_id for elem in nav_links[:5]]
            })
        
        return suggestions
    
    def _take_screenshot(self, filename_prefix: str) -> Optional[str]:
        """Takes a screenshot"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.png"
            screenshot_path = self.screenshots_dir / filename
            
            self.webdriver.save_screenshot(str(screenshot_path))
            return str(screenshot_path)
            
        except Exception as e:
            logger.error(f"❌ Screenshot error: {e}")
            return None
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Closes a navigation session"""
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        
        # Generate a session report
        session_duration = (datetime.now() - session.session_start_time).total_seconds()
        report = {
            'session_id': session_id,
            'duration_seconds': session_duration,
            'pages_visited': len(session.visited_urls),
            'interactions_performed': len(session.interactions_performed),
            'successful_interactions': sum(1 for r in session.interactions_performed if r.success),
            'elements_discovered': len(session.discovered_elements),
            'visited_urls': session.visited_urls,
            'goals_achieved': []  # To be implemented according to goals
        }
        
        # Remove the session
        del self.active_sessions[session_id]
        
        logger.info(f"📊 Session closed: {session_id} - {report['interactions_performed']} interactions")
        return {'success': True, 'report': report}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Returns interactive navigator statistics"""
        return {
            'stats': self.stats,
            'active_sessions': len(self.active_sessions),
            'config': self.config
        }
    
    def close(self):
        """Closes the navigator and cleans up resources"""
        if self.webdriver:
            try:
                self.webdriver.quit()
                logger.info("🔚 WebDriver closed")
            except Exception as e:
                logger.error(f"❌ WebDriver closing error: {e}")

# Global instance
_interactive_navigator = None

def get_interactive_navigator() -> InteractiveWebNavigator:
    """Returns the global interactive navigator instance"""
    global _interactive_navigator
    if _interactive_navigator is None:
        _interactive_navigator = InteractiveWebNavigator()
    return _interactive_navigator

def initialize_interactive_navigator() -> InteractiveWebNavigator:
    """Initializes the interactive navigator"""
    navigator = get_interactive_navigator()
    if navigator.initialize_webdriver():
        logger.info("🚀 Interactive navigator initialized successfully")
        return navigator
    else:
        logger.error("❌ Failed to initialize interactive navigator")
        return None

# Utility functions for the Google Gemini 2.0 Flash AI API
def create_interactive_navigation_session(session_id: str, start_url: str, 
                                        goals: List[str] = None) -> Dict[str, Any]:
    """Creates an interactive navigation session for Google Gemini 2.0 Flash AI"""
    navigator = get_interactive_navigator()
    session = navigator.create_interactive_session(session_id, start_url, goals)
    result = navigator.navigate_to_url(session_id, start_url)
    return result

def interact_with_web_element(session_id: str, element_id: str, 
                            action: str = 'click') -> Dict[str, Any]:
    """Interacts with a specific web element"""
    navigator = get_interactive_navigator()
    result = navigator.interact_with_element(session_id, element_id, action)
    
    return {
        'success': result.success,
        'action_performed': result.action_performed,
        'page_changed': result.page_changed,
        'new_url': result.new_url,
        'error_message': result.error_message,
        'execution_time': result.execution_time,
        'element_text': result.element.text if result.element else None
    }

def get_page_interactive_elements(session_id: str) -> Dict[str, Any]:
    """Returns the interactive elements of the current page"""
    navigator = get_interactive_navigator()
    return navigator.get_interactive_elements_summary(session_id)

def close_interactive_session(session_id: str) -> Dict[str, Any]:
    """Closes an interactive navigation session"""
    navigator = get_interactive_navigator()
    return navigator.close_session(session_id)
