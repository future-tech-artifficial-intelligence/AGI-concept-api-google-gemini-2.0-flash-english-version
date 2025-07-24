"""
Système de Navigation Web Interactive pour l'API Gemini
Ce module permet à l'API Gemini d'interagir avec les éléments des sites web
(cliquer sur des onglets, boutons, liens, remplir des formulaires, etc.)
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

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('InteractiveWebNavigator')

@dataclass
class InteractiveElement:
    """Représente un élément interactif sur une page web"""
    element_id: str
    element_type: str  # button, link, tab, form, input, etc.
    text: str
    xpath: str
    css_selector: str
    position: Dict[str, int]  # x, y, width, height
    is_visible: bool
    is_clickable: bool
    attributes: Dict[str, str]
    interaction_score: float  # Score d'importance pour l'interaction
    
@dataclass
class InteractionResult:
    """Résultat d'une interaction avec un élément"""
    success: bool
    element: InteractiveElement
    action_performed: str
    new_url: Optional[str]
    page_changed: bool
    error_message: str = ""
    execution_time: float = 0.0
    screenshot_path: Optional[str] = None

@dataclass
class NavigationSession:
    """Session de navigation interactive"""
    session_id: str
    start_url: str
    current_url: str
    visited_urls: List[str]
    interactions_performed: List[InteractionResult]
    discovered_elements: List[InteractiveElement]
    navigation_depth: int
    session_start_time: datetime
    last_interaction_time: datetime
    goals: List[str]  # Objectifs de navigation
    
class InteractiveElementAnalyzer:
    """Analyseur d'éléments interactifs sur une page web"""
    
    def __init__(self):
        # Sélecteurs CSS pour différents types d'éléments interactifs
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
        
        # Mots-clés pour identifier l'importance des éléments
        self.importance_keywords = {
            'high': ['next', 'continue', 'submit', 'login', 'register', 'buy', 'purchase', 'checkout', 'search'],
            'medium': ['more', 'details', 'info', 'about', 'contact', 'help', 'support'],
            'low': ['home', 'back', 'close', 'cancel']
        }
    
    def analyze_page_elements(self, webdriver) -> List[InteractiveElement]:
        """Analyse tous les éléments interactifs d'une page"""
        elements = []
        element_counter = 0
        
        try:
            for element_type, selectors in self.element_selectors.items():
                for selector in selectors:
                    try:
                        web_elements = webdriver.find_elements('css selector', selector)
                        
                        for web_element in web_elements:
                            try:
                                # Vérifier si l'élément est visible et interactif
                                if not web_element.is_displayed():
                                    continue
                                
                                element_counter += 1
                                element_id = f"elem_{element_counter}_{int(time.time() * 1000)}"
                                
                                # Extraire les informations de l'élément
                                text = self._extract_element_text(web_element)
                                xpath = self._get_element_xpath(webdriver, web_element)
                                css_sel = self._generate_css_selector(web_element)
                                position = self._get_element_position(web_element)
                                attributes = self._extract_element_attributes(web_element)
                                is_clickable = self._is_element_clickable(web_element)
                                
                                # Calculer le score d'interaction
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
                                logger.debug(f"Erreur analyse élément individuel: {e}")
                                continue
                                
                    except Exception as e:
                        logger.debug(f"Erreur sélecteur {selector}: {e}")
                        continue
            
            # Trier par score d'interaction (plus important en premier)
            elements.sort(key=lambda x: x.interaction_score, reverse=True)
            
            logger.info(f"🔍 Analysé {len(elements)} éléments interactifs")
            return elements
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse éléments page: {e}")
            return []
    
    def _extract_element_text(self, element) -> str:
        """Extrait le texte d'un élément"""
        try:
            # Essayer différentes méthodes pour extraire le texte
            text = element.text.strip()
            
            if not text:
                # Essayer les attributs
                for attr in ['aria-label', 'title', 'alt', 'value', 'placeholder']:
                    attr_value = element.get_attribute(attr)
                    if attr_value:
                        text = attr_value.strip()
                        break
            
            if not text:
                # Essayer le contenu HTML
                innerHTML = element.get_attribute('innerHTML')
                if innerHTML:
                    # Supprimer les balises HTML basiques
                    import re
                    text = re.sub(r'<[^>]+>', '', innerHTML).strip()
            
            return text[:200]  # Limiter la longueur
            
        except Exception:
            return ""
    
    def _get_element_xpath(self, webdriver, element) -> str:
        """Génère le XPath d'un élément"""
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
        """Génère un sélecteur CSS pour l'élément"""
        try:
            # Si l'élément a un ID unique
            element_id = element.get_attribute('id')
            if element_id:
                return f"#{element_id}"
            
            # Si l'élément a des classes
            classes = element.get_attribute('class')
            if classes:
                class_selector = '.' + '.'.join(classes.split())
                return f"{element.tag_name}{class_selector}"
            
            # Sélecteur par tag et attributs
            tag_name = element.tag_name
            
            # Ajouter des attributs distinctifs
            distinctive_attrs = ['name', 'type', 'role', 'data-tab']
            for attr in distinctive_attrs:
                attr_value = element.get_attribute(attr)
                if attr_value:
                    return f"{tag_name}[{attr}='{attr_value}']"
            
            return tag_name
            
        except Exception:
            return element.tag_name if hasattr(element, 'tag_name') else ""
    
    def _get_element_position(self, element) -> Dict[str, int]:
        """Obtient la position et taille d'un élément"""
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
        """Extrait les attributs importants d'un élément"""
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
        """Détermine si un élément est cliquable"""
        try:
            # Vérifier si l'élément est enabled et visible
            if not element.is_enabled() or not element.is_displayed():
                return False
            
            # Vérifier le tag et les attributs
            tag_name = element.tag_name.lower()
            clickable_tags = ['a', 'button', 'input', 'select']
            
            if tag_name in clickable_tags:
                return True
            
            # Vérifier les attributs de rôle et événements
            role = element.get_attribute('role')
            onclick = element.get_attribute('onclick')
            
            if role in ['button', 'link', 'tab'] or onclick:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_interaction_score(self, text: str, attributes: Dict[str, str], 
                                   element_type: str, position: Dict[str, int]) -> float:
        """Calcule un score d'importance pour l'interaction avec l'élément"""
        score = 0.0
        
        # Score de base selon le type d'élément
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
        
        # Score basé sur le texte
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
        
        # Score basé sur la position (éléments plus hauts sont souvent plus importants)
        if position['y'] < 600:  # Au-dessus du pli
            score += 0.2
        
        # Score basé sur la taille
        area = position['width'] * position['height']
        if area > 10000:  # Gros éléments
            score += 0.1
        
        # Bonus pour certains attributs
        if 'id' in attributes:
            score += 0.1
        if 'aria-label' in attributes:
            score += 0.1
        
        return min(score, 1.0)  # Limiter à 1.0

class InteractiveWebNavigator:
    """Navigateur web interactif principal"""
    
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
        
        # Statistiques
        self.stats = {
            'sessions_created': 0,
            'interactions_performed': 0,
            'successful_interactions': 0,
            'pages_navigated': 0,
            'elements_discovered': 0
        }
    
    def initialize_webdriver(self) -> bool:
        """Initialise le WebDriver pour l'interaction"""
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.webdriver.common.action_chains import ActionChains
            
            # Configuration Chrome optimisée pour l'interaction
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_argument('--window-size=1920,1080')
            
            # Créer le driver
            self.webdriver = webdriver.Chrome(options=chrome_options)
            self.webdriver.set_page_load_timeout(self.config['page_load_timeout'])
            self.webdriver.implicitly_wait(5)
            
            # Modules Selenium pour utilisation
            self.By = By
            self.WebDriverWait = WebDriverWait
            self.EC = EC
            self.ActionChains = ActionChains
            
            logger.info("✅ WebDriver initialisé pour navigation interactive")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation WebDriver: {e}")
            return False
    
    def create_interactive_session(self, session_id: str, start_url: str, 
                                 navigation_goals: List[str] = None) -> NavigationSession:
        """Crée une nouvelle session de navigation interactive"""
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
        
        logger.info(f"🎯 Session interactive créée: {session_id}")
        return session
    
    def navigate_to_url(self, session_id: str, url: str) -> Dict[str, Any]:
        """Navigue vers une URL et analyse les éléments interactifs"""
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Session non trouvée'}
        
        session = self.active_sessions[session_id]
        
        try:
            if not self.webdriver:
                if not self.initialize_webdriver():
                    return {'success': False, 'error': 'Impossible d\'initialiser WebDriver'}
            
            logger.info(f"🌐 Navigation vers: {url}")
            self.webdriver.get(url)
            
            # Attendre le chargement complet
            time.sleep(3)
            
            # Mettre à jour la session
            session.current_url = self.webdriver.current_url
            if url not in session.visited_urls:
                session.visited_urls.append(url)
                self.stats['pages_navigated'] += 1
            
            # Analyser les éléments interactifs
            elements = self.element_analyzer.analyze_page_elements(self.webdriver)
            session.discovered_elements = elements
            self.stats['elements_discovered'] += len(elements)
            
            # Prendre une capture d'écran
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
                    for elem in elements[:20]  # Top 20 éléments
                ],
                'screenshot': screenshot_path
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur navigation: {e}")
            return {'success': False, 'error': str(e)}
    
    def interact_with_element(self, session_id: str, element_id: str, 
                            action: str = 'click') -> InteractionResult:
        """Interagit avec un élément spécifique"""
        if session_id not in self.active_sessions:
            return InteractionResult(
                success=False,
                element=None,
                action_performed=action,
                new_url=None,
                page_changed=False,
                error_message="Session non trouvée"
            )
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        # Trouver l'élément dans la session
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
                error_message="Élément non trouvé"
            )
        
        try:
            current_url = self.webdriver.current_url
            
            # Localiser l'élément sur la page
            web_element = None
            
            # Essayer par CSS selector en premier
            if target_element.css_selector:
                try:
                    web_element = self.webdriver.find_element(self.By.CSS_SELECTOR, target_element.css_selector)
                except:
                    pass
            
            # Essayer par XPath si CSS a échoué
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
                    error_message="Impossible de localiser l'élément sur la page"
                )
            
            # Scroller vers l'élément si nécessaire
            self.webdriver.execute_script("arguments[0].scrollIntoView();", web_element)
            time.sleep(0.5)
            
            # Effectuer l'action
            success = False
            if action == 'click':
                # Essayer le clic normal
                try:
                    web_element.click()
                    success = True
                except:
                    # Essayer le clic JavaScript si le clic normal échoue
                    try:
                        self.webdriver.execute_script("arguments[0].click();", web_element)
                        success = True
                    except Exception as e:
                        logger.error(f"Erreur clic: {e}")
            
            elif action == 'hover':
                try:
                    actions = self.ActionChains(self.webdriver)
                    actions.move_to_element(web_element).perform()
                    success = True
                except Exception as e:
                    logger.error(f"Erreur hover: {e}")
            
            # Attendre les changements potentiels
            time.sleep(2)
            
            # Vérifier si la page a changé
            new_url = self.webdriver.current_url
            page_changed = (new_url != current_url)
            
            # Prendre une capture d'écran après l'interaction
            screenshot_path = None
            if self.config['screenshot_on_interaction']:
                screenshot_path = self._take_screenshot(f"interaction_{session_id}_{element_id}")
            
            # Créer le résultat
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
            
            # Mettre à jour la session
            session.interactions_performed.append(result)
            session.last_interaction_time = datetime.now()
            if page_changed:
                session.current_url = new_url
                if new_url not in session.visited_urls:
                    session.visited_urls.append(new_url)
            
            # Réanalyser les éléments si la page a changé
            if page_changed:
                time.sleep(1)  # Attendre le chargement
                session.discovered_elements = self.element_analyzer.analyze_page_elements(self.webdriver)
            
            # Mettre à jour les statistiques
            self.stats['interactions_performed'] += 1
            if success:
                self.stats['successful_interactions'] += 1
            
            logger.info(f"{'✅' if success else '❌'} Interaction {action} sur {target_element.text[:30]} - "
                       f"Page changée: {page_changed}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Erreur interaction: {e}")
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
        """Retourne un résumé des éléments interactifs découverts"""
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Session non trouvée'}
        
        session = self.active_sessions[session_id]
        
        # Grouper les éléments par type
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
        
        # Identifier les éléments les plus prometteurs
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
        """Génère des suggestions d'interaction basées sur les objectifs"""
        suggestions = []
        
        # Identifier les onglets disponibles
        tab_elements = [elem for elem in session.discovered_elements if elem.element_type == 'tabs']
        if tab_elements:
            suggestions.append({
                'type': 'explore_tabs',
                'description': f"Explorer {len(tab_elements)} onglets disponibles",
                'elements': [elem.element_id for elem in tab_elements[:5]]
            })
        
        # Identifier les formulaires
        form_elements = [elem for elem in session.discovered_elements if elem.element_type == 'forms']
        if form_elements:
            suggestions.append({
                'type': 'interact_forms',
                'description': f"Interagir avec {len(form_elements)} formulaires",
                'elements': [elem.element_id for elem in form_elements[:3]]
            })
        
        # Identifier les liens de navigation importants
        nav_links = [elem for elem in session.discovered_elements 
                    if elem.element_type == 'navigation' and elem.interaction_score > 0.6]
        if nav_links:
            suggestions.append({
                'type': 'follow_navigation',
                'description': f"Suivre {len(nav_links)} liens de navigation importants",
                'elements': [elem.element_id for elem in nav_links[:5]]
            })
        
        return suggestions
    
    def _take_screenshot(self, filename_prefix: str) -> Optional[str]:
        """Prend une capture d'écran"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.png"
            screenshot_path = self.screenshots_dir / filename
            
            self.webdriver.save_screenshot(str(screenshot_path))
            return str(screenshot_path)
            
        except Exception as e:
            logger.error(f"❌ Erreur capture d'écran: {e}")
            return None
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Ferme une session de navigation"""
        if session_id not in self.active_sessions:
            return {'success': False, 'error': 'Session non trouvée'}
        
        session = self.active_sessions[session_id]
        
        # Générer un rapport de session
        session_duration = (datetime.now() - session.session_start_time).total_seconds()
        report = {
            'session_id': session_id,
            'duration_seconds': session_duration,
            'pages_visited': len(session.visited_urls),
            'interactions_performed': len(session.interactions_performed),
            'successful_interactions': sum(1 for r in session.interactions_performed if r.success),
            'elements_discovered': len(session.discovered_elements),
            'visited_urls': session.visited_urls,
            'goals_achieved': []  # À implémenter selon les objectifs
        }
        
        # Supprimer la session
        del self.active_sessions[session_id]
        
        logger.info(f"📊 Session fermée: {session_id} - {report['interactions_performed']} interactions")
        return {'success': True, 'report': report}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du navigateur interactif"""
        return {
            'stats': self.stats,
            'active_sessions': len(self.active_sessions),
            'config': self.config
        }
    
    def close(self):
        """Ferme le navigateur et nettoie les ressources"""
        if self.webdriver:
            try:
                self.webdriver.quit()
                logger.info("🔚 WebDriver fermé")
            except Exception as e:
                logger.error(f"❌ Erreur fermeture WebDriver: {e}")

# Instance globale
_interactive_navigator = None

def get_interactive_navigator() -> InteractiveWebNavigator:
    """Retourne l'instance globale du navigateur interactif"""
    global _interactive_navigator
    if _interactive_navigator is None:
        _interactive_navigator = InteractiveWebNavigator()
    return _interactive_navigator

def initialize_interactive_navigator() -> InteractiveWebNavigator:
    """Initialise le navigateur interactif"""
    navigator = get_interactive_navigator()
    if navigator.initialize_webdriver():
        logger.info("🚀 Navigateur interactif initialisé avec succès")
        return navigator
    else:
        logger.error("❌ Échec de l'initialisation du navigateur interactif")
        return None

# Fonctions utilitaires pour l'API Gemini
def create_interactive_navigation_session(session_id: str, start_url: str, 
                                        goals: List[str] = None) -> Dict[str, Any]:
    """Crée une session de navigation interactive pour Gemini"""
    navigator = get_interactive_navigator()
    session = navigator.create_interactive_session(session_id, start_url, goals)
    result = navigator.navigate_to_url(session_id, start_url)
    return result

def interact_with_web_element(session_id: str, element_id: str, 
                            action: str = 'click') -> Dict[str, Any]:
    """Interagit avec un élément web spécifique"""
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
    """Retourne les éléments interactifs de la page actuelle"""
    navigator = get_interactive_navigator()
    return navigator.get_interactive_elements_summary(session_id)

def close_interactive_session(session_id: str) -> Dict[str, Any]:
    """Ferme une session de navigation interactive"""
    navigator = get_interactive_navigator()
    return navigator.close_session(session_id)
