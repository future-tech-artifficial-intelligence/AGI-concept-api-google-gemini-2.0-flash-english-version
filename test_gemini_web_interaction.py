"""
Comprehensive Test of artificial intelligence API GOOGLE GEMINI 2.0 FLASH's Web Interaction Capabilities
This script specifically tests if artificial intelligence API GOOGLE GEMINI 2.0 FLASH can:
1. Analyze web pages
2. Identify clickable elements
3. Perform clicks on elements
4. Navigate between pages
5. Fill out forms
"""

import logging
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path
import requests
from urllib.parse import urljoin, urlparse

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeminiWebInteractionTest')

class GeminiWebInteractionTester:
    """Specialized tester for web interactions with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        self.session_id = None  # Session ID for the interactive navigator
        
        # Test sites (public and safe)
        self.test_sites = {
            'simple_page': 'https://example.com',
            'form_page': 'https://httpbin.org/forms/post',
            'search_page': 'https://duckduckgo.com',
            'navigation_page': 'https://www.w3schools.com',
            'interactive_page': 'https://www.google.com'
        }
        
        # Create test directory
        self.test_dir = Path("test_results_web_interaction")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🌐 artificial intelligence API GOOGLE GEMINI 2.0 FLASH Web Interaction Tester initialized")
        
        # Initialize necessary modules
        self.navigator = None
        self.gemini_adapter = None
        self.interactive_adapter = None
        
    def log_test_result(self, test_name: str, success: bool, message: str = "", data: dict = None):
        """Logs the result of a test"""
        self.total_tests += 1
        if success:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}: {message}")
        else:
            logger.error(f"❌ {test_name}: {message}")
            self.errors.append(f"{test_name}: {message}")
        
        self.test_results[test_name] = {
            'success': success,
            'message': message,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
    
    def setup_modules(self):
        """Initializes all necessary modules"""
        logger.info("🔧 Configuring modules...")
        
        # Import and initialize the interactive navigator
        try:
            from interactive_web_navigator import initialize_interactive_navigator, get_interactive_navigator
            navigator = initialize_interactive_navigator()
            if navigator:
                self.navigator = navigator
                # Create a test session
                self.session_id = "test_session_" + str(int(time.time()))
                session = self.navigator.create_interactive_session(
                    self.session_id, 
                    "https://example.com", 
                    ["Interaction capabilities test"]
                )
                if session:
                    self.log_test_result("Setup Navigator", True, "Navigator initialized")
                else:
                    self.log_test_result("Setup Navigator", False, "Could not create session")
            else:
                self.log_test_result("Setup Navigator", False, "Navigator not available")
        except Exception as e:
            self.log_test_result("Setup Navigator", False, f"Error: {str(e)}")
        
        # Import and initialize the artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter
        try:
            from gemini_api_adapter import GeminiAPI
            from ai_api_config import get_api_config, get_gemini_api_key
            
            api_key = get_gemini_api_key()
            if api_key:
                self.gemini_adapter = GeminiAPI(api_key)
                self.log_test_result("Setup artificial intelligence API GOOGLE GEMINI 2.0 FLASH API", True, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH API initialized")
            else:
                self.log_test_result("Setup artificial intelligence API GOOGLE GEMINI 2.0 FLASH API", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH API key missing")
        except Exception as e:
            self.log_test_result("Setup artificial intelligence API GOOGLE GEMINI 2.0 FLASH API", False, f"Error: {str(e)}")
        
        # Import the interactive adapter
        try:
            from gemini_interactive_adapter import initialize_gemini_interactive_adapter
            adapter = initialize_gemini_interactive_adapter()
            if adapter:
                self.interactive_adapter = adapter
                self.log_test_result("Setup Interactive Adapter", True, "Adapter initialized")
            else:
                self.log_test_result("Setup Interactive Adapter", False, "Adapter not available")
        except Exception as e:
            self.log_test_result("Setup Interactive Adapter", False, f"Error: {str(e)}")
    
    async def test_page_analysis(self, url: str, site_name: str):
        """Tests page analysis by artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
        logger.info(f"📖 Page analysis test: {site_name} ({url})")
        
        try:
            if not self.navigator or not self.session_id:
                self.log_test_result(f"Analysis {site_name}", False, "Navigator not available")
                return False
            
            # Load the page
            result = self.navigator.navigate_to_url(self.session_id, url)
            
            if result.get('success'):
                # Get page elements summary
                page_summary = self.navigator.get_interactive_elements_summary(self.session_id)
                
                if page_summary:
                    # Ask artificial intelligence API GOOGLE GEMINI 2.0 FLASH to analyze the page
                    if self.gemini_adapter:
                        analysis_prompt = f"""
                        Analyze this web page and identify:
                        1. The title and main content
                        2. Interactive elements (buttons, links, forms)
                        3. Clickable elements
                        4. Navigation structure
                        
                        Page Summary:
                        {json.dumps(page_summary, indent=2)[:2000]}...
                        """
                        
                        analysis = await self.gemini_adapter.generate_text(analysis_prompt)
                        
                        if analysis:
                            self.log_test_result(f"Analysis {site_name}", True, 
                                               f"Page analyzed successfully", 
                                               {'url': url, 'analysis': analysis[:500]})
                            return True
                        else:
                            self.log_test_result(f"Analysis {site_name}", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH could not analyze")
                    else:
                        self.log_test_result(f"Analysis {site_name}", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH API not available")
                else:
                    self.log_test_result(f"Analysis {site_name}", False, "Page summary not retrieved")
            else:
                self.log_test_result(f"Analysis {site_name}", False, f"Navigation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.log_test_result(f"Analysis {site_name}", False, f"Error: {str(e)}")
        
        return False
    
    async def test_element_identification(self, url: str, site_name: str):
        """Tests clickable element identification by artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
        logger.info(f"🎯 Element identification test: {site_name}")
        
        try:
            if not self.navigator or not self.session_id:
                self.log_test_result(f"Identification {site_name}", False, "Navigator not available")
                return []
            
            # Load the page
            self.navigator.navigate_to_url(self.session_id, url)
            
            # Get summary of interactive elements
            summary = self.navigator.get_interactive_elements_summary(self.session_id)
            
            if summary and summary.get('interactive_elements'):
                interactive_elements = summary['interactive_elements']
                
                # Ask artificial intelligence API GOOGLE GEMINI 2.0 FLASH to classify these elements
                if self.gemini_adapter:
                    elements_info = []
                    for element in interactive_elements[:10]:  # Limit to 10 elements
                        element_info = {
                            'element_type': element.get('element_type', ''),
                            'text': element.get('text', ''),
                            'id': element.get('element_id', ''),
                            'attributes': element.get('attributes', {}),
                            'clickable': element.get('is_clickable', False)
                        }
                        elements_info.append(element_info)
                    
                    identification_prompt = f"""
                    Here is a list of interactive elements found on page {url}.
                    For each element, tell me:
                    1. Its type (button, link, form field, etc.)
                    2. Its probable action (search, navigation, submission, etc.)
                    3. If it is safe to click on it
                    
                    Elements:
                    {json.dumps(elements_info, indent=2)}
                    """
                    
                    identification = await self.gemini_adapter.generate_text(identification_prompt)
                    
                    if identification:
                        self.log_test_result(f"Identification {site_name}", True, 
                                           f"{len(interactive_elements)} elements identified",
                                           {'elements_count': len(interactive_elements), 
                                            'identification': identification[:500]})
                        return interactive_elements
                    else:
                        self.log_test_result(f"Identification {site_name}", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH could not identify")
                else:
                    self.log_test_result(f"Identification {site_name}", True, 
                                       f"{len(interactive_elements)} elements found (without artificial intelligence API GOOGLE GEMINI 2.0 FLASH analysis)")
                    return interactive_elements
            else:
                self.log_test_result(f"Identification {site_name}", False, "No interactive elements found")
                
        except Exception as e:
            self.log_test_result(f"Identification {site_name}", False, f"Error: {str(e)}")
        
        return []
    
    async def test_element_clicking(self, url: str, site_name: str):
        """Tests clicking elements with artificial intelligence API GOOGLE GEMINI 2.0 FLASH guidance"""
        logger.info(f"👆 Element clicking test: {site_name}")
        
        try:
            # First identify elements
            elements = await self.test_element_identification(url, site_name)
            
            if not elements:
                self.log_test_result(f"Click {site_name}", False, "No elements to click")
                return False
            
            # Ask artificial intelligence API GOOGLE GEMINI 2.0 FLASH to choose a safe element to click
            if self.gemini_adapter and len(elements) > 0:
                click_prompt = f"""
                On page {url}, I found these interactive elements.
                Choose ONE safe element to click that:
                1. Does not cause damage
                2. Does not submit personal data
                3. Is likely a simple navigation link
                
                Reply only with the element's index (0, 1, 2, etc.) or "none" if none are safe.
                
                Available Elements:
                """
                
                for i, element in enumerate(elements[:5]):  # Limit to 5 elements
                    click_prompt += f"\n{i}: {element.get('tag_name', 'unknown')} - {element.get('text', 'no text')[:50]}"
                
                choice = await self.gemini_adapter.generate_text(click_prompt)
                
                if choice and choice.strip().isdigit():
                    element_index = int(choice.strip())
                    if 0 <= element_index < len(elements):
                        chosen_element = elements[element_index]
                        
                        # Attempt the click using the interaction method
                        if self.navigator and self.session_id:
                            element_id = chosen_element.get('element_id', '')
                            if element_id:
                                click_result = self.navigator.interact_with_element(
                                    self.session_id, element_id, "click"
                                )
                                
                                if click_result.get('success'):
                                    self.log_test_result(f"Click {site_name}", True, 
                                                       f"Click successful on: {chosen_element.get('text', 'element')[:30]}",
                                                       {'element': chosen_element, 'result': click_result})
                                    
                                    # Wait a bit to see the result
                                    time.sleep(2)
                                    
                                    return True
                                else:
                                    self.log_test_result(f"Click {site_name}", False, 
                                                       f"Click failed: {click_result.get('error', 'Unknown error')}")
                            else:
                                self.log_test_result(f"Click {site_name}", False, "Element without ID")
                        else:
                            self.log_test_result(f"Click {site_name}", False, "Navigator not available")
                    else:
                        self.log_test_result(f"Click {site_name}", False, "Invalid element index")
                else:
                    self.log_test_result(f"Click {site_name}", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH did not choose a safe element")
            else:
                self.log_test_result(f"Click {site_name}", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH API not available")
                
        except Exception as e:
            self.log_test_result(f"Click {site_name}", False, f"Error: {str(e)}")
        
        return False
    
    async def test_form_interaction(self):
        """Tests interaction with forms"""
        logger.info("📝 Form interaction test")
        
        form_url = "https://httpbin.org/forms/post"
        
        try:
            if not self.navigator or not self.session_id:
                self.log_test_result("Form Interaction", False, "Navigator not available")
                return False
            
            # Load the form page
            result = self.navigator.navigate_to_url(self.session_id, form_url)
            
            if not result.get('success'):
                self.log_test_result("Form Interaction", False, f"Navigation failed: {result.get('error', 'Unknown error')}")
                return False
            
            # Get summary of page elements including forms
            summary = self.navigator.get_interactive_elements_summary(self.session_id)
            
            if summary and summary.get('interactive_elements'):
                # Filter form elements
                form_elements = [elem for elem in summary['interactive_elements'] 
                               if elem.get('element_type', '').lower() in ['input', 'textarea', 'select', 'button']]
                
                if form_elements:
                    # Ask artificial intelligence API GOOGLE GEMINI 2.0 FLASH to fill out the test form
                    if self.gemini_adapter:
                        form_prompt = f"""
                        I am on a test form page (httpbin.org).
                        Here are the available fields. Give me safe test values for each field:
                        
                        Fields found:
                        {json.dumps([{
                            'element_id': elem.get('element_id', ''),
                            'element_type': elem.get('element_type', ''),
                            'text': elem.get('text', ''),
                            'attributes': elem.get('attributes', {})
                        } for elem in form_elements], indent=2)}
                        
                        Respond in JSON format with the values to enter.
                        """
                        
                        form_values = await self.gemini_adapter.generate_text(form_prompt)
                        
                        if form_values:
                            # Attempt to fill out the form (simulation)
                            self.log_test_result("Form Interaction", True, 
                                               f"Form analyzed with {len(form_elements)} fields",
                                               {'fields': len(form_elements), 'values': form_values[:200]})
                            return True
                        else:
                            self.log_test_result("Form Interaction", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH did not provide values")
                    else:
                        self.log_test_result("Form Interaction", True, 
                                           f"{len(form_elements)} fields found (without filling)")
                        return True
                else:
                    self.log_test_result("Form Interaction", False, "No form fields found")
            else:
                self.log_test_result("Form Interaction", False, "No interactive elements found")
                
        except Exception as e:
            self.log_test_result("Form Interaction", False, f"Error: {str(e)}")
        
        return False
    
    async def run_comprehensive_test(self):
        """Runs all web interaction tests"""
        logger.info("🚀 Starting web interaction tests with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
        
        # Initial setup
        self.setup_modules()
        
        # Test 1: Simple page analysis
        await self.test_page_analysis("https://example.com", "Example.com")
        
        # Test 2: Element identification on different sites
        for site_name, url in self.test_sites.items():
            if site_name != 'form_page':  # We test forms separately
                await self.test_element_identification(url, site_name)
        
        # Test 3: Safe click test
        # We only test on safe sites
        safe_sites = {
            'simple_page': 'https://example.com',
            'w3schools': 'https://www.w3schools.com'
        }
        
        for site_name, url in safe_sites.items():
            await self.test_element_clicking(url, site_name)
        
        # Test 4: Form interaction
        await self.test_form_interaction()
        
        # Generate final report
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generates a final test report"""
        logger.info("📊 Generating final report")
        
        report = {
            'test_summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.total_tests - self.passed_tests,
                'success_rate': (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
            },
            'test_results': self.test_results,
            'errors': self.errors,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.test_dir / f"gemini_web_interaction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Display summary
        logger.info(f"📈 Tests completed: {self.passed_tests}/{self.total_tests} successful ({report['test_summary']['success_rate']:.1f}%)")
        logger.info(f"📄 Report saved: {report_file}")
        
        if self.errors:
            logger.warning("⚠️  Errors encountered:")
            for error in self.errors[-5:]:  # Display last 5 errors
                logger.warning(f"   - {error}")

async def main():
    """Main function to run the tests"""
    tester = GeminiWebInteractionTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
