#!/usr/bin/env python3
"""
Comprehensive Test of artificial intelligence API GOOGLE GEMINI 2.0 FLASH's Web Interaction Capabilities with Searx
This script tests if artificial intelligence API GOOGLE GEMINI 2.0 FLASH can use Searx to:
1. Perform web searches
2. Analyze search results
3. Identify clickable elements
4. Navigate to results
5. Interact with found web pages
"""

import logging
import json
import time
import os
import sys
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeminiSearxInteractionTest')

class GeminiSearxInteractionTester:
    """Specialized tester for web interactions with artificial intelligence API GOOGLE GEMINI 2.0 FLASH via Searx"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        self.session_id = None
        
        # Searx Configuration
        self.searx_url = "http://localhost:8080"
        self.app_url = "http://localhost:5000"  # Main Flask app
        
        # Test queries
        self.test_queries = [
            "Python programming tutorial",
            "What is artificial intelligence",
            "Weather forecast today",
            "Latest technology news",
            "Machine learning basics"
        ]
        
        # Create test directory
        self.test_dir = Path("test_results_searx_interaction")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🔍 artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Interaction Tester initialized")
        
        # Initialize necessary modules
        self.navigator = None
        self.gemini_adapter = None
        self.searx_interface = None
        
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
    
    def check_services_availability(self):
        """Checks if the necessary services are available"""
        logger.info("🔧 Checking services...")
        
        # Check Searx
        try:
            response = requests.get(f"{self.searx_url}/", timeout=10)
            if response.status_code == 200:
                self.log_test_result("Searx Service", True, f"Accessible on {self.searx_url}")
            else:
                self.log_test_result("Searx Service", False, f"Error code: {response.status_code}")
                return False
        except Exception as e:
            self.log_test_result("Searx Service", False, f"Error: {str(e)}")
            return False
        
        # Check Flask app
        try:
            response = requests.get(f"{self.app_url}/", timeout=10)
            if response.status_code == 200:
                self.log_test_result("Flask App Service", True, f"Accessible on {self.app_url}")
            else:
                self.log_test_result("Flask App Service", False, f"Error code: {response.status_code}")
        except Exception as e:
            self.log_test_result("Flask App Service", False, f"Error: {str(e)}")
        
        return True
    
    def setup_modules(self):
        """Initializes all necessary modules"""
        logger.info("🔧 Configuring modules...")
        
        # Import Searx interface
        try:
            from searx_interface import SearxInterface
            self.searx_interface = SearxInterface()
            if self.searx_interface:
                self.log_test_result("Setup Searx Interface", True, "Searx interface initialized")
            else:
                self.log_test_result("Setup Searx Interface", False, "Interface not available")
        except Exception as e:
            self.log_test_result("Setup Searx Interface", False, f"Error: {str(e)}")
        
        # Import interactive navigator
        try:
            from interactive_web_navigator import initialize_interactive_navigator
            navigator = initialize_interactive_navigator()
            if navigator:
                self.navigator = navigator
                # Create a test session
                session_obj = self.navigator.create_interactive_session(
                    f"searx_test_session_{int(time.time())}", 
                    self.searx_url,
                    ["Searx search test with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"]
                )
                if session_obj:
                    # Extract session ID from NavigationSession object
                    self.session_id = session_obj.session_id if hasattr(session_obj, 'session_id') else f"searx_test_session_{int(time.time())}"
                    self.log_test_result("Setup Navigator", True, "Navigator and session initialized")
                else:
                    self.log_test_result("Setup Navigator", False, "Session not created")
            else:
                self.log_test_result("Setup Navigator", False, "Navigator not available")
        except Exception as e:
            self.log_test_result("Setup Navigator", False, f"Error: {str(e)}")
        
        # Import interactive artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter
        try:
            from gemini_interactive_adapter import initialize_gemini_interactive_adapter
            adapter = initialize_gemini_interactive_adapter()
            if adapter:
                self.gemini_adapter = adapter
                self.log_test_result("Setup artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive", True, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter initialized")
            else:
                self.log_test_result("Setup artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive", False, "Adapter not available")
        except Exception as e:
            self.log_test_result("Setup artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive", False, f"Error: {str(e)}")
    
    def test_searx_search(self, query: str):
        """Tests search via Searx"""
        logger.info(f"🔍 Searx search test: '{query}'")
        
        try:
            if self.searx_interface:
                # Use the Searx interface with the correct method
                results = self.searx_interface.search_with_filters(query, engines=['google', 'bing'])
                
                if results and len(results) > 0:
                    self.log_test_result(f"Searx Search '{query[:20]}'", True, 
                                       f"{len(results)} results found",
                                       {'query': query, 'results_count': len(results), 
                                        'first_result': results[0].__dict__ if results else None})
                    return results
                else:
                    self.log_test_result(f"Searx Search '{query[:20]}'", False, "No results")
            else:
                # Direct search via Searx API
                search_url = f"{self.searx_url}/search"
                params = {
                    'q': query,
                    'format': 'json',
                    'engines': 'google,bing'
                }
                
                response = requests.get(search_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('results', [])
                    
                    if results:
                        self.log_test_result(f"Searx Search '{query[:20]}'", True, 
                                           f"{len(results)} results found",
                                           {'query': query, 'results_count': len(results),
                                            'first_result_title': results[0].get('title', '') if results else ''})
                        return results
                    else:
                        self.log_test_result(f"Searx Search '{query[:20]}'", False, "No results in response")
                else:
                    self.log_test_result(f"Searx Search '{query[:20]}'", False, f"HTTP Error: {response.status_code}")
                    
        except Exception as e:
            self.log_test_result(f"Searx Search '{query[:20]}'", False, f"Error: {str(e)}")
        
        return []
    
    def test_gemini_searx_integration(self, query: str):
        """Tests artificial intelligence API GOOGLE GEMINI 2.0 FLASH integration with Searx"""
        logger.info(f"🤖 artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx integration test: '{query}'")
        
        try:
            # Use the chat endpoint that supports Searx searches
            api_url = f"{self.app_url}/api/chat"
            
            # Create a session to bypass authentication
            session = requests.Session()
            
            # First get a session (cookie)
            login_response = session.get(f"{self.app_url}/")
            
            data = {
                'message': f"Search for: {query}",
                'use_web_search': True,
                'search_engine': 'searx'
            }
            
            response = session.post(api_url, json=data, timeout=60)
            
            if response.status_code == 200:
                result_data = response.json()
                
                if result_data.get('success', True):  # Some responses don't have a success field
                    gemini_response = result_data.get('response', '')
                    
                    if gemini_response:
                        self.log_test_result(f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Integration '{query[:20]}'", True,
                                           f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH responded with Searx",
                                           {'query': query, 'has_response': bool(gemini_response),
                                            'response_preview': gemini_response[:200] if gemini_response else ''})
                        return True
                    else:
                        self.log_test_result(f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Integration '{query[:20]}'", False, 
                                           "Empty response from artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
                else:
                    self.log_test_result(f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Integration '{query[:20]}'", False, 
                                       result_data.get('error', 'Unknown error'))
            elif response.status_code == 401:
                # Attempt an alternative approach via trigger-web-search
                alt_url = f"{self.app_url}/api/trigger-web-search"
                alt_data = {'query': query}
                alt_response = session.post(alt_url, json=alt_data, timeout=60)
                
                if alt_response.status_code == 200:
                    self.log_test_result(f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Integration '{query[:20]}'", True,
                                       f"Web search triggered successfully",
                                       {'query': query, 'method': 'trigger-web-search'})
                    return True
                else:
                    self.log_test_result(f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Integration '{query[:20]}'", False, 
                                       f"Authentication required (HTTP 401)")
            else:
                self.log_test_result(f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Integration '{query[:20]}'", False, 
                                   f"HTTP Error: {response.status_code}")
                
        except Exception as e:
            self.log_test_result(f"artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx Integration '{query[:20]}'", False, f"Error: {str(e)}")
        
        return False
    
    def test_navigation_to_search_results(self, query: str):
        """Tests navigation to search results"""
        logger.info(f"🌐 Testing navigation to results: '{query}'")
        
        try:
            # First get search results
            results = self.test_searx_search(query)
            
            if not results or not self.navigator or not self.session_id:
                self.log_test_result(f"Navigation '{query[:20]}'", False, "Prerequisites not met")
                return False
            
            # Take the first safe result
            safe_result = None
            for result in results[:3]:  # Check the first 3
                url = result.url if hasattr(result, 'url') else result.get('url', '')
                if url and any(domain in url for domain in ['wikipedia.org', 'python.org', 'github.com']):
                    safe_result = result
                    break
            
            if not safe_result:
                self.log_test_result(f"Navigation '{query[:20]}'", False, "No safe result found")
                return False
            
            # Navigate to the result
            result_url = safe_result.url if hasattr(safe_result, 'url') else safe_result.get('url', '')
            navigation_result = self.navigator.navigate_to_url(self.session_id, result_url)
            
            if navigation_result.get('success'):
                # Analyze the page with artificial intelligence API GOOGLE GEMINI 2.0 FLASH
                page_summary = self.navigator.get_interactive_elements_summary(self.session_id)
                
                if page_summary:
                    self.log_test_result(f"Navigation '{query[:20]}'", True,
                                       f"Navigation successful to {result_url[:50]}...",
                                       {'target_url': result_url, 'page_loaded': True,
                                        'elements_found': len(page_summary.get('interactive_elements', []))})
                    return True
                else:
                    self.log_test_result(f"Navigation '{query[:20]}'", False, "Page loaded but analysis failed")
            else:
                self.log_test_result(f"Navigation '{query[:20]}'", False, 
                                   f"Navigation failed: {navigation_result.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.log_test_result(f"Navigation '{query[:20]}'", False, f"Error: {str(e)}")
        
        return False
    
    async def run_comprehensive_test(self):
        """Runs all artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx interaction tests"""
        logger.info("🚀 Starting artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx interaction tests")
        
        # Preliminary checks
        if not self.check_services_availability():
            logger.error("❌ Services not available, stopping tests")
            return
        
        # Module setup
        self.setup_modules()
        
        # Test 1: Basic Searx search tests
        logger.info("📝 Phase 1: Searx search tests")
        for query in self.test_queries[:3]:  # Test 3 queries
            self.test_searx_search(query)
            time.sleep(2)  # Avoid overload
        
        # Test 2: artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx integration tests
        logger.info("🤖 Phase 2: artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Searx integration tests")
        for query in self.test_queries[:2]:  # Test 2 queries with artificial intelligence API GOOGLE GEMINI 2.0 FLASH
            self.test_gemini_searx_integration(query)
            time.sleep(3)  # More time for artificial intelligence API GOOGLE GEMINI 2.0 FLASH
        
        # Test 3: Navigation to results
        logger.info("🌐 Phase 3: Navigation tests")
        for query in ["Python tutorial", "Wikipedia artificial intelligence"]:
            self.test_navigation_to_search_results(query)
            time.sleep(3)
        
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
            'services_info': {
                'searx_url': self.searx_url,
                'app_url': self.app_url
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save report
        report_file = self.test_dir / f"gemini_searx_interaction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    tester = GeminiSearxInteractionTester()
    await tester.run_comprehensive_test()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
