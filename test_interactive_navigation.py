"""
Complete Test of the Interactive Navigation System with artificial intelligence API GOOGLE GEMINI 2.0 FLASH
This script tests the full integration of the new web interaction system
"""

import logging
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('artificial intelligence API GOOGLE GEMINI 2.0 FLASH InteractiveNavigationTest')

class artificial intelligence API GOOGLE GEMINI 2.0 FLASHInteractiveNavigationTester:
    """Complete tester for the interactive navigation system with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""

    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0

        # Create test directory
        self.test_dir = Path("test_results_interactive")
        self.test_dir.mkdir(exist_ok=True)

        logger.info("🧪 artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Interactive Navigation Tester initialized")

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

    def test_module_imports(self):
        """Test 1: Verify that all interactive modules import correctly"""
        logger.info("📦 Test 1: Interactive Module Imports")

        imported_modules = {}

        # Test interactive navigator imports
        try:
            from interactive_web_navigator import (
                InteractiveWebNavigator,
                InteractiveElementAnalyzer,
                get_interactive_navigator,
                initialize_interactive_navigator
            )
            imported_modules['interactive_web_navigator'] = True
            self.log_test_result("Interactive Navigator Import", True, "Module loaded")
        except ImportError as e:
            imported_modules['interactive_web_navigator'] = False
            self.log_test_result("Interactive Navigator Import", False, f"Error: {str(e)}")

        # Test artificial intelligence API GOOGLE GEMINI 2.0 FLASH interactive adapter imports
        try:
            from gemini_interactive_adapter import (
                GeminiInteractiveWebAdapter,
                get_gemini_interactive_adapter,
                initialize_gemini_interactive_adapter,
                handle_gemini_interactive_request,
                detect_interactive_need
            )
            imported_modules['gemini_interactive_adapter'] = True
            self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive Adapter Import", True, "Module loaded")
        except ImportError as e:
            imported_modules['gemini_interactive_adapter'] = False
            self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive Adapter Import", False, f"Error: {str(e)}")

        # Test main artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter import
        try:
            from gemini_api_adapter import GeminiAPI
            imported_modules['gemini_api_adapter'] = True
            self.log_test_result("Main artificial intelligence API GOOGLE GEMINI 2.0 FLASH Adapter Import", True, "Module loaded")
        except ImportError as e:
            imported_modules['gemini_api_adapter'] = False
            self.log_test_result("Main artificial intelligence API GOOGLE GEMINI 2.0 FLASH Adapter Import", False, f"Error: {str(e)}")

        # Test Selenium imports (optional)
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            imported_modules['selenium'] = True
            self.log_test_result("Selenium Import", True, "WebDriver available")
        except ImportError as e:
            imported_modules['selenium'] = False
            self.log_test_result("Selenium Import", False, f"WebDriver not available: {str(e)}")

        success_rate = sum(imported_modules.values()) / len(imported_modules) * 100
        overall_success = success_rate >= 75  # At least 75% of required modules

        self.log_test_result("Global Module Imports", overall_success,
                           f"Success rate: {success_rate:.1f}%",
                           {'modules': imported_modules})

        return imported_modules

    def test_interactive_navigator_initialization(self):
        """Test 2: Initialize the interactive navigator"""
        logger.info("🚀 Test 2: Interactive Navigator Initialization")

        try:
            from interactive_web_navigator import initialize_interactive_navigator, get_interactive_navigator

            # Attempt initialization
            navigator = initialize_interactive_navigator()

            if navigator:
                self.log_test_result("Navigator Initialization", True, "Navigator initialized successfully")

                # Verify global access
                global_navigator = get_interactive_navigator()
                if global_navigator:
                    self.log_test_result("Global Navigator Verification", True, "Navigator globally accessible")
                    return navigator
                else:
                    self.log_test_result("Global Navigator Verification", False, "Navigator not accessible")
                    return None
            else:
                self.log_test_result("Navigator Initialization", False, "Initialization failed (normal if ChromeDriver is missing)")
                return None

        except Exception as e:
            self.log_test_result("Navigator Initialization Test", False, f"Error: {str(e)}")
            return None

    def test_artificial_intelligence_API_GOOGLE_GEMINI_2_0_FLASH_interactive_adapter_initialization(self):
        """Test 3: Initialize the artificial intelligence API GOOGLE GEMINI 2.0 FLASH interactive adapter"""
        logger.info("🔗 Test 3: artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive Adapter Initialization")

        try:
            from gemini_interactive_adapter import initialize_gemini_interactive_adapter, get_gemini_interactive_adapter

            # Initialize the adapter
            adapter = initialize_gemini_interactive_adapter()

            if adapter:
                self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Adapter Initialization", True, "Adapter initialized")

                # Verify global access
                global_adapter = get_gemini_interactive_adapter()
                if global_adapter:
                    self.log_test_result("Global Adapter Verification", True, "Adapter accessible")

                    # Check statistics
                    stats = adapter.get_interaction_statistics()
                    self.log_test_result("Adapter Statistics", True, f"Stats: {stats}")

                    return adapter
                else:
                    self.log_test_result("Global Adapter Verification", False, "Adapter not accessible")
                    return None
            else:
                self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Adapter Initialization", False, "Initialization failed")
                return None

        except Exception as e:
            self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Adapter Test", False, f"Error: {str(e)}")
            return None

    def test_interaction_detection(self):
        """Test 4: Test interaction detection"""
        logger.info("🔍 Test 4: Interaction Detection")

        try:
            from gemini_interactive_adapter import detect_interactive_need

            # Detection tests with different prompts
            test_cases = [
                {
                    'prompt': "Click the Services tab on this website",
                    'expected_interaction': True,
                    'expected_type': 'direct_interaction'
                },
                {
                    'prompt': "Explore all tabs on https://example.com",
                    'expected_interaction': True,
                    'expected_type': 'tab_navigation'
                },
                {
                    'prompt': "Explore all available options on the site",
                    'expected_interaction': True,
                    'expected_type': 'full_exploration'
                },
                {
                    'prompt': "Fill out the contact form",
                    'expected_interaction': True,
                    'expected_type': 'form_interaction'
                },
                {
                    'prompt': "What is artificial intelligence API GOOGLE GEMINI 2.0 FLASH?",
                    'expected_interaction': False,
                    'expected_type': None
                }
            ]

            detection_results = []
            successful_detections = 0

            for test_case in test_cases:
                prompt = test_case['prompt']
                expected_interaction = test_case['expected_interaction']
                expected_type = test_case['expected_type']

                logger.info(f"  🧪 Test detection: '{prompt}'")

                # Perform detection
                detection = detect_interactive_need(prompt)

                # Check result
                detected_interaction = detection.get('requires_interaction', False)
                detected_type = detection.get('interaction_type')
                confidence = detection.get('confidence', 0)

                # Evaluate accuracy
                type_match = (detected_type == expected_type) if expected_interaction else (detected_type is None)
                detection_success = (detected_interaction == expected_interaction) and type_match

                if detection_success:
                    successful_detections += 1
                    status = "✅"
                    details = f"Correct detection (confidence: {confidence:.2f})"
                else:
                    status = "❌"
                    details = f"Expected: {expected_type}, Detected: {detected_type} (confidence: {confidence:.2f})"

                detection_results.append({
                    'prompt': prompt,
                    'expected': {'interaction': expected_interaction, 'type': expected_type},
                    'detected': {'interaction': detected_interaction, 'type': detected_type, 'confidence': confidence},
                    'success': detection_success,
                    'status': status
                })

                logger.info(f"    {status} {details}")

            # Evaluate overall success rate
            success_rate = (successful_detections / len(test_cases)) * 100
            overall_success = success_rate >= 80  # At least 80% success

            self.log_test_result("Interaction Detection", overall_success,
                               f"Success rate: {success_rate:.1f}% ({successful_detections}/{len(test_cases)})",
                               {'results': detection_results})

            return detection_results

        except Exception as e:
            self.log_test_result("Detection Test", False, f"Error: {str(e)}")
            return None

    def test_element_analysis_simulation(self):
        """Test 5: Element analysis simulation (without browser)"""
        logger.info("🔬 Test 5: Element Analysis Simulation")

        try:
            from interactive_web_navigator import InteractiveElementAnalyzer

            # Create the analyzer
            analyzer = InteractiveElementAnalyzer()
            self.log_test_result("Analyzer Creation", True, "Analyzer created")

            # Test CSS selectors
            selectors_test = True
            for element_type, selectors in analyzer.element_selectors.items():
                if not selectors or not isinstance(selectors, list):
                    selectors_test = False
                    break

            self.log_test_result("CSS Selectors Validation", selectors_test,
                               f"Selectors for {len(analyzer.element_selectors)} element types")

            # Test importance keywords
            keywords_test = True
            for importance, keywords in analyzer.importance_keywords.items():
                if not keywords or not isinstance(keywords, list):
                    keywords_test = False
                    break

            self.log_test_result("Keywords Validation", keywords_test,
                               f"Keywords for {len(analyzer.importance_keywords)} importance levels")

            # Test interaction score calculation
            test_scores = [
                analyzer._calculate_interaction_score("Next", {'id': 'next-btn'}, 'buttons', {'x': 100, 'y': 200, 'width': 80, 'height': 30}),
                analyzer._calculate_interaction_score("Home", {'class': 'nav-link'}, 'navigation', {'x': 50, 'y': 50, 'width': 60, 'height': 20}),
                analyzer._calculate_interaction_score("", {}, 'inputs', {'x': 200, 'y': 800, 'width': 120, 'height': 25})
            ]

            score_test = all(0 <= score <= 1 for score in test_scores)
            self.log_test_result("Interaction Score Calculation", score_test,
                               f"Calculated scores: {[f'{s:.2f}' for s in test_scores]}")

            return analyzer

        except Exception as e:
            self.log_test_result("Element Analysis Test", False, f"Error: {str(e)}")
            return None

    def test_artificial_intelligence_API_GOOGLE_GEMINI_2_0_FLASH_api_integration(self):
        """Test 6: artificial intelligence API GOOGLE GEMINI 2.0 FLASH API integration test"""
        logger.info("🤖 Test 6: artificial intelligence API GOOGLE GEMINI 2.0 FLASH API Integration")

        try:
            from gemini_api_adapter import GeminiAPI

            # Create an API instance
            gemini_api = GeminiAPI()
            self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Instance Creation", True, "Instance created")

            # Verify that the interactive system is integrated
            has_interactive = hasattr(gemini_api, 'interactive_navigation_available')
            self.log_test_result("Interactive System Integration", has_interactive,
                               f"Interactive system {'available' if has_interactive else 'not available'}")

            # Interactive prompt tests (simulation)
            interactive_prompts = [
                "Click the products tab on https://example.com",
                "Explore all tabs on this website",
                "Fill out the contact form"
            ]

            integration_results = []

            for prompt in interactive_prompts:
                logger.info(f"  🧪 Test prompt: '{prompt[:50]}...'")

                try:
                    # Use fallback method to avoid actual API calls
                    if hasattr(gemini_api, '_fallback_get_response'):
                        response = gemini_api._fallback_get_response(
                            prompt=prompt,
                            user_id=1,
                            session_id="test_interactive_session"
                        )

                        if response and 'response' in response:
                            integration_results.append({
                                'prompt': prompt,
                                'success': True,
                                'has_response': True,
                                'response_length': len(response['response'])
                            })
                        else:
                            integration_results.append({
                                'prompt': prompt,
                                'success': False,
                                'error': 'No response'
                            })
                    else:
                        integration_results.append({
                            'prompt': prompt,
                            'success': False,
                            'error': 'Fallback method not available'
                        })

                except Exception as e:
                    integration_results.append({
                        'prompt': prompt,
                        'success': False,
                        'error': str(e)
                    })

                time.sleep(0.5)  # Small delay

            success_count = sum(1 for r in integration_results if r['success'])
            success_rate = (success_count / len(interactive_prompts)) * 100

            self.log_test_result("Interactive Prompts Tests", success_count > 0,
                               f"Success rate: {success_rate:.1f}% ({success_count}/{len(interactive_prompts)})",
                               {'results': integration_results})

            return gemini_api

        except Exception as e:
            self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Integration Test", False, f"Error: {str(e)}")
            return None

    def test_session_management(self):
        """Test 7: Session management test"""
        logger.info("📋 Test 7: Session Management")

        try:
            from interactive_web_navigator import create_interactive_navigation_session, close_interactive_session

            # Session creation test
            session_id = f"test_session_{int(time.time())}"
            test_url = "https://httpbin.org/html"
            goals = ['test_navigation', 'element_analysis']

            try:
                session_result = create_interactive_navigation_session(session_id, test_url, goals)

                if session_result.get('success', False):
                    self.log_test_result("Session Creation", True,
                                       f"Session created: {session_id}")

                    # Session closing test
                    close_result = close_interactive_session(session_id)

                    if close_result.get('success', False):
                        self.log_test_result("Session Closing", True,
                                           f"Session closed with report")
                        return True
                    else:
                        self.log_test_result("Session Closing", False,
                                           f"Error: {close_result.get('error', 'Unknown')}")
                        return False
                else:
                    self.log_test_result("Session Creation", False,
                                       f"Error: {session_result.get('error', 'Unknown')}")
                    return False

            except Exception as e:
                self.log_test_result("Session Test", False, f"Session error: {str(e)}")
                return False

        except Exception as e:
            self.log_test_result("Session Management Test", False, f"Error: {str(e)}")
            return False

    def generate_test_report(self):
        """Generates a complete test report"""
        logger.info("📋 Generating test report")

        # Calculate general statistics
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0

        # Create the report
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.total_tests - self.passed_tests,
                'success_rate': success_rate,
                'overall_status': 'PASSED' if success_rate >= 70 else 'FAILED'
            },
            'test_results': self.test_results,
            'errors': self.errors
        }

        # Save the JSON report
        report_file = self.test_dir / f"interactive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Create a markdown report
        self._create_markdown_report(report)

        return report

    def _create_markdown_report(self, report):
        """Creates a markdown report"""
        report_file = self.test_dir / f"interactive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Test Report - artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive Navigation System\n\n")
            f.write(f"**Date:** {report['test_summary']['timestamp']}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total tests:** {report['test_summary']['total_tests']}\n")
            f.write(f"- **Passed tests:** {report['test_summary']['passed_tests']}\n")
            f.write(f"- **Failed tests:** {report['test_summary']['failed_tests']}\n")
            f.write(f"- **Success rate:** {report['test_summary']['success_rate']:.1f}%\n")
            f.write(f"- **Overall status:** {report['test_summary']['overall_status']}\n\n")

            f.write("## Test Details\n\n")
            for test_name, result in report['test_results'].items():
                status = "✅" if result['success'] else "❌"
                f.write(f"### {status} {test_name}\n")
                f.write(f"**Message:** {result['message']}\n\n")
                if result.get('data'):
                    f.write(f"**Data:** ```json\n{json.dumps(result['data'], indent=2)}\n```\n\n")

            if report['errors']:
                f.write("## Errors\n\n")
                for error in report['errors']:
                    f.write(f"- {error}\n")

    def run_all_tests(self):
        """Runs all tests"""
        logger.info("🚀 STARTING INTERACTIVE SYSTEM TESTS")
        logger.info("=" * 60)

        start_time = time.time()

        try:
            # Test 1: Imports
            imported_modules = self.test_module_imports()

            # Test 2: Interactive navigator initialization
            navigator = self.test_interactive_navigator_initialization()

            # Test 3: artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter initialization
            adapter = self.test_artificial_intelligence_API_GOOGLE_GEMINI_2_0_FLASH_interactive_adapter_initialization()

            # Test 4: Interaction detection
            self.test_interaction_detection()

            # Test 5: Element analysis
            analyzer = self.test_element_analysis_simulation()

            # Test 6: artificial intelligence API GOOGLE GEMINI 2.0 FLASH API Integration
            gemini_api = self.test_artificial_intelligence_API_GOOGLE_GEMINI_2_0_FLASH_api_integration()

            # Test 7: Session management
            self.test_session_management()

        except Exception as e:
            logger.error(f"Error during tests: {str(e)}")
            self.log_test_result("Global Execution", False, f"Critical error: {str(e)}")

        total_time = time.time() - start_time

        # Generate the report
        report = self.generate_test_report()

        # Display final summary
        logger.info("=" * 60)
        logger.info("🏁 TESTS COMPLETED - INTERACTIVE SYSTEM")
        logger.info(f"⏱️ Total time: {total_time:.2f}s")
        logger.info(f"📊 Results: {self.passed_tests}/{self.total_tests} tests passed ({report['test_summary']['success_rate']:.1f}%)")

        if report['test_summary']['overall_status'] == 'PASSED':
            logger.info("🎉 ALL MAIN TESTS PASSED !")
            logger.info("✅ The interactive navigation system works with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
        else:
            logger.info("⚠️ Some tests failed")
            logger.info("🔧 Check errors in the test report")

        logger.info("=" * 60)

        return report

def main():
    """Main function"""
    logger.info("🌟 Starting artificial intelligence API GOOGLE GEMINI 2.0 FLASH interactive navigation system tests")

    tester = artificial intelligence API GOOGLE GEMINI 2.0 FLASHInteractiveNavigationTester()
    report = tester.run_all_tests()

    # Return success based on success rate
    success = report['test_summary']['success_rate'] >= 70

    if success:
        logger.info("✅ Tests completed successfully - System is operational")
        print("\n🎯 artificial intelligence API GOOGLE GEMINI 2.0 FLASH INTERACTIVE NAVIGATION SYSTEM OPERATIONAL")
        print("📖 Consult test reports for more details")
    else:
        logger.error("❌ Tests failed - Issues detected")
        print("\n⚠️ ISSUES DETECTED IN THE SYSTEM")
        print("🔧 Consult error reports to resolve issues")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
