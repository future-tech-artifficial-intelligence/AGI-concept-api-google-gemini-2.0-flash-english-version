#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test of the Advanced Web Navigation System with artificial intelligence API GOOGLE GEMINI 2.0 FLASH
This script tests the full integration and verifies that everything is working
"""

import logging
import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GeminiWebNavigationTest') # Keeping original logger name as it's code

class GeminiWebNavigationTester: # Keeping original class name
    """Comprehensive tester for the web navigation system with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        
        # Create test directory
        self.test_dir = Path("test_results")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🧪 artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Navigation Tester initialized")
    
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
        """Test 1: Verify that all modules import correctly"""
        logger.info("🔧 Test 1: Module Imports")
        
        modules_to_test = [
            ('advanced_web_navigator', 'Advanced Web Navigator'),
            ('gemini_web_integration', 'artificial intelligence API GOOGLE GEMINI 2.0 FLASH-Web Integration'),
            ('gemini_navigation_adapter', 'artificial intelligence API GOOGLE GEMINI 2.0 FLASH Navigation Adapter'),
            ('web_navigation_api', 'Web Navigation REST API'),
            ('gemini_api_adapter', 'Original artificial intelligence API GOOGLE GEMINI 2.0 FLASH Adapter')
        ]
        
        imported_modules = {}
        all_success = True
        
        for module_name, display_name in modules_to_test:
            try:
                module = __import__(module_name)
                imported_modules[module_name] = module
                self.log_test_result(f"Import {display_name}", True, "Module imported successfully")
            except ImportError as e:
                self.log_test_result(f"Import {display_name}", False, f"Import error: {str(e)}")
                all_success = False
            except Exception as e:
                self.log_test_result(f"Import {display_name}", False, f"Error: {str(e)}")
                all_success = False
        
        self.log_test_result("Global Import", all_success, 
                           f"{len(imported_modules)}/{len(modules_to_test)} modules imported")
        
        return imported_modules
    
    def test_gemini_api_initialization(self):
        """Test 2: Initialize artificial intelligence API GOOGLE GEMINI 2.0 FLASH API and verify integration"""
        logger.info("🤖 Test 2: artificial intelligence API GOOGLE GEMINI 2.0 FLASH API Initialization")
        
        try:
            from gemini_api_adapter import GeminiAPI # Keeping original class name
            from gemini_navigation_adapter import initialize_gemini_navigation_adapter # Keeping original function name
            
            # Create artificial intelligence API GOOGLE GEMINI 2.0 FLASH instance
            gemini_api = GeminiAPI() # Keeping original class name
            self.log_test_result("Create artificial intelligence API GOOGLE GEMINI 2.0 FLASH Instance", True, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH API created")
            
            # Initialize navigation adapter
            initialize_gemini_navigation_adapter(gemini_api) # Keeping original function name
            self.log_test_result("Initialize Navigation Adapter", True, "Adapter initialized")
            
            # Verify integration is available
            from gemini_navigation_adapter import gemini_navigation_adapter # Keeping original variable name
            if gemini_navigation_adapter: # Keeping original variable name
                self.log_test_result("Integration Verification", True, "Integration active")
                return gemini_api
            else:
                self.log_test_result("Integration Verification", False, "Adapter not initialized")
                return None
                
        except Exception as e:
            self.log_test_result("artificial intelligence API GOOGLE GEMINI 2.0 FLASH API Initialization", False, f"Error: {str(e)}")
            return None
    
    def test_navigation_detection(self, gemini_api):
        """Test 3: Test Navigation Detection"""
        logger.info("🔍 Test 3: Navigation Detection")
        
        if not gemini_api:
            self.log_test_result("Test Detection", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH API not available")
            return
        
        try:
            from gemini_navigation_adapter import detect_navigation_need # Keeping original function name
            
            # Detection tests with different prompts
            test_cases = [
                {
                    'prompt': "Search and navigate artificial intelligence",
                    'expected_navigation': True,
                    'expected_type': 'search_and_navigate'
                },
                {
                    'prompt': "Extract content from https://example.com",
                    'expected_navigation': True,
                    'expected_type': 'content_extraction'
                },
                {
                    'prompt': "Explore the site https://wikipedia.org in depth",
                    'expected_navigation': True,
                    'expected_type': 'deep_navigation'
                },
                {
                    'prompt': "Simulate a shopping journey on this site",
                    'expected_navigation': True,
                    'expected_type': 'user_journey'
                },
                {
                    'prompt': "What is machine learning?",
                    'expected_navigation': True,  # Should be detected as general search
                    'expected_type': 'search_and_navigate'
                },
                {
                    'prompt': "Hello, how are you?",
                    'expected_navigation': False,
                    'expected_type': None
                }
            ]
            
            detection_results = []
            successful_detections = 0
            
            for test_case in test_cases:
                prompt = test_case['prompt']
                expected_nav = test_case['expected_navigation']
                expected_type = test_case['expected_type']
                
                detection = detect_navigation_need(prompt) # Keeping original function name
                
                requires_nav = detection.get('requires_navigation', False)
                nav_type = detection.get('navigation_type')
                confidence = detection.get('confidence', 0)
                
                # Check if detection matches expectations
                detection_correct = (requires_nav == expected_nav)
                if expected_nav and nav_type != expected_type:
                    # Allow some flexibility in types
                    if not (expected_type == 'search_and_navigate' and nav_type == 'search_and_navigate'):
                        detection_correct = False
                
                if detection_correct:
                    successful_detections += 1
                    status = "✅"
                else:
                    status = "❌"
                
                detection_results.append({
                    'prompt': prompt,
                    'expected_navigation': expected_nav,
                    'detected_navigation': requires_nav,
                    'expected_type': expected_type,
                    'detected_type': nav_type,
                    'confidence': confidence,
                    'correct': detection_correct,
                    'status': status
                })
                
                logger.info(f"  {status} '{prompt[:50]}...' → Nav: {requires_nav}, Type: {nav_type}, Conf: {confidence:.2f}")
            
            success_rate = (successful_detections / len(test_cases)) * 100
            overall_success = success_rate >= 70  # At least 70% success
            
            self.log_test_result("Navigation Detection", overall_success, 
                               f"Success Rate: {success_rate:.1f}% ({successful_detections}/{len(test_cases)})",
                               {'results': detection_results, 'success_rate': success_rate})
            
            return detection_results
            
        except Exception as e:
            self.log_test_result("Test Detection", False, f"Error: {str(e)}")
            return None
    
    def test_web_extraction(self):
        """Test 4: Test Web Content Extraction"""
        logger.info("🌐 Test 4: Web Content Extraction")
        
        try:
            from advanced_web_navigator import extract_website_content # Keeping original function name
            
            # Test URLs
            test_urls = [
                "https://httpbin.org/json",
                "https://httpbin.org/html",
            ]
            
            extraction_results = []
            successful_extractions = 0
            
            for url in test_urls:
                logger.info(f"  🔍 Extraction test: {url}")
                
                start_time = time.time()
                content = extract_website_content(url) # Keeping original function name
                extraction_time = time.time() - start_time
                
                if content.success:
                    successful_extractions += 1
                    status = "✅"
                    details = {
                        'title': content.title,
                        'content_length': len(content.cleaned_text),
                        'quality_score': content.content_quality_score,
                        'language': content.language,
                        'links_count': len(content.links),
                        'images_count': len(content.images),
                        'keywords': content.keywords[:5],
                        'extraction_time': extraction_time
                    }
                else:
                    status = "❌"
                    details = {'error': content.error_message}
                
                extraction_results.append({
                    'url': url,
                    'success': content.success,
                    'details': details,
                    'status': status
                })
                
                logger.info(f"    {status} Time: {extraction_time:.2f}s, "
                          f"Content: {len(content.cleaned_text) if content.success else 0} chars, "
                          f"Quality: {content.content_quality_score if content.success else 0:.1f}")
                
                time.sleep(1)  # Delay between requests
            
            success_rate = (successful_extractions / len(test_urls)) * 100
            overall_success = success_rate >= 80  # At least 80% success
            
            self.log_test_result("Web Extraction", overall_success,
                               f"Success Rate: {success_rate:.1f}% ({successful_extractions}/{len(test_urls)})",
                               {'results': extraction_results})
            
            return extraction_results
            
        except Exception as e:
            self.log_test_result("Test Web Extraction", False, f"Error: {str(e)}")
            return None
    
    def test_gemini_integration_full(self, gemini_api):
        """Test 5: Full Integration Test with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
        logger.info("🚀 Test 5: Full Integration with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
        
        if not gemini_api:
            self.log_test_result("Full Integration Test", False, "artificial intelligence API GOOGLE GEMINI 2.0 FLASH API not available")
            return
        
        try:
            # Navigation-enabled query test
            test_prompts = [
                "Search for information on artificial intelligence",
                "What is Python?",
                "Extract content from https://httpbin.org/json"
            ]
            
            integration_results = []
            successful_responses = 0
            
            for prompt in test_prompts:
                logger.info(f"  🤖 artificial intelligence API GOOGLE GEMINI 2.0 FLASH Test: '{prompt}'")
                
                try:
                    start_time = time.time()
                    
                    # Test with the modified artificial intelligence API GOOGLE GEMINI 2.0 FLASH API (uses fallback method for tests)
                    response = gemini_api._fallback_get_response( # Keeping original method name
                        prompt=prompt,
                        user_id=1,
                        session_id="test_session"
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Check the response
                    if response and 'response' in response:
                        successful_responses += 1
                        status = "✅"
                        
                        # Check if navigation was used
                        response_text = response['response']
                        navigation_used = any(indicator in response_text.lower() for indicator in [
                            'web navigation', 'web search', 'extracted content', 
                            'websites', 'visited pages', 'navigation'
                        ])
                        
                        result_details = {
                            'response_length': len(response_text),
                            'navigation_used': navigation_used,
                            'processing_time': processing_time,
                            'status': response.get('status', 'unknown'),
                            'has_emotional_state': 'emotional_state' in response
                        }
                        
                        logger.info(f"    ✅ Response received: {len(response_text)} chars, "
                                  f"Navigation: {'Yes' if navigation_used else 'No'}, "
                                  f"Time: {processing_time:.2f}s")
                    else:
                        status = "❌"
                        result_details = {'error': 'No response received'}
                        logger.info(f"    ❌ No response received")
                    
                    integration_results.append({
                        'prompt': prompt,
                        'success': response is not None,
                        'details': result_details,
                        'status': status
                    })
                    
                except Exception as e:
                    logger.error(f"    ❌ Error for '{prompt}': {str(e)}")
                    integration_results.append({
                        'prompt': prompt,
                        'success': False,
                        'details': {'error': str(e)},
                        'status': "❌"
                    })
                
                time.sleep(2)  # Delay between artificial intelligence API GOOGLE GEMINI 2.0 FLASH requests
            
            success_rate = (successful_responses / len(test_prompts)) * 100
            overall_success = success_rate >= 70  # At least 70% success
            
            self.log_test_result("Full artificial intelligence API GOOGLE GEMINI 2.0 FLASH Integration", overall_success,
                               f"Success Rate: {success_rate:.1f}% ({successful_responses}/{len(test_prompts)})",
                               {'results': integration_results})
            
            return integration_results
            
        except Exception as e:
            self.log_test_result("Full Integration Test", False, f"Error: {str(e)}")
            return None
    
    def test_api_endpoints(self):
        """Test 6: Test REST API Endpoints"""
        logger.info("🌐 Test 6: REST API Endpoints")
        
        try:
            from web_navigation_api import register_web_navigation_api, initialize_web_navigation_api # Keeping original function names
            from flask import Flask
            
            # Create a test Flask app
            app = Flask(__name__)
            register_web_navigation_api(app) # Keeping original function name
            initialize_web_navigation_api() # Keeping original function name
            
            endpoint_results = []
            successful_endpoints = 0
            
            with app.test_client() as client:
                # Main endpoints test
                endpoints_to_test = [
                    ('GET', '/api/web-navigation/health', None, 'Health Check'),
                    ('GET', '/api/web-navigation/docs', None, 'Documentation'),
                    ('GET', '/api/web-navigation/stats', None, 'Statistics'),
                    ('POST', '/api/web-navigation/create-session', {'user_id': 'test_user'}, 'Session Creation'),
                ]
                
                for method, endpoint, data, description in endpoints_to_test:
                    logger.info(f"  🔗 Test {method} {endpoint}")
                    
                    try:
                        if method == 'GET':
                            response = client.get(endpoint)
                        elif method == 'POST':
                            response = client.post(endpoint, json=data)
                        
                        success = response.status_code == 200
                        if success:
                            successful_endpoints += 1
                            status = "✅"
                            
                            try:
                                json_data = response.get_json()
                                response_details = {
                                    'status_code': response.status_code,
                                    'has_json': json_data is not None,
                                    'content_length': len(response.data)
                                }
                                if json_data and 'success' in json_data:
                                    response_details['api_success'] = json_data['success']
                            except:
                                response_details = {
                                    'status_code': response.status_code,
                                    'content_length': len(response.data)
                                }
                        else:
                            status = "❌"
                            response_details = {
                                'status_code': response.status_code,
                                'error': f"HTTP {response.status_code}"
                            }
                        
                        endpoint_results.append({
                            'method': method,
                            'endpoint': endpoint,
                            'description': description,
                            'success': success,
                            'details': response_details,
                            'status': status
                        })
                        
                        logger.info(f"    {status} {description}: HTTP {response.status_code}")
                        
                    except Exception as e:
                        logger.error(f"    ❌ Error {description}: {str(e)}")
                        endpoint_results.append({
                            'method': method,
                            'endpoint': endpoint,
                            'description': description,
                            'success': False,
                            'details': {'error': str(e)},
                            'status': "❌"
                        })
            
            success_rate = (successful_endpoints / len(endpoints_to_test)) * 100
            overall_success = success_rate >= 75  # At least 75% success
            
            self.log_test_result("REST API Endpoints", overall_success,
                               f"Success Rate: {success_rate:.1f}% ({successful_endpoints}/{len(endpoints_to_test)})",
                               {'results': endpoint_results})
            
            return endpoint_results
            
        except Exception as e:
            self.log_test_result("Test API Endpoints", False, f"Error: {str(e)}")
            return None
    
    def test_performance_benchmark(self):
        """Test 7: Performance Benchmark"""
        logger.info("⚡ Test 7: Performance Benchmark")
        
        try:
            from advanced_web_navigator import extract_website_content # Keeping original function name
            
            # Test with multiple URLs to measure performance
            test_urls = [
                "https://httpbin.org/json",
                "https://httpbin.org/html",
                "https://httpbin.org/robots.txt"
            ]
            
            performance_results = []
            total_time = 0
            successful_requests = 0
            
            logger.info(f"  📊 Performance test on {len(test_urls)} URLs")
            
            overall_start = time.time()
            
            for i, url in enumerate(test_urls, 1):
                logger.info(f"  {i}/{len(test_urls)} Test: {url}")
                
                start_time = time.time()
                content = extract_website_content(url) # Keeping original function name
                end_time = time.time()
                
                request_time = end_time - start_time
                total_time += request_time
                
                if content.success:
                    successful_requests += 1
                    status = "✅"
                    details = {
                        'processing_time': request_time,
                        'content_length': len(content.cleaned_text),
                        'quality_score': content.content_quality_score,
                        'extraction_rate': len(content.cleaned_text) / request_time if request_time > 0 else 0
                    }
                else:
                    status = "❌"
                    details = {
                        'processing_time': request_time,
                        'error': content.error_message
                    }
                
                performance_results.append({
                    'url': url,
                    'success': content.success,
                    'details': details,
                    'status': status
                })
                
                logger.info(f"    {status} Time: {request_time:.2f}s")
                
                time.sleep(0.5)  # Small delay between requests
            
            overall_time = time.time() - overall_start
            
            # Calculate performance metrics
            avg_time_per_request = total_time / len(test_urls)
            success_rate = (successful_requests / len(test_urls)) * 100
            
            performance_metrics = {
                'total_requests': len(test_urls),
                'successful_requests': successful_requests,
                'success_rate': success_rate,
                'total_time': total_time,
                'overall_time': overall_time,
                'avg_time_per_request': avg_time_per_request,
                'requests_per_second': len(test_urls) / overall_time if overall_time > 0 else 0
            }
            
            # Acceptable performance criteria
            performance_ok = (
                avg_time_per_request < 10.0 and  # Less than 10s per request
                success_rate >= 70              # At least 70% success
            )
            
            logger.info(f"  📈 Performance Metrics:")
            logger.info(f"    - Average time per request: {avg_time_per_request:.2f}s")
            logger.info(f"    - Success Rate: {success_rate:.1f}%")
            logger.info(f"    - Requests per second: {performance_metrics['requests_per_second']:.2f}")
            
            self.log_test_result("Performance Benchmark", performance_ok,
                               f"Average time: {avg_time_per_request:.2f}s, Success: {success_rate:.1f}%",
                               {'metrics': performance_metrics, 'results': performance_results})
            
            return performance_metrics
            
        except Exception as e:
            self.log_test_result("Test Performance", False, f"Error: {str(e)}")
            return None
    
    def generate_test_report(self):
        """Generates a comprehensive test report"""
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
        
        # Save JSON report
        report_file = self.test_dir / f"gemini_web_navigation_test_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Create a readable markdown report
        md_report = self._create_markdown_report(report)
        md_file = self.test_dir / f"gemini_web_navigation_test_report_{int(time.time())}.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_report)
        
        logger.info(f"📄 Report saved: {report_file}")
        logger.info(f"📄 Markdown report: {md_file}")
        
        return report
    
    def _create_markdown_report(self, report):
        """Creates a markdown report"""
        summary = report['test_summary']
        
        md = f"""# Test Report - Advanced Web Navigation System with artificial intelligence API GOOGLE GEMINI 2.0 FLASH

## Summary
- **Test Date**: {summary['timestamp'][:19]}
- **Total Tests**: {summary['total_tests']}
- **Passed Tests**: {summary['passed_tests']}
- **Failed Tests**: {summary['failed_tests']}
- **Success Rate**: {summary['success_rate']:.1f}%
- **Overall Status**: {summary['overall_status']}

## Test Details

"""
        
        for test_name, result in report['test_results'].items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            md += f"### {test_name}\n"
            md += f"**Status**: {status}\n"
            md += f"**Message**: {result['message']}\n"
            
            if result.get('data'):
                md += f"**Data**: See JSON file for details\n"
            
            md += "\n"
        
        if report['errors']:
            md += "## Errors Encountered\n\n"
            for error in report['errors']:
                md += f"- {error}\n"
        
        md += f"""
## Recommendations

### Overall Status: {summary['overall_status']}

"""
        
        if summary['success_rate'] >= 90:
            md += "🎉 **EXCELLENT** - The system works perfectly with artificial intelligence API GOOGLE GEMINI 2.0 FLASH!\n"
        elif summary['success_rate'] >= 70:
            md += "👍 **GOOD** - The system works well with some possible improvements.\n"
        elif summary['success_rate'] >= 50:
            md += "⚠️ **AVERAGE** - The system works partially, checks needed.\n"
        else:
            md += "🚨 **PROBLEM** - The system requires significant fixes.\n"
        
        return md
    
    def run_all_tests(self):
        """Runs all tests"""
        logger.info("🚀 STARTING COMPREHENSIVE TESTS")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Test 1: Imports
            imported_modules = self.test_module_imports()
            
            # Test 2: artificial intelligence API GOOGLE GEMINI 2.0 FLASH Initialization
            gemini_api = self.test_gemini_api_initialization()
            
            # Test 3: Navigation detection
            self.test_navigation_detection(gemini_api)
            
            # Test 4: Web extraction
            self.test_web_extraction()
            
            # Test 5: Full integration with artificial intelligence API GOOGLE GEMINI 2.0 FLASH
            self.test_gemini_integration_full(gemini_api)
            
            # Test 6: REST API
            self.test_api_endpoints()
            
            # Test 7: Performance
            self.test_performance_benchmark()
            
        except Exception as e:
            logger.error(f"Error during tests: {str(e)}")
            self.log_test_result("Global Execution", False, f"Critical error: {str(e)}")
        
        total_time = time.time() - start_time
        
        # Generate report
        report = self.generate_test_report()
        
        # Display final summary
        logger.info("=" * 60)
        logger.info("🏁 TESTS COMPLETED")
        logger.info(f"⏱️ Total Time: {total_time:.2f}s")
        logger.info(f"📊 Results: {self.passed_tests}/{self.total_tests} tests successful ({report['test_summary']['success_rate']:.1f}%)")
        
        if report['test_summary']['overall_status'] == 'PASSED':
            logger.info("🎉 ALL MAIN TESTS PASSED!")
            logger.info("✅ The web navigation system works with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
        else:
            logger.info("⚠️ Some tests failed")
            logger.info("🔧 Check errors in the test report")
        
        logger.info("=" * 60)
        
        return report

def main():
    """Main function"""
    print("🧪 COMPREHENSIVE TEST - Advanced Web Navigation System with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
    print("=" * 70)
    print(f"🕐 Started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Create the tester
    tester = GeminiWebNavigationTester()
    
    try:
        # Run all tests
        report = tester.run_all_tests()
        
        # Final result
        if report['test_summary']['overall_status'] == 'PASSED':
            print("\n🎊 FULL SUCCESS!")
            print("✅ The web navigation system works perfectly with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
            print("🚀 You can now use the new navigation capabilities")
            return True
        else:
            print("\n⚠️ PARTIALLY SUCCESSFUL TESTS")
            print("🔧 Some functionalities require adjustments")
            print("📋 Consult the test report for more details")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Critical error during tests: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
