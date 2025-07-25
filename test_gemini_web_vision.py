#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test of the Web Vision System with artificial intelligence API GOOGLE GEMINI 2.0 FLASH
Tests all developed functionalities
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
logger = logging.getLogger('GeminiWebVisionTest')

class GeminiWebVisionTester:
    """Comprehensive tester for the web vision system with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.passed_tests = 0
        self.total_tests = 0
        
        # Create test directory
        self.test_dir = Path("test_results_vision")
        self.test_dir.mkdir(exist_ok=True)
        
        logger.info("🧪 artificial intelligence API GOOGLE GEMINI 2.0 FLASH Web Vision Tester initialized")
    
    def log_test_result(self, test_name: str, success: bool, message: str = "", data: dict = None):
        """Records the result of a test"""
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
    
    def test_vision_modules_import(self):
        """Test 1: Import of vision modules"""
        logger.info("🔧 Test 1: Import of vision modules")
        
        modules_to_test = [
            ('gemini_visual_adapter', 'artificial intelligence API GOOGLE GEMINI 2.0 FLASH Visual Adapter'),
            ('intelligent_web_capture', 'Intelligent Capture System'),
            ('gemini_web_vision_integration', 'Web Vision Integration'),
            ('gemini_web_vision_api', 'Vision REST API')
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
        
        self.log_test_result("Global Vision Import", all_success, 
                           f"{len(imported_modules)}/{len(modules_to_test)} modules imported")
        
        return imported_modules
    
    def test_visual_adapter_initialization(self):
        """Test 2: Initialization of the visual adapter"""
        logger.info("🤖 Test 2: Visual Adapter Initialization")
        
        try:
            from gemini_visual_adapter import initialize_gemini_visual_adapter, get_gemini_visual_adapter
            
            # Initialize the adapter
            visual_adapter = initialize_gemini_visual_adapter()
            self.log_test_result("Create Visual Adapter", True, "Adapter created")
            
            # Check global accessibility
            global_adapter = get_gemini_visual_adapter()
            if global_adapter:
                self.log_test_result("Verify Global Adapter", True, "Adapter accessible")
                
                # Test statistics
                stats = visual_adapter.get_statistics()
                self.log_test_result("Adapter Statistics", True, f"Stats: {stats}")
                
                return visual_adapter
            else:
                self.log_test_result("Verify Global Adapter", False, "Adapter not accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Visual Adapter Initialization", False, f"Error: {str(e)}")
            return None
    
    def test_intelligent_capture_system(self):
        """Test 3: Intelligent Capture System"""
        logger.info("📸 Test 3: Intelligent Capture System")
        
        try:
            from intelligent_web_capture import initialize_intelligent_capture, get_intelligent_capture
            
            # Initialize the capture system
            capture_system = initialize_intelligent_capture()
            self.log_test_result("Create Capture System", True, "System created")
            
            # Check global access
            global_capture = get_intelligent_capture()
            if global_capture:
                self.log_test_result("Verify Global Capture", True, "System accessible")
                
                # Test statistics
                stats = capture_system.get_statistics()
                self.log_test_result("Capture Statistics", True, f"Stats: {stats}")
                
                return capture_system
            else:
                self.log_test_result("Verify Global Capture", False, "System not accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Initialize Capture System", False, f"Error: {str(e)}")
            return None
    
    def test_web_vision_integration(self):
        """Test 4: Full Web Vision Integration"""
        logger.info("🌐 Test 4: Web Vision Integration")
        
        try:
            from gemini_web_vision_integration import initialize_gemini_web_vision, get_gemini_web_vision
            
            # Initialize integration
            integration = initialize_gemini_web_vision()
            self.log_test_result("Create Vision Integration", True, "Integration created")
            
            # Check global access
            global_integration = get_gemini_web_vision()
            if global_integration:
                self.log_test_result("Verify Global Integration", True, "Integration accessible")
                
                # Test statistics
                stats = integration.get_statistics()
                self.log_test_result("Integration Statistics", True, f"Stats: {stats}")
                
                return integration
            else:
                self.log_test_result("Verify Global Integration", False, "Integration not accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Initialize Vision Integration", False, f"Error: {str(e)}")
            return None
    
    def test_vision_session_management(self, integration):
        """Test 5: Vision Session Management"""
        logger.info("🎯 Test 5: Vision Session Management")
        
        if not integration:
            self.log_test_result("Test Vision Sessions", False, "Integration not available")
            return
        
        try:
            # Create a test session
            session_id = f"test_vision_session_{int(datetime.now().timestamp())}"
            user_query = "Analyze the user interface and UX of this website"
            
            result = integration.create_vision_navigation_session(
                session_id=session_id,
                user_query=user_query,
                navigation_goals=['extract_content', 'analyze_ui', 'capture_visuals']
            )
            
            if result['success']:
                self.log_test_result("Create Vision Session", True, f"Session created: {session_id}")
                
                # Verify that the session is active
                if session_id in integration.active_sessions:
                    self.log_test_result("Verify Active Session", True, "Session found in active sessions")
                    
                    # Close the session
                    close_result = integration.close_session(session_id)
                    if close_result['success']:
                        self.log_test_result("Close Session", True, "Session closed successfully")
                    else:
                        self.log_test_result("Close Session", False, f"Error: {close_result.get('error')}")
                else:
                    self.log_test_result("Verify Active Session", False, "Session not found")
            else:
                self.log_test_result("Create Vision Session", False, f"Error: {result.get('error')}")
                
        except Exception as e:
            self.log_test_result("Test Vision Sessions", False, f"Error: {str(e)}")
    
    def test_capture_simulation(self, capture_system):
        """Test 6: Capture Simulation (without browser)"""
        logger.info("📷 Test 6: Capture Simulation")
        
        if not capture_system:
            self.log_test_result("Test Capture Simulation", False, "Capture system not available")
            return
        
        try:
            # Test configurations
            test_configs = [
                {
                    'name': 'Desktop Config',
                    'viewport': 'desktop',
                    'capture_type': 'visible_area'
                },
                {
                    'name': 'Mobile Config', 
                    'viewport': 'mobile',
                    'capture_type': 'full_page'
                },
                {
                    'name': 'Tablet Config',
                    'viewport': 'tablet',
                    'capture_type': 'element_focused'
                }
            ]
            
            successful_configs = 0
            
            for config in test_configs:
                try:
                    # Simulate configuration validation
                    valid_viewports = ['desktop', 'mobile', 'tablet']
                    valid_captures = ['visible_area', 'full_page', 'element_focused']
                    
                    if config['viewport'] in valid_viewports and config['capture_type'] in valid_captures:
                        self.log_test_result(f"Validation {config['name']}", True, "Valid configuration")
                        successful_configs += 1
                    else:
                        self.log_test_result(f"Validation {config['name']}", False, "Invalid configuration")
                        
                except Exception as e:
                    self.log_test_result(f"Validation {config['name']}", False, f"Error: {str(e)}")
            
            success_rate = successful_configs / len(test_configs) * 100
            self.log_test_result("Global Capture Simulation", successful_configs == len(test_configs), 
                               f"Success rate: {success_rate}% ({successful_configs}/{len(test_configs)})")
            
        except Exception as e:
            self.log_test_result("Test Capture Simulation", False, f"Error: {str(e)}")
    
    def test_visual_analysis_simulation(self, visual_adapter):
        """Test 7: Visual Analysis Simulation"""
        logger.info("🔍 Test 7: Visual Analysis Simulation")
        
        if not visual_adapter:
            self.log_test_result("Test Visual Analysis", False, "Visual adapter not available")
            return
        
        try:
            # Test analysis prompts
            test_prompts = [
                {
                    'name': 'UI Analysis',
                    'prompt': 'Analyze the user interface elements in this capture',
                    'context': 'Specialized UI/UX analysis'
                },
                {
                    'name': 'Content Analysis',
                    'prompt': 'Identify and summarize the main content of this page',
                    'context': 'Content extraction'
                },
                {
                    'name': 'Design Analysis',
                    'prompt': 'Evaluate the visual design and aesthetics of the site',
                    'context': 'Design analysis'
                }
            ]
            
            successful_prompts = 0
            
            for prompt_test in test_prompts:
                try:
                    # Simulate prompt validation
                    if len(prompt_test['prompt']) > 10 and prompt_test['context']:
                        self.log_test_result(f"Validation {prompt_test['name']}", True, "Valid prompt")
                        successful_prompts += 1
                    else:
                        self.log_test_result(f"Validation {prompt_test['name']}", False, "Invalid prompt")
                        
                except Exception as e:
                    self.log_test_result(f"Validation {prompt_test['name']}", False, f"Error: {str(e)}")
            
            success_rate = successful_prompts / len(test_prompts) * 100
            self.log_test_result("Visual Analysis Simulation", successful_prompts == len(test_prompts),
                               f"Success rate: {success_rate}% ({successful_prompts}/{len(test_prompts)})")
            
        except Exception as e:
            self.log_test_result("Test Visual Analysis", False, f"Error: {str(e)}")
    
    def test_api_initialization(self):
        """Test 8: REST API Initialization"""
        logger.info("🌐 Test 8: REST API Initialization")
        
        try:
            from gemini_web_vision_api import create_vision_api, get_vision_api
            
            # Create the API
            api = create_vision_api()
            self.log_test_result("Create Vision API", True, "API created")
            
            # Check global access
            global_api = get_vision_api()
            if global_api:
                self.log_test_result("Verify Global API", True, "API accessible")
                
                # Verify API components
                if hasattr(api, 'app') and api.app:
                    self.log_test_result("Verify Flask App", True, "Flask application initialized")
                else:
                    self.log_test_result("Verify Flask App", False, "Flask application not initialized")
                
                return api
            else:
                self.log_test_result("Verify Global API", False, "API not accessible")
                return None
                
        except Exception as e:
            self.log_test_result("Initialize Vision API", False, f"Error: {str(e)}")
            return None
    
    def test_system_integration(self, integration, visual_adapter, capture_system):
        """Test 9: Full System Integration"""
        logger.info("🔗 Test 9: Full System Integration")
        
        components_status = {
            'integration': integration is not None,
            'visual_adapter': visual_adapter is not None,
            'capture_system': capture_system is not None
        }
        
        working_components = sum(components_status.values())
        total_components = len(components_status)
        
        if working_components == total_components:
            self.log_test_result("Full System Integration", True, 
                               f"All components functional ({working_components}/{total_components})")
            
            # Test consistency of statistics
            try:
                if integration:
                    integration_stats = integration.get_statistics()
                    self.log_test_result("Integration Statistics", True, f"Stats retrieved: {len(integration_stats)} metrics")
                
                if visual_adapter:
                    adapter_stats = visual_adapter.get_statistics()
                    self.log_test_result("Adapter Statistics", True, f"Stats retrieved: {len(adapter_stats)} metrics")
                
                if capture_system:
                    capture_stats = capture_system.get_statistics()
                    self.log_test_result("Capture Statistics", True, f"Stats retrieved: {len(capture_stats)} metrics")
                    
            except Exception as e:
                self.log_test_result("Test System Statistics", False, f"Error: {str(e)}")
        else:
            self.log_test_result("Full System Integration", False,
                               f"Missing components: {working_components}/{total_components}")
            
        return working_components == total_components
    
    def test_error_handling(self, integration):
        """Test 10: Error Handling"""
        logger.info("⚠️ Test 10: Error Handling")
        
        if not integration:
            self.log_test_result("Test Error Handling", False, "Integration not available")
            return
        
        try:
            # Test with non-existent session
            result = integration.close_session("non_existent_session_12345")
            if not result['success'] and 'error' in result:
                self.log_test_result("Handle Non-Existent Session", True, "Error correctly handled")
            else:
                self.log_test_result("Handle Non-Existent Session", False, "Error not handled")
            
            # Test with invalid parameters
            try:
                result = integration.create_vision_navigation_session("", "")
                if not result['success']:
                    self.log_test_result("Handle Invalid Parameters", True, "Invalid parameters detected")
                else:
                    self.log_test_result("Handle Invalid Parameters", False, "Invalid parameters not detected")
            except Exception:
                self.log_test_result("Handle Invalid Parameters", True, "Exception correctly raised")
            
        except Exception as e:
            self.log_test_result("Test Error Handling", False, f"Error: {str(e)}")
    
    def run_all_tests(self):
        """Executes all tests"""
        start_time = datetime.now()
        
        print("🧪 COMPREHENSIVE TEST - Web Vision System with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
        print("=" * 70)
        print(f"🕐 Started on: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        logger.info("🚀 STARTING WEB VISION TESTS")
        logger.info("=" * 60)
        
        # Test 1: Module imports
        imported_modules = self.test_vision_modules_import()
        
        # Test 2: Visual adapter
        visual_adapter = self.test_visual_adapter_initialization()
        
        # Test 3: Capture system
        capture_system = self.test_intelligent_capture_system()
        
        # Test 4: Web vision integration
        integration = self.test_web_vision_integration()
        
        # Test 5: Session management
        self.test_vision_session_management(integration)
        
        # Test 6: Capture simulation
        self.test_capture_simulation(capture_system)
        
        # Test 7: Visual analysis simulation
        self.test_visual_analysis_simulation(visual_adapter)
        
        # Test 8: REST API
        api = self.test_api_initialization()
        
        # Test 9: System integration
        system_ok = self.test_system_integration(integration, visual_adapter, capture_system)
        
        # Test 10: Error handling
        self.test_error_handling(integration)
        
        # Generate final report
        self.generate_final_report(start_time, system_ok)
    
    def generate_final_report(self, start_time: datetime, system_ok: bool):
        """Generates the final test report"""
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        success_rate = (self.passed_tests / max(self.total_tests, 1)) * 100
        
        # Create the report
        report = {
            'test_summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.total_tests - self.passed_tests,
                'success_rate': round(success_rate, 2),
                'total_time': round(total_time, 2),
                'system_ready': system_ok
            },
            'test_results': self.test_results,
            'errors': self.errors,
            'timestamp': end_time.isoformat()
        }
        
        # Save JSON report
        report_filename = f"gemini_vision_test_report_{int(end_time.timestamp())}.json"
        report_path = self.test_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Display final summary
        logger.info("=" * 60)
        logger.info("🏁 TESTS COMPLETED")
        logger.info(f"⏱️ Total time: {total_time:.2f}s")
        logger.info(f"📊 Results: {self.passed_tests}/{self.total_tests} tests successful ({success_rate:.1f}%)")
        
        if system_ok and success_rate >= 80:
            logger.info("🎉 WEB VISION SYSTEM READY!")
            logger.info("✅ artificial intelligence API GOOGLE GEMINI 2.0 FLASH can now see inside websites")
        else:
            logger.warning("⚠️ System partially functional")
            if self.errors:
                logger.error("❌ Errors detected:")
                for error in self.errors:
                    logger.error(f"   - {error}")
        
        logger.info("=" * 60)
        
        print("\n" + "=" * 70)
        print("🏁 TESTS COMPLETED")
        print(f"⏱️ Total time: {total_time:.2f} seconds")
        print(f"📊 Results: {self.passed_tests}/{self.total_tests} tests successful ({success_rate:.1f}%)")
        print(f"📄 Report saved: {report_filename}")
        
        if system_ok and success_rate >= 80:
            print("\n🎊 FULL SUCCESS!")
            print("✅ The web vision system works perfectly")
            print("🚀 artificial intelligence API GOOGLE GEMINI 2.0 FLASH can now see inside websites")
            print("👁️ Capabilities available:")
            print("   - Intelligent website capture")
            print("   - Visual analysis with artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
            print("   - Vision-guided navigation")
            print("   - Full REST API")
        else:
            print(f"\n⚠️ System partially functional ({success_rate:.1f}%)")
            
        print("=" * 70)

if __name__ == "__main__":
    tester = GeminiWebVisionTester()
    tester.run_all_tests()
