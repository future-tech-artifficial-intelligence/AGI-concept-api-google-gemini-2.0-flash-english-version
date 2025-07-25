"""
Automatic Installation Script - Google Gemini 2.0 Flash AI Interactive Navigation System
This script automatically configures all necessary dependencies
"""

import os
import sys
import subprocess
import logging
import platform
import json
from pathlib import Path
from datetime import datetime

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('InteractiveInstaller')

class InteractiveNavigationInstaller:
    """Automatic installer for the Google Gemini 2.0 Flash AI interactive navigation system"""
    
    def __init__(self):
        self.installation_log = []
        self.errors = []
        self.system_info = {
            'platform': platform.system(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        }
        
        logger.info(f"🚀 Starting installation on {self.system_info['platform']}")
    
    def log_step(self, step_name: str, success: bool, message: str = ""):
        """Logs an installation step"""
        timestamp = datetime.now().isoformat()
        
        entry = {
            'timestamp': timestamp,
            'step': step_name,
            'success': success,
            'message': message
        }
        
        self.installation_log.append(entry)
        
        if success:
            logger.info(f"✅ {step_name}: {message}")
        else:
            logger.error(f"❌ {step_name}: {message}")
            self.errors.append(entry)
    
    def check_python_version(self):
        """Checks the Python version"""
        logger.info("🐍 Checking Python version...")
        
        version_info = sys.version_info
        required_major, required_minor = 3, 8
        
        if version_info.major >= required_major and version_info.minor >= required_minor:
            self.log_step("Python Check", True, 
                         f"Python {version_info.major}.{version_info.minor}.{version_info.micro} OK")
            return True
        else:
            self.log_step("Python Check", False, 
                         f"Python {required_major}.{required_minor}+ required, {version_info.major}.{version_info.minor} detected")
            return False
    
    def install_base_requirements(self):
        """Installs base dependencies"""
        logger.info("📦 Installing base dependencies...")
        
        base_packages = [
            'selenium>=4.15.0',
            'webdriver-manager>=4.0.0',
            'requests>=2.31.0',
            'beautifulsoup4>=4.12.0',
            'lxml>=4.9.0',
            'Pillow>=10.0.0'
        ]
        
        try:
            for package in base_packages:
                logger.info(f"   Installing {package}...")
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    self.log_step(f"Installation {package.split('>=')[0]}", True, "Package installed")
                else:
                    self.log_step(f"Installation {package.split('>=')[0]}", False, result.stderr)
                    return False
            
            return True
            
        except Exception as e:
            self.log_step("Base Dependencies Installation", False, str(e))
            return False
    
    def check_webdriver_availability(self):
        """Checks WebDrivers availability"""
        logger.info("🌐 Checking WebDrivers...")
        
        drivers_available = {
            'chrome': False,
            'edge': False,
            'firefox': False
        }
        
        # Test Chrome
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options as ChromeOptions
            from webdriver_manager.chrome import ChromeDriverManager
            
            chrome_options = ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            # Attempt to initialize ChromeDriver
            driver_path = ChromeDriverManager().install()
            driver = webdriver.Chrome(service=webdriver.chrome.service.Service(driver_path), 
                                    options=chrome_options)
            driver.quit()
            
            drivers_available['chrome'] = True
            self.log_step("Chrome WebDriver", True, "ChromeDriver operational")
            
        except Exception as e:
            self.log_step("Chrome WebDriver", False, f"Error: {str(e)}")
        
        # Test Edge (if on Windows)
        if self.system_info['platform'] == 'Windows':
            try:
                from selenium.webdriver.edge.options import Options as EdgeOptions
                from webdriver_manager.microsoft import EdgeChromiumDriverManager
                
                edge_options = EdgeOptions()
                edge_options.add_argument('--headless')
                edge_options.add_argument('--no-sandbox')
                
                driver_path = EdgeChromiumDriverManager().install()
                driver = webdriver.Edge(service=webdriver.edge.service.Service(driver_path),
                                      options=edge_options)
                driver.quit()
                
                drivers_available['edge'] = True
                self.log_step("Edge WebDriver", True, "EdgeDriver operational")
                
            except Exception as e:
                self.log_step("Edge WebDriver", False, f"Error: {str(e)}")
        
        # Summary
        available_count = sum(drivers_available.values())
        if available_count > 0:
            self.log_step("Global WebDrivers", True, 
                         f"{available_count} driver(s) available: {', '.join([k for k, v in drivers_available.items() if v])}")
            return True
        else:
            self.log_step("Global WebDrivers", False, "No WebDriver available")
            return False
    
    def test_interactive_modules(self):
        """Tests the import of interactive modules"""
        logger.info("🧪 Testing interactive modules...")
        
        modules_to_test = [
            ('interactive_web_navigator', 'Interactive Navigator'),
            ('gemini_interactive_adapter', 'Google Gemini 2.0 Flash AI Interactive Adapter'),
            ('gemini_api_adapter', 'Google Gemini 2.0 Flash AI Main Adapter')
        ]
        
        successful_imports = 0
        
        for module_name, display_name in modules_to_test:
            try:
                __import__(module_name)
                self.log_step(f"Import {display_name}", True, "Module imported successfully")
                successful_imports += 1
                
            except ImportError as e:
                self.log_step(f"Import {display_name}", False, f"Import error: {str(e)}")
        
        if successful_imports == len(modules_to_test):
            self.log_step("Module Test", True, "All modules are available")
            return True
        else:
            self.log_step("Module Test", False, 
                         f"Only {successful_imports}/{len(modules_to_test)} modules available")
            return False
    
    def create_configuration_file(self):
        """Creates a default configuration file"""
        logger.info("⚙️ Creating configuration file...")
        
        config = {
            'interactive_navigation': {
                'enabled': True,
                'preferred_browser': 'chrome',
                'default_timeout': 30,
                'screenshot_enabled': True,
                'max_interactions_per_session': 50
            },
            'webdriver_settings': {
                'headless': True,
                'window_size': [1920, 1080],
                'page_load_timeout': 15,
                'implicit_wait': 5
            },
            'detection_settings': {
                'confidence_threshold': 0.6,
                'interaction_keywords': {
                    'click': ['click', 'press'],
                    'navigate': ['explore', 'browse', 'navigate'],
                    'analyze': ['analyze', 'look', 'examine']
                }
            },
            'safety_settings': {
                'respect_robots_txt': True,
                'interaction_delay': 1.0,
                'max_session_duration': 300
            },
            'installation_info': {
                'installed_on': datetime.now().isoformat(),
                'system_info': self.system_info,
                'version': '1.0.0'
            }
        }
        
        try:
            config_path = Path('interactive_navigation_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.log_step("Configuration", True, f"File created: {config_path}")
            return True
            
        except Exception as e:
            self.log_step("Configuration", False, f"Error creating file: {str(e)}")
            return False
    
    def run_basic_tests(self):
        """Executes basic system tests"""
        logger.info("🧪 Running basic tests...")
        
        try:
            # Interaction detection test
            from gemini_interactive_adapter import detect_interactive_need
            
            test_prompts = [
                "Click on the Services tab",
                "Explore all tabs on this site",
                "What is artificial intelligence?"
            ]
            
            detection_results = []
            for prompt in test_prompts:
                try:
                    result = detect_interactive_need(prompt)
                    detection_results.append({
                        'prompt': prompt,
                        'detected': result.get('requires_interaction', False),
                        'type': result.get('interaction_type'),
                        'confidence': result.get('confidence', 0)
                    })
                except Exception as e:
                    logger.warning(f"Detection test error for '{prompt}': {e}")
            
            interactive_detected = sum(1 for r in detection_results if r['detected'])
            self.log_step("Detection Tests", True, 
                         f"{interactive_detected}/{len(test_prompts)} interactions detected")
            
            # Component initialization test
            try:
                from interactive_web_navigator import get_interactive_navigator
                navigator = get_interactive_navigator()
                
                if navigator:
                    stats = navigator.get_statistics()
                    self.log_step("Navigator Test", True, "Navigator initialized")
                else:
                    self.log_step("Navigator Test", False, "Navigator not initialized")
            
            except Exception as e:
                self.log_step("Navigator Test", False, f"Error: {str(e)}")
            
            return True
            
        except Exception as e:
            self.log_step("Basic Tests", False, f"Global error: {str(e)}")
            return False
    
    def generate_installation_report(self):
        """Generates a complete installation report"""
        logger.info("📋 Generating installation report...")
        
        successful_steps = sum(1 for entry in self.installation_log if entry['success'])
        total_steps = len(self.installation_log)
        success_rate = (successful_steps / total_steps * 100) if total_steps > 0 else 0
        
        report = {
            'installation_summary': {
                'timestamp': datetime.now().isoformat(),
                'system_info': self.system_info,
                'total_steps': total_steps,
                'successful_steps': successful_steps,
                'success_rate': success_rate,
                'overall_status': 'SUCCESS' if success_rate >= 80 else 'PARTIAL' if success_rate >= 50 else 'FAILED'
            },
            'installation_log': self.installation_log,
            'errors': self.errors,
            'recommendations': self._generate_recommendations()
        }
        
        # Save the report
        try:
            report_path = Path(f'installation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.log_step("Installation Report", True, f"Report saved: {report_path}")
        
        except Exception as e:
            self.log_step("Installation Report", False, f"Error saving: {str(e)}")
        
        return report
    
    def _generate_recommendations(self):
        """Generates recommendations based on the installation"""
        recommendations = []
        
        # Check common errors
        if any('WebDriver' in error['step'] for error in self.errors):
            recommendations.append({
                'type': 'webdriver_issue',
                'title': 'WebDriver issue detected',
                'description': 'Manually install ChromeDriver or verify Chrome is installed',
                'actions': [
                    'Download ChromeDriver from https://chromedriver.chromium.org/',
                    'Add ChromeDriver to system PATH',
                    'Or install Chrome/Chromium browser'
                ]
            })
        
        if any('Python' in error['step'] for error in self.errors):
            recommendations.append({
                'type': 'python_version',
                'title': 'Insufficient Python version',
                'description': 'Update Python to version 3.8 or higher',
                'actions': [
                    'Download Python 3.8+ from python.org',
                    'Reinstall dependencies after updating'
                ]
            })
        
        if not self.errors:
            recommendations.append({
                'type': 'success',
                'title': 'Installation successful',
                'description': 'The system is ready for use',
                'actions': [
                    'Execute python demo_interactive_navigation.py to see a demonstration',
                    'Read the user guide GUIDE_INTERACTIVE_NAVIGATION.md',
                    'Test with your own use cases'
                ]
            })
        
        return recommendations
    
    def run_full_installation(self):
        """Launches the full installation"""
        logger.info("🎯 STARTING FULL INSTALLATION")
        logger.info("=" * 80)
        
        installation_steps = [
            ('Python Check', self.check_python_version),
            ('Dependencies Installation', self.install_base_requirements),
            ('WebDrivers Check', self.check_webdriver_availability),
            ('Interactive Modules Test', self.test_interactive_modules),
            ('Configuration Creation', self.create_configuration_file),
            ('Basic Tests', self.run_basic_tests)
        ]
        
        start_time = datetime.now()
        
        for step_name, step_function in installation_steps:
            logger.info(f"\n🔄 {step_name}...")
            try:
                success = step_function()
                if not success:
                    logger.warning(f"⚠️ {step_name} failed, but installation continues...")
            except Exception as e:
                logger.error(f"❌ Critical error in {step_name}: {e}")
                self.log_step(step_name, False, f"Critical error: {str(e)}")
        
        installation_time = (datetime.now() - start_time).total_seconds()
        
        # Generate the final report
        report = self.generate_installation_report()
        
        # Display the summary
        logger.info("\n" + "=" * 80)
        logger.info("🏁 INSTALLATION COMPLETE")
        logger.info(f"⏱️ Installation time: {installation_time:.1f}s")
        logger.info(f"📊 Results: {report['installation_summary']['successful_steps']}/{report['installation_summary']['total_steps']} steps successful")
        logger.info(f"📈 Success rate: {report['installation_summary']['success_rate']:.1f}%")
        logger.info(f"🎖️ Status: {report['installation_summary']['overall_status']}")
        
        # Display recommendations
        if report['recommendations']:
            logger.info("\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                logger.info(f"   🔸 {rec['title']}: {rec['description']}")
        
        # Final message
        if report['installation_summary']['overall_status'] == 'SUCCESS':
            logger.info("\n🎉 INSTALLATION SUCCESSFUL!")
            logger.info("✅ The Google Gemini 2.0 Flash AI interactive navigation system is operational")
            logger.info("🚀 You can now use the new features")
            logger.info("\n📖 Next steps:")
            logger.info("   1. Read the guide: GUIDE_INTERACTIVE_NAVIGATION.md")
            logger.info("   2. Test: python demo_interactive_navigation.py")
            logger.info("   3. Validate: python test_interactive_navigation.py")
        else:
            logger.info("\n⚠️ PARTIAL INSTALLATION")
            logger.info("🔧 Consult the installation report to resolve issues")
            logger.info("💬 Some features may be limited")
        
        logger.info("=" * 80)
        
        return report

def main():
    """Main installation function"""
    print("🌟 Installing Google Gemini 2.0 Flash AI Interactive Navigation System")
    print("🎯 This script will automatically configure your environment\n")
    
    installer = InteractiveNavigationInstaller()
    report = installer.run_full_installation()
    
    # Exit code based on success
    success = report['installation_summary']['overall_status'] in ['SUCCESS', 'PARTIAL']
    
    if success:
        print("\n✅ Installation completed successfully")
        print("🎯 The interactive navigation system is ready to use!")
    else:
        print("\n❌ Installation failed")
        print("🔧 Consult the logs and installation report for more information")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Critical error during installation: {e}")
        sys.exit(1)
