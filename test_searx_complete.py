#!/usr/bin/env python3
"""
Comprehensive test script for the intelligent Searx system v2
Verifies all components and dependencies with advanced error handling
"""

import sys
import logging
import traceback
import time
from pathlib import Path

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('SearxSystemTest')

class SearxSystemTester:
    """Comprehensive tester for the Searx system"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def test_imports(self):
        """Tests imports of main modules"""
        print("🔍 IMPORTS TEST")
        print("=" * 30)
        
        tests = [
            ("requests", "HTTP Requests"),
            ("psutil", "Process Management"),
            ("bs4", "BeautifulSoup for HTML parsing"),
            ("selenium", "Web Automation"),
            ("PIL", "Pillow for images"),
            ("json", "Standard JSON"),
            ("socket", "Network Sockets"),
            ("subprocess", "System Processes"),
            ("platform", "Platform Information"),
            ("docker", "Python Docker Client")
        ]
        
        success_count = 0
        for module, description in tests:
            try:
                __import__(module)
                print(f"✅ {module:12} - {description}")
                success_count += 1
            except ImportError as e:
                print(f"❌ {module:12} - {description} - ERROR: {e}")
        
        print(f"\n📊 Result: {success_count}/{len(tests)} modules available")
        self.results['imports'] = success_count == len(tests)
        return self.results['imports']

    def test_port_manager(self):
        """Tests the port manager"""
        print("\n🔧 PORT MANAGER TEST")
        print("=" * 40)
        
        try:
            from port_manager import PortManager
            pm = PortManager()
            
            # Port detection test
            port_8080_available = pm.is_port_available(8080)
            print(f"📍 Port 8080 available: {'✅ Yes' if port_8080_available else '❌ No'}")
            
            if not port_8080_available:
                process = pm.get_process_using_port(8080)
                if process:
                    print(f"🔍 Process on 8080: {process['name']} (PID: {process['pid']})")
                    print(f"   Command: {process['cmdline'][:50]}...")
            
            # Alternative port search test
            alt_port = pm.find_available_port(8081, 5)
            if alt_port:
                print(f"🔄 Alternative port found: {alt_port}")
            else:
                print("⚠️  No alternative port found")
            
            # Configuration generation test
            config_success, port, compose_file = pm.setup_searx_with_available_port()
            if config_success:
                print(f"✅ Configuration generated: {compose_file} (port {port})")
            else:
                print("⚠️  Unable to generate configuration")
            
            print("✅ Port manager functional")
            self.results['port_manager'] = True
            return True
            
        except Exception as e:
            print(f"❌ Port manager error: {e}")
            self.results['port_manager'] = False
            return False

    def test_searx_interface(self):
        """Tests the Searx interface"""
        print("\n🔍 SEARX INTERFACE TEST")
        print("=" * 35)
        
        try:
            from searx_interface import SearxInterface
            
            # Create an instance without starting Searx
            searx = SearxInterface()
            print("✅ Searx interface created")
            
            # Verify port manager initialization
            if searx.port_manager:
                print("✅ Port manager integrated")
            else:
                print("⚠️  Port manager not initialized")
            
            # Verify visual capture initialization
            if searx.visual_capture:
                print("✅ Visual capture integrated")
            else:
                print("⚠️  Visual capture not initialized (normal if ChromeDriver missing)")
            
            print("✅ Searx interface functional")
            self.results['searx_interface'] = True
            return True
            
        except Exception as e:
            print(f"❌ Searx interface error: {e}")
            self.results['searx_interface'] = False
            return False

    def test_docker(self):
        """Tests Docker availability"""
        print("\n🐳 DOCKER TEST")
        print("=" * 20)
        
        try:
            import subprocess
            
            # Check if Docker is installed
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Docker available: {version}")
                
                # Check if Docker is running
                result = subprocess.run(['docker', 'ps'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print("✅ Docker daemon active")
                    
                    # Check Docker Compose
                    result = subprocess.run(['docker-compose', '--version'], 
                                          capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        print(f"✅ Docker Compose available: {result.stdout.strip()}")
                    else:
                        print("⚠️  Docker Compose not available")
                    
                    self.results['docker'] = True
                    return True
                else:
                    print("⚠️  Docker installed but daemon not active")
                    print("💡 Start Docker Desktop")
                    self.results['docker'] = False
                    return False
            else:
                print("❌ Docker not installed")
                print("💡 Install Docker Desktop")
                self.results['docker'] = False
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Docker not responding (timeout)")
            self.results['docker'] = False
            return False
        except FileNotFoundError:
            print("❌ Docker not found in PATH")
            self.results['docker'] = False
            return False
        except Exception as e:
            print(f"❌ Docker error: {e}")
            self.results['docker'] = False
            return False

    def test_files(self):
        """Tests the presence of necessary files"""
        print("\n📋 SYSTEM FILES TEST")
        print("=" * 32)
        
        required_files = [
            ("port_manager.py", "Port manager"),
            ("searx_interface.py", "Searx interface"),
            ("searx_smart_start.py", "Startup script"),
            ("searx_manager.bat", "Windows script"),
            ("requirements.txt", "Python dependencies")
        ]
        
        optional_files = [
            ("docker-compose.searx.yml", "Main Docker config"),
            ("docker-compose.searx-alt.yml", "Alternative Docker config"),
            ("free_port_8080.bat", "Port release script"),
            ("searx_visual_capture.py", "Visual capture")
        ]
        
        required_count = 0
        optional_count = 0
        
        print("📁 Required files:")
        for filename, description in required_files:
            if Path(filename).exists():
                print(f"✅ {filename:25} - {description}")
                required_count += 1
            else:
                print(f"❌ {filename:25} - {description} - MISSING")
        
        print("\n📁 Optional files:")
        for filename, description in optional_files:
            if Path(filename).exists():
                print(f"✅ {filename:25} - {description}")
                optional_count += 1
            else:
                print(f"⚠️  {filename:25} - {description} - Not found")
        
        total_required = len(required_files)
        total_optional = len(optional_files)
        
        print(f"\n📊 Required files: {required_count}/{total_required}")
        print(f"📊 Optional files: {optional_count}/{total_optional}")
        
        self.results['files'] = required_count == total_required
        return self.results['files']

    def test_smart_start(self):
        """Tests the smart startup script"""
        print("\n🚀 SMART STARTUP TEST")
        print("=" * 38)
        
        try:
            # Import startup module
            import searx_smart_start
            print("✅ Smart startup module imported")
            
            # Test main functions
            if hasattr(searx_smart_start, 'main'):
                print("✅ main() function available")
            
            if hasattr(searx_smart_start, 'show_status'):
                print("✅ show_status() function available")
            
            if hasattr(searx_smart_start, 'stop_all'):
                print("✅ stop_all() function available")
            
            print("✅ Smart startup script functional")
            self.results['smart_start'] = True
            return True
            
        except Exception as e:
            print(f"❌ Startup script error: {e}")
            self.results['smart_start'] = False
            return False

    def run_full_test(self):
        """Runs all tests and generates a report"""
        print("🎯 SEARX SYSTEM TESTS START")
        print("=" * 60)
        print(f"📅 Date: {time.ctime()}")
        print(f"🖥️  Platform: {sys.platform}")
        print(f"🐍 Python: {sys.version}")
        print("=" * 60)
        
        tests = [
            ("Python Imports", self.test_imports),
            ("Port Manager", self.test_port_manager),
            ("Searx Interface", self.test_searx_interface),
            ("Docker", self.test_docker),
            ("System Files", self.test_files),
            ("Smart Startup", self.test_smart_start)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ CRITICAL ERROR in {test_name}: {e}")
                self.results[test_name.lower().replace(' ', '_')] = False
        
        # Generate final report
        self._generate_final_report(passed_tests, total_tests)
        
        return passed_tests >= total_tests * 0.8

    def _generate_final_report(self, passed_tests, total_tests):
        """Generates the final report"""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📋 FINAL REPORT - INTELLIGENT SEARX SYSTEM")
        print("=" * 60)
        
        print(f"⏱️  Test duration: {elapsed_time:.2f} seconds")
        print(f"🏆 Score: {passed_tests}/{total_tests} tests passed")
        
        # Results detail
        print("\n📊 RESULTS DETAIL:")
        for test_name, result in self.results.items():
            status = "✅ SUCCESS" if result else "❌ FAILED"
            print(f"  {status} - {test_name.replace('_', ' ').title()}")
        
        # Global status
        success_rate = passed_tests / total_tests
        if success_rate >= 0.9:
            print("\n🎉 EXCELLENT! System fully functional")
            print("🚀 Ready for production - Run: python searx_smart_start.py")
        elif success_rate >= 0.7:
            print("\n✅ GOOD! System largely functional")
            print("💡 Some improvements possible")
        elif success_rate >= 0.5:
            print("\n⚠️  AVERAGE! System partially functional")
            print("🔧 Fixes needed")
        else:
            print("\n❌ CRITICAL! System not functional")
            print("🆘 Urgent intervention required")
        
        # Specific recommendations
        self._generate_recommendations()

    def _generate_recommendations(self):
        """Generates recommendations based on results"""
        print("\n💡 SPECIFIC RECOMMENDATIONS:")
        
        if not self.results.get('imports', True):
            print("📦 DEPENDENCIES:")
            print("   - Run: pip install -r requirements.txt")
            print("   - Check your Python environment")
        
        if not self.results.get('docker', True):
            print("🐳 DOCKER:")
            print("   - Install Docker Desktop: https://docker.com/products/docker-desktop")
            print("   - Start the Docker service")
            print("   - Verify Docker is running: docker ps")
        
        if not self.results.get('files', True):
            print("📁 FILES:")
            print("   - Check project integrity")
            print("   - Re-download missing files")
        
        if not self.results.get('port_manager', True):
            print("🔧 PORT MANAGER:")
            print("   - Install psutil: pip install psutil")
            print("   - Check system permissions")
        
        print("\n🔗 RESOURCES:")
        print("   - Searx Documentation: https://searx.github.io/searx/")
        print("   - Docker Guide: https://docs.docker.com/get-started/")
        print("   - Python Support: https://python.org/downloads/")

def main():
    """Main function"""
    try:
        tester = SearxSystemTester()
        success = tester.run_full_test()
        
        if success:
            print("\n🎯 NEXT STEPS:")
            print("1. Launch: python searx_smart_start.py")
            print("2. Or use: searx_manager.bat (Windows)")
            print("3. Test the web interface once started")
        else:
            print("\n🔧 REQUIRED ACTIONS:")
            print("1. Correct the reported errors")
            print("2. Rerun this test: python test_searx_complete.py")
            print("3. Contact support if problems persist")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        return 2
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    sys.exit(main())
