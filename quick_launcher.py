#!/usr/bin/env python3
"""
Quick Launcher - artificial intelligence API GOOGLE GEMINI 2.0 FLASH Interactive Navigation System
Simple interface to quickly start all functionalities
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime

class InteractiveNavigationLauncher:
    """Launcher for the interactive navigation system"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_loaded = False
        self.available_actions = {}
        
    def print_header(self):
        """Prints the launcher header"""
        print("=" * 80)
        print("🚀 ARTIFICIAL INTELLIGENCE API GOOGLE GEMINI 2.0 FLASH INTERACTIVE NAVIGATION SYSTEM LAUNCHER")
        print("🎯 Quick Start Interface")
        print("=" * 80)
        
    def print_menu(self):
        """Prints the main menu"""
        print("\n📋 AVAILABLE ACTIONS:")
        print("=" * 50)
        
        actions = {
            "1": ("🏗️  Installation", "Install the complete system", "install"),
            "2": ("🎭 Demonstration", "Launch the interactive demonstration", "demo"),
            "3": ("🧪 Tests", "Run automated tests", "test"),
            "4": ("🔧 Maintenance", "Perform system maintenance", "maintenance"),
            "5": ("🌐 Navigation", "Start interactive navigation", "navigate"),
            "6": ("📊 Report", "Generate a status report", "status"),
            "7": ("🔍 Diagnosis", "Diagnose problems", "diagnose"),
            "8": ("📖 Guide", "Show the user guide", "guide"),
            "9": ("⚙️  Configuration", "Configure the system", "config"),
            "0": ("🚪 Exit", "Close the launcher", "exit")
        }
        
        for key, (icon, desc, action) in actions.items():
            print(f"   {key}. {icon} {desc}")
            self.available_actions[key] = action
            
        print("=" * 50)
        
    def check_prerequisites(self) -> bool:
        """Checks system prerequisites"""
        print("\n🔍 Checking prerequisites...")
        
        # Python check
        if sys.version_info < (3, 8):
            print(f"❌ Python 3.8+ required (current version: {sys.version_info.major}.{sys.version_info.minor})")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        
        # Critical file check
        critical_files = [
            'interactive_web_navigator.py',
            'gemini_interactive_adapter.py',
            'install_interactive_navigation.py'
        ]
        
        missing_files = []
        for file_name in critical_files:
            if not (self.project_root / file_name).exists():
                missing_files.append(file_name)
                
        if missing_files:
            print(f"❌ Missing files: {', '.join(missing_files)}")
            return False
        print("✅ Critical files present")
        
        # Configuration check
        env_file = self.project_root / '.env'
        if not env_file.exists():
            print("⚠️ .env file missing - configuration required")
            return False
            
        # API Key check
        with open(env_file, 'r', encoding='utf-8') as f:
            env_content = f.read()
            
        if 'GEMINI_API_KEY' not in env_content or 'your_api_key_here' in env_content: # Changed 'votre_cle_api_ici' to 'your_api_key_here' for consistency in English
            print("⚠️ artificial intelligence API GOOGLE GEMINI 2.0 FLASH API Key not configured")
            return False
        print("✅ Basic configuration present")
        
        return True
        
    def run_installation(self):
        """Launches the installation"""
        print("\n🏗️ SYSTEM INSTALLATION")
        print("-" * 40)
        
        install_script = self.project_root / 'install_interactive_navigation.py'
        if not install_script.exists():
            print("❌ Installation script not found")
            return False
            
        try:
            print("🚀 Launching installation...")
            result = subprocess.run([sys.executable, str(install_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Installation completed successfully")
                return True
            else:
                print(f"❌ Installation failed (code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"💥 Error during installation: {e}")
            return False
            
    def run_demo(self):
        """Launches the demonstration"""
        print("\n🎭 INTERACTIVE DEMONSTRATION")
        print("-" * 40)
        
        demo_script = self.project_root / 'demo_interactive_navigation.py'
        if not demo_script.exists():
            print("❌ Demonstration script not found")
            return False
            
        try:
            print("🎯 Launching demonstration...")
            result = subprocess.run([sys.executable, str(demo_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Demonstration completed")
                return True
            else:
                print(f"⚠️ Demonstration completed with warnings")
                return True
                
        except Exception as e:
            print(f"💥 Error during demonstration: {e}")
            return False
            
    def run_tests(self):
        """Launches the tests"""
        print("\n🧪 AUTOMATED TESTS")
        print("-" * 40)
        
        test_script = self.project_root / 'test_interactive_navigation.py'
        if not test_script.exists():
            print("❌ Test script not found")
            return False
            
        try:
            print("🔬 Executing tests...")
            result = subprocess.run([sys.executable, str(test_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed")
                return True
            else:
                print(f"⚠️ Some tests failed (code: {result.returncode})")
                return False
                
        except Exception as e:
            print(f"💥 Error during tests: {e}")
            return False
            
    def run_maintenance(self):
        """Launches maintenance"""
        print("\n🔧 SYSTEM MAINTENANCE")
        print("-" * 40)
        
        maintenance_script = self.project_root / 'maintenance_interactive_navigation.py'
        if not maintenance_script.exists():
            print("❌ Maintenance script not found")
            return False
            
        try:
            print("🛠️ Launching maintenance...")
            result = subprocess.run([sys.executable, str(maintenance_script)], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ Maintenance completed successfully")
                return True
            else:
                print(f"⚠️ Maintenance completed with warnings")
                return True
                
        except Exception as e:
            print(f"💥 Error during maintenance: {e}")
            return False
            
    def start_navigation(self):
        """Starts interactive navigation"""
        print("\n🌐 INTERACTIVE NAVIGATION")
        print("-" * 40)
        
        navigator_script = self.project_root / 'interactive_web_navigator.py'
        if not navigator_script.exists():
            print("❌ Interactive navigator not found")
            return False
            
        print("🎯 Starting interactive navigation...")
        print("💡 Use Ctrl+C to stop")
        
        try:
            result = subprocess.run([sys.executable, str(navigator_script)], 
                                  capture_output=False, text=True)
            
            print(f"\n🏁 Navigation completed (code: {result.returncode})")
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ Navigation interrupted by user")
            return True
        except Exception as e:
            print(f"💥 Error during navigation: {e}")
            return False
            
    def generate_status_report(self):
        """Generates a status report"""
        print("\n📊 STATUS REPORT")
        print("-" * 40)
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "project_root": str(self.project_root),
            "files_status": {},
            "config_status": {}
        }
        
        # File check
        critical_files = [
            'interactive_web_navigator.py',
            'gemini_interactive_adapter.py',
            'install_interactive_navigation.py',
            'demo_interactive_navigation.py',
            'test_interactive_navigation.py',
            'maintenance_interactive_navigation.py',
            'GUIDE_NAVIGATION_INTERACTIVE.md'
        ]
        
        for file_name in critical_files:
            file_path = self.project_root / file_name
            status["files_status"][file_name] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
            
        # Configuration check
        config_files = ['.env', 'ai_api_config.json']
        for config_file in config_files:
            config_path = self.project_root / config_file
            status["config_status"][config_file] = {
                "exists": config_path.exists(),
                "configured": True  # Simplified for this example
            }
            
        # Displaying the report
        print(f"📅 Timestamp: {status['timestamp']}")
        print(f"🐍 Python: {status['python_version']}")
        print(f"📁 Project: {status['project_root']}")
        
        print("\n📁 FILES:")
        for file_name, file_info in status["files_status"].items():
            status_icon = "✅" if file_info["exists"] else "❌"
            size_info = f"({file_info['size']} bytes)" if file_info["exists"] else ""
            print(f"   {status_icon} {file_name} {size_info}")
            
        print("\n⚙️ CONFIGURATION:")
        for config_name, config_info in status["config_status"].items():
            status_icon = "✅" if config_info["exists"] else "❌"
            print(f"   {status_icon} {config_name}")
            
        # Saving the report
        report_file = f"status_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Report saved: {report_file}")
        
        return True
        
    def run_diagnostics(self):
        """Launches diagnostics"""
        print("\n🔍 SYSTEM DIAGNOSIS")
        print("-" * 40)
        
        issues = []
        
        # Import diagnostics
        print("🔬 Testing imports...")
        test_modules = [
            'google.generativeai',
            'selenium',
            'requests',
            'bs4',
            'PIL'
        ]
        
        for module in test_modules:
            try:
                __import__(module)
                print(f"   ✅ {module}")
            except ImportError:
                print(f"   ❌ {module}")
                issues.append(f"Missing module: {module}")
                
        # Configuration diagnostics
        print("\n⚙️ Checking configuration...")
        env_file = self.project_root / '.env'
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                env_content = f.read()
                
            if 'GEMINI_API_KEY=your_api_key_here' in env_content: # Consistent with 'your_api_key_here'
                print("   ⚠️ API Key not configured")
                issues.append("artificial intelligence API GOOGLE GEMINI 2.0 FLASH API Key not configured")
            else:
                print("   ✅ API Configuration")
        else:
            print("   ❌ .env file missing")
            issues.append("Configuration file missing")
            
        # Diagnosis summary
        print(f"\n📋 SUMMARY:")
        if not issues:
            print("🎉 No problems detected!")
        else:
            print(f"⚠️ {len(issues)} issue(s) detected:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
                
        return len(issues) == 0
        
    def show_guide(self):
        """Displays the user guide"""
        print("\n📖 USER GUIDE")
        print("-" * 40)
        
        guide_file = self.project_root / 'GUIDE_NAVIGATION_INTERACTIVE.md'
        if guide_file.exists():
            print(f"📚 Guide available: {guide_file.name}")
            print("💡 Open this file in a markdown editor for complete documentation")
            
            # Displaying first lines
            try:
                with open(guide_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:20]
                    
                print("\n📋 OVERVIEW:")
                for line in lines:
                    print(f"   {line.rstrip()}")
                    
                if len(lines) >= 20:
                    print("   ... (see full file for more details)")
                    
            except Exception as e:
                print(f"❌ Error reading guide: {e}")
        else:
            print("❌ Guide not found")
            print("💡 Consult README.md or online documentation")
            
        return True
        
    def configure_system(self):
        """Interactive system configuration"""
        print("\n⚙️ SYSTEM CONFIGURATION")
        print("-" * 40)
        
        env_file = self.project_root / '.env'
        
        print("🔑 Configuring artificial intelligence API GOOGLE GEMINI 2.0 FLASH API Key")
        current_key = "not configured"
        
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'GEMINI_API_KEY=' in content:
                    for line in content.split('\n'):
                        if line.startswith('GEMINI_API_KEY='):
                            key_value = line.split('=', 1)[1]
                            if key_value and key_value != 'your_api_key_here': # Consistent with 'your_api_key_here'
                                current_key = "configured"
                            break
                            
        print(f"📊 Current status: {current_key}")
        
        if current_key == "not configured":
            print("\n💡 To configure your API Key:")
            print("   1. Get an API key from https://makersuite.google.com/app/apikey")
            print("   2. Edit the .env file")
            print("   3. Replace 'your_api_key_here' with your actual key") # Consistent with 'your_api_key_here'
            print("   4. Relaunch this configurator")
        else:
            print("✅ API Key configured")
            
        return True
        
    def run_interactive_menu(self):
        """Launches the main interactive menu"""
        while True:
            self.print_header()
            
            # Quick prerequisite check
            prereq_ok = self.check_prerequisites()
            if not prereq_ok:
                print("\n⚠️ WARNING: Prerequisites not met")
                print("💡 Recommendation: Start with installation (option 1)")
                
            self.print_menu()
            
            try:
                choice = input("\n🎯 Your choice (0-9): ").strip()
                
                if choice not in self.available_actions:
                    print("❌ Invalid choice. Please select 0-9.")
                    input("Press Enter to continue...")
                    continue
                    
                action = self.available_actions[choice]
                
                if action == "exit":
                    print("\n👋 Goodbye!")
                    break
                    
                # Execute the action
                print(f"\n🚀 Executing: {action}")
                time.sleep(0.5)  # Small pause for UX
                
                if action == "install":
                    self.run_installation()
                elif action == "demo":
                    self.run_demo()
                elif action == "test":
                    self.run_tests()
                elif action == "maintenance":
                    self.run_maintenance()
                elif action == "navigate":
                    self.start_navigation()
                elif action == "status":
                    self.generate_status_report()
                elif action == "diagnose":
                    self.run_diagnostics()
                elif action == "guide":
                    self.show_guide()
                elif action == "config":
                    self.configure_system()
                    
                input("\n⏸️ Press Enter to return to the menu...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n💥 Error: {e}")
                input("Press Enter to continue...")

def main():
    """Main entry point"""
    launcher = InteractiveNavigationLauncher()
    
    try:
        launcher.run_interactive_menu()
        return True
    except Exception as e:
        print(f"💥 Critical error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
