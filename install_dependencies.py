#!/usr/bin/env python3
"""
Dependency installation script for new users
Usage: python install_dependencies.py
"""

import os
import sys
import subprocess
from auto_installer import run_auto_installer

def main():
    """Main entry point for installation"""
    print("🚀 DEPENDENCY INSTALLATION - AGI Concept API - Google Gemini 2.0 Flash Project open-source ")
    print("="*60)
    print("This script will automatically install all necessary dependencies")
    print("to run the project.")
    print("="*60 + "\n")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor} detected")
    
    # Install dependencies from requirements.txt first
    if os.path.exists('requirements.txt'):
        print("\n📦 Installing basic dependencies...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Basic dependencies installed successfully")
            else:
                print(f"⚠️ Warning during installation: {result.stderr}")
        except Exception as e:
            print(f"❌ Error installing basic dependencies: {str(e)}")
    
    # Run the auto-installer for additional modules
    print("\n🔧 Checking and installing additional modules...")
    success = run_auto_installer()
    
    print("\n" + "="*60)
    if success:
        print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        print("\n📋 Next steps:")
        print("1. Configure your Gemini API key in the .env file")
        print("2. Run the application with: python app.py")
        print("3. Open your browser to http://localhost:5000")
    else:
        print("⚠️ INSTALLATION PARTIALLY SUCCESSFUL")
        print("\n📋 Recommended actions:")
        print("1. Check the errors above")
        print("2. Manually install missing modules if necessary")
        print("3. Try running the application with: python app.py")
    print("="*60)
    
    return success

if __name__ == "__main__":
    main()
