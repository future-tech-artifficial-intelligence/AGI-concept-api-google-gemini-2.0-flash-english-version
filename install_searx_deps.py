#!/usr/bin/env python3
"""
Dependency installation for the Searx system
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('searx_deps_installer')

def install_package(package_name):
    """Installs a Python package with pip"""
    try:
        logger.info(f"Installing {package_name}...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', package_name
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"✅ {package_name} installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error installing {package_name}: {e.stderr}")
        return False

def main():
    """Installs all necessary dependencies for Searx"""
    logger.info("🔧 Installing dependencies for the Searx system")
    
    dependencies = [
        'beautifulsoup4',  # For HTML parsing
        'lxml',           # Faster XML/HTML parser
        'requests',       # HTTP client (usually already installed)
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    for package in dependencies:
        if install_package(package):
            success_count += 1
    
    logger.info(f"📊 Installation complete: {success_count}/{total_count} packages installed")
    
    if success_count == total_count:
        logger.info("✅ All dependencies have been installed successfully")
        return True
    else:
        logger.warning("⚠️ Some dependencies could not be installed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
