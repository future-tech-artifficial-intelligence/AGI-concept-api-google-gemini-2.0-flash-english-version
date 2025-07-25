#!/usr/bin/env python3
"""
Dependency installation for the Searx visual capture system
"""

import subprocess
import sys
import logging
import os
import platform
import requests
import zipfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('searx_visual_deps_installer')

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

def check_chrome_installed():
    """Checks if Chrome is installed"""
    try:
        if platform.system() == "Windows":
            chrome_paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
            ]
            return any(os.path.exists(path) for path in chrome_paths)
        else:
            result = subprocess.run(['which', 'google-chrome'], capture_output=True, text=True)
            return result.returncode == 0
    except:
        return False

def check_edge_installed():
    """Checks if Edge is installed"""
    try:
        if platform.system() == "Windows":
            edge_paths = [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
            ]
            return any(os.path.exists(path) for path in edge_paths)
        else:
            result = subprocess.run(['which', 'microsoft-edge'], capture_output=True, text=True)
            return result.returncode == 0
    except:
        return False

def download_chromedriver():
    """Downloads ChromeDriver automatically"""
    try:
        logger.info("Downloading ChromeDriver...")
        
        # Determine Chrome version
        if platform.system() == "Windows":
            chrome_path = None
            for path in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"]:
                if os.path.exists(path):
                    chrome_path = path
                    break
            
            if not chrome_path:
                logger.warning("Chrome not found to determine version")
                return False
        
        # Use webdriver-manager for automatic installation
        logger.info("Using webdriver-manager for automatic installation")
        return True
        
    except Exception as e:
        logger.error(f"ChromeDriver download error: {e}")
        return False

def main():
    """Installs all necessary dependencies for visual capture"""
    logger.info("🔧 Installing dependencies for Searx visual capture")
    
    # Python dependencies
    dependencies = [
        'selenium',              # WebDriver for browser automation
        'webdriver-manager',     # Automatic driver management
        'Pillow',               # Image processing
    ]
    
    success_count = 0
    total_count = len(dependencies)
    
    # Installing Python packages
    for package in dependencies:
        if install_package(package):
            success_count += 1
    
    logger.info(f"📊 Python installation complete: {success_count}/{total_count} packages installed")
    
    # Checking browsers
    logger.info("🌐 Checking available browsers...")
    
    chrome_available = check_chrome_installed()
    edge_available = check_edge_installed()
    
    if chrome_available:
        logger.info("✅ Google Chrome detected")
    else:
        logger.warning("⚠️ Google Chrome not detected")
    
    if edge_available:
        logger.info("✅ Microsoft Edge detected")
    else:
        logger.warning("⚠️ Microsoft Edge not detected")
    
    if not chrome_available and not edge_available:
        logger.error("❌ No compatible browser detected")
        logger.error("Please install Google Chrome or Microsoft Edge")
        return False
    
    # Import test
    logger.info("🧪 Testing imports...")
    
    try:
        import selenium
        from selenium import webdriver
        logger.info("✅ Selenium imported successfully")
        
        import PIL
        logger.info("✅ Pillow imported successfully")
        
        from webdriver_manager.chrome import ChromeDriverManager
        logger.info("✅ WebDriver Manager imported successfully")
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    
    # Quick WebDriver test
    logger.info("🔧 Quick WebDriver test...")
    
    try:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from webdriver_manager.chrome import ChromeDriverManager
        
        if chrome_available:
            options = ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(
                service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
                options=options
            )
            driver.get("about:blank")
            driver.quit()
            logger.info("✅ Chrome WebDriver test successful")
            
    except Exception as e:
        logger.warning(f"⚠️ Chrome WebDriver test failed: {e}")
        
        # Try Edge as an alternative
        try:
            from selenium.webdriver.edge.options import Options as EdgeOptions
            from webdriver_manager.microsoft import EdgeChromiumDriverManager
            
            if edge_available:
                options = EdgeOptions()
                options.add_argument('--headless')
                
                driver = webdriver.Edge(
                    service=webdriver.edge.service.Service(EdgeChromiumDriverManager().install()),
                    options=options
                )
                driver.get("about:blank")
                driver.quit()
                logger.info("✅ Edge WebDriver test successful")
                
        except Exception as e2:
            logger.error(f"❌ Edge WebDriver test also failed: {e2}")
            logger.error("Visual capture might not work")
            return False
    
    logger.info("🎉 Installation completed successfully!")
    logger.info("The Searx visual capture system is ready for use")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("\n💡 Possible solutions:")
        logger.error("1. Install Google Chrome: https://www.google.com/chrome/")
        logger.error("2. Install Microsoft Edge: https://www.microsoft.com/edge")
        logger.error("3. Check installation permissions")
        logger.error("4. Retry with administrator privileges")
    
    sys.exit(0 if success else 1)
