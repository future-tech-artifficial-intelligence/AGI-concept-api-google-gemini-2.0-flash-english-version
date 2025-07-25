#!/usr/bin/env python3
"""
**Automatic Missing Dependency Installation System**
Analyzes required modules and installs them automatically

"""

import subprocess
import sys
import importlib
import logging
import os
from typing import List, Dict, Tuple

# Import platform detector if available
try:
    from platform_detector import get_platform_detector
    PLATFORM_DETECTION_AVAILABLE = True
except ImportError:
    PLATFORM_DETECTION_AVAILABLE = False

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('AutoInstaller')

class AutoInstaller:
    """Automatic dependency installation manager"""
    
    def __init__(self):
        # Detect platform if available
        if PLATFORM_DETECTION_AVAILABLE:
            self.detector = get_platform_detector()
            self.is_termux = self.detector.platform_info.get('is_termux', False)
            self.platform_type = self.detector.platform_info.get('platform_type', 'unknown')
        else:
            self.is_termux = self._detect_termux_fallback()
            self.platform_type = 'termux' if self.is_termux else 'standard'
        
        # Modules adapted by platform
        if self.is_termux:
            self.required_modules = self._get_termux_modules()
        else:
            self.required_modules = self._get_standard_modules()
        
        self.optional_modules = {
            # Optional modules that can improve performance
            'memory_profiler': 'memory-profiler>=0.61.0',
            'checksumdir': 'checksumdir>=1.2.0',
            'readability': 'readability-lxml>=0.8.1',
        }
    
    def _detect_termux_fallback(self) -> bool:
        """Fallback Termux detection if platform_detector is not available"""
        termux_indicators = [
            'TERMUX_VERSION' in os.environ,
            'PREFIX' in os.environ and '/data/data/com.termux' in os.environ.get('PREFIX', ''),
            os.path.exists('/data/data/com.termux')
        ]
        return any(termux_indicators)
    
    def _get_termux_modules(self) -> Dict[str, str]:
        """Termux compatible modules"""
        return {
            # Core modules - Termux compatible
            'requests': 'requests>=2.31.0',
            'flask': 'flask>=2.3.0',
            'flask_compress': 'flask-compress>=1.14',
            'numpy': 'numpy>=1.24.0',
            'pandas': 'pandas>=2.0.0',
            'pillow': 'pillow>=10.0.0',
            'beautifulsoup4': 'beautifulsoup4>=4.12.0',
            'lxml': 'lxml>=4.9.0',
            'aiohttp': 'aiohttp>=3.8.0',
            'networkx': 'networkx>=3.0',
            
            # Alternatives for Termux
            'cv2': 'opencv-python-headless>=4.8.0',  # Instead of opencv-python
            'matplotlib': 'matplotlib>=3.7.0',
            'scipy': 'scipy>=1.10.0',
            'textblob': 'textblob>=0.17.1',
            'nltk': 'nltk>=3.8.1',
            'psutil': 'psutil>=5.9.6',
            'tenacity': 'tenacity>=8.2.3',
            'py7zr': 'py7zr>=0.20.8',
            'xlsxwriter': 'xlsxwriter>=3.1.9',
            'feedparser': 'feedparser>=6.0.10',
        }
    
    def _get_standard_modules(self) -> Dict[str, str]:
        """Modules for standard platforms (non-Termux)"""
        return {
            # Modules for web scraping
            'aiohttp': 'aiohttp>=3.9.5',
            'networkx': 'networkx>=3.2.1',
            
            # Modules for image analysis and OCR
            'cv2': 'opencv-python>=4.10.0.84',
            'pytesseract': 'pytesseract>=0.3.10',
            
            # Modules for data analysis
            'matplotlib': 'matplotlib>=3.8.2',
            'scipy': 'scipy>=1.11.4',
            
            # Modules for audio processing
            'librosa': 'librosa>=0.10.2',
            'soundfile': 'soundfile>=0.12.1',
            
            # Modules for web scraping
            'selenium': 'selenium>=4.15.2',
            'beautifulsoup4': 'beautifulsoup4>=4.12.3',
            'lxml': 'lxml>=5.2.2',
            
            # Modules for databases
            'pymongo': 'pymongo>=4.6.0',
            'redis': 'redis>=5.0.1',
            
            # Modules for text analysis
            'textblob': 'textblob>=0.17.1',
            'nltk': 'nltk>=3.8.1',
            
            # System modules
            'psutil': 'psutil>=5.9.6',
            'tenacity': 'tenacity>=8.2.3',
            
            # Compression modules
            'py7zr': 'py7zr>=0.20.8',
            
            # File formats
            'xlsxwriter': 'xlsxwriter>=3.1.9',
            'feedparser': 'feedparser>=6.0.10',
        }
    
    def check_module_availability(self, module_name: str) -> bool:
        """Checks if a module is available"""
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    
    def install_package(self, package_spec: str) -> bool:
        """Installs a package via pip with Termux optimizations"""
        try:
            logger.info(f"📦 Installing {package_spec}...")
            
            # Pip arguments adapted to the platform
            pip_args = [sys.executable, '-m', 'pip', 'install']
            
            if self.is_termux:
                # Termux specific optimizations
                pip_args.extend(['--no-cache-dir', '--timeout', '600'])
            
            pip_args.append(package_spec)
            
            timeout = 600 if self.is_termux else 300  # More time for Termux
            
            result = subprocess.run(
                pip_args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {package_spec} installed successfully")
                return True
            else:
                logger.error(f"❌ Error installing {package_spec}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout during installation of {package_spec}")
            return False
        except Exception as e:
            logger.error(f"💥 Exception during installation of {package_spec}: {str(e)}")
            return False
    
    def upgrade_pip(self) -> bool:
        """Upgrades pip to the latest version"""
        try:
            logger.info("🔄 Upgrading pip...")
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"⚠️ Failed to upgrade pip: {str(e)}")
            return False
    
    def install_missing_modules(self, include_optional: bool = False) -> Dict[str, bool]:
        """Installs all missing modules"""
        results = {}
        
        # Upgrade pip first
        self.upgrade_pip()
        
        # Required modules
        missing_required = []
        for module_name, package_spec in self.required_modules.items():
            if not self.check_module_availability(module_name):
                missing_required.append((module_name, package_spec))
        
        if missing_required:
            logger.info(f"🔍 {len(missing_required)} required modules detected as missing")
            
            for module_name, package_spec in missing_required:
                success = self.install_package(package_spec)
                results[module_name] = success
        else:
            logger.info("✅ All required modules are already installed")
        
        # Optional modules if requested
        if include_optional:
            missing_optional = []
            for module_name, package_spec in self.optional_modules.items():
                if not self.check_module_availability(module_name):
                    missing_optional.append((module_name, package_spec))
            
            if missing_optional:
                logger.info(f"🔍 {len(missing_optional)} optional modules detected as missing")
                
                for module_name, package_spec in missing_optional:
                    success = self.install_package(package_spec)
                    results[f"{module_name} (optional)"] = success
        
        return results
    
    def check_and_install_all(self, include_optional: bool = False) -> bool:
        """Checks and installs all missing modules"""
        try:
            logger.info("🚀 Starting dependency check...")
            
            results = self.install_missing_modules(include_optional)
            
            if results:
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                logger.info(f"📊 Results: {successful}/{total} modules installed successfully")
                
                if successful == total:
                    logger.info("🎉 All dependencies installed successfully!")
                    return True
                else:
                    failed_modules = [name for name, success in results.items() if not success]
                    logger.warning(f"⚠️ Modules not installed: {', '.join(failed_modules)}")
                    return False
            else:
                logger.info("✅ No installation needed")
                return True
                
        except Exception as e:
            logger.error(f"💥 Error during dependency check: {str(e)}")
            return False
    
    def generate_missing_modules_report(self) -> str:
        """Generates a report of missing modules"""
        missing = []
        
        for module_name in self.required_modules:
            if not self.check_module_availability(module_name):
                missing.append(f"❌ {module_name} (required)")
        
        for module_name in self.optional_modules:
            if not self.check_module_availability(module_name):
                missing.append(f"⚠️ {module_name} (optional)")
        
        if missing:
            return f"Missing modules:\n" + "\n".join(missing)
        else:
            return "✅ All modules are installed"

def run_auto_installer():
    """Main entry point for automatic installation"""
    installer = AutoInstaller()
    
    # Display a report of missing modules
    print("\n" + "="*50)
    print("🔧 DEPENDENCY CHECK")
    print("="*50)
    print(installer.generate_missing_modules_report())
    print("="*50 + "\n")
    
    # Install missing modules
    success = installer.check_and_install_all(include_optional=False)
    
    if success:
        print("\n🎉 Automatic installation completed successfully!")
        print("🚀 You can now use all project features.")
    else:
        print("\n⚠️ Some modules could not be installed automatically.")
        print("📝 You can install them manually with: pip install <module_name>")
    
    return success

if __name__ == "__main__":
    run_auto_installer()
