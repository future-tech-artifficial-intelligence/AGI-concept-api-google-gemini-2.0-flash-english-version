#!/usr/bin/env python3
"""
Platform Detector for Multi-environment Compatibility
Supports Windows, Linux, Android/Termux, and other systems
"""

import os
import platform
import subprocess
import sys
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger('PlatformDetector')

class PlatformDetector:
    """Detects and configures the execution environment"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self.config = self._get_platform_config()
    
    def _detect_platform(self) -> Dict[str, Any]:
        """Detects the execution platform"""
        # First, collect basic information
        is_termux = self._is_termux()
        is_android = self._is_android()
        is_windows = platform.system() == 'Windows'
        is_linux = platform.system() == 'Linux'
        
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'is_termux': is_termux,
            'is_android': is_android,
            'is_windows': is_windows,
            'is_linux': is_linux,
            'has_gui': self._has_gui_support(is_termux, is_windows, is_linux),
            'architecture': platform.architecture()[0]
        }
        
        # Determine the main platform type
        if info['is_termux']:
            info['platform_type'] = 'termux'
        elif info['is_android']:
            info['platform_type'] = 'android'
        elif info['is_windows']:
            info['platform_type'] = 'windows'
        elif info['is_linux']:
            info['platform_type'] = 'linux'
        else:
            info['platform_type'] = 'unknown'
        
        return info
    
    def _is_termux(self) -> bool:
        """Checks if running in Termux"""
        # Multiple checks to detect Termux
        termux_indicators = [
            'TERMUX_VERSION' in os.environ,
            'PREFIX' in os.environ and '/data/data/com.termux' in os.environ.get('PREFIX', ''),
            os.path.exists('/data/data/com.termux'),
            'termux' in str(sys.executable).lower()
        ]
        
        return any(termux_indicators)
    
    def _is_android(self) -> bool:
        """Checks if running on Android (but not necessarily Termux)"""
        android_indicators = [
            'ANDROID_ROOT' in os.environ,
            'ANDROID_DATA' in os.environ,
            os.path.exists('/system/build.prop'),
            'android' in platform.platform().lower()
        ]
        
        return any(android_indicators)
    
    def _has_gui_support(self, is_termux: bool, is_windows: bool, is_linux: bool) -> bool:
        """Checks if the environment supports graphical interfaces"""
        if is_termux:
            # Termux can have X11 with VNC
            return 'DISPLAY' in os.environ
        elif is_windows:
            return True
        elif is_linux:
            return 'DISPLAY' in os.environ or 'WAYLAND_DISPLAY' in os.environ
        
        return False
    
    def _get_platform_config(self) -> Dict[str, Any]:
        """Returns platform-specific configuration"""
        platform_type = self.platform_info['platform_type']
        
        configs = {
            'termux': {
                'data_path': os.path.expanduser('~/storage/shared/AI_Data'),
                'temp_path': '/data/data/com.termux/files/usr/tmp',
                'max_memory_usage': '512MB',
                'package_manager': 'pkg',
                'python_executable': 'python',
                'supported_features': {
                    'web_scraping': True,
                    'image_processing': True,  # With limitations
                    'audio_processing': False,  # Limited on Android
                    'gui': False,  # Unless X11 is configured
                    'file_system_access': True,
                    'network_access': True
                },
                'recommended_packages': [
                    'python', 'python-pip', 'git', 'curl', 'wget',
                    'clang', 'pkg-config', 'libjpeg-turbo', 'libpng'
                ]
            },
            'android': {
                'data_path': '/sdcard/AI_Data',
                'temp_path': '/tmp',
                'max_memory_usage': '256MB',
                'package_manager': None,
                'python_executable': 'python',
                'supported_features': {
                    'web_scraping': True,
                    'image_processing': False,
                    'audio_processing': False,
                    'gui': False,
                    'file_system_access': True,
                    'network_access': True
                }
            },
            'windows': {
                'data_path': os.path.expanduser('~/Documents/AI_Data'),
                'temp_path': os.environ.get('TEMP', 'C:\\temp'),
                'max_memory_usage': '2GB',
                'package_manager': 'pip',
                'python_executable': 'python',
                'supported_features': {
                    'web_scraping': True,
                    'image_processing': True,
                    'audio_processing': True,
                    'gui': True,
                    'file_system_access': True,
                    'network_access': True
                }
            },
            'linux': {
                'data_path': os.path.expanduser('~/AI_Data'),
                'temp_path': '/tmp',
                'max_memory_usage': '1GB',
                'package_manager': 'pip',
                'python_executable': 'python3',
                'supported_features': {
                    'web_scraping': True,
                    'image_processing': True,
                    'audio_processing': True,
                    'gui': True,
                    'file_system_access': True,
                    'network_access': True
                }
            }
        }
        
        return configs.get(platform_type, configs['linux'])
    
    def get_data_path(self) -> str:
        """Returns the appropriate data path for the platform"""
        path = self.config['data_path']
        os.makedirs(path, exist_ok=True)
        return path
    
    def get_temp_path(self) -> str:
        """Returns the appropriate temporary path"""
        return self.config['temp_path']
    
    def is_feature_supported(self, feature: str) -> bool:
        """Checks if a feature is supported on this platform"""
        return self.config['supported_features'].get(feature, False)
    
    def get_package_manager(self) -> Optional[str]:
        """Returns the appropriate package manager"""
        return self.config.get('package_manager')
    
    def install_system_packages(self, packages: list) -> bool:
        """Installs required system packages"""
        if not self.platform_info['is_termux']:
            logger.info("System package installation not necessary on this platform")
            return True
        
        try:
            for package in packages:
                logger.info(f"Installing system package: {package}")
                result = subprocess.run(['pkg', 'install', '-y', package], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to install {package}: {result.stderr}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error installing packages: {e}")
            return False
    
    def get_platform_summary(self) -> str:
        """Returns a summary of the detected platform"""
        info = self.platform_info
        config = self.config
        
        summary = f"""
🖥️  PLATFORM DETECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📱 Platform Type: {info['platform_type'].upper()}
🔧 System: {info['system']} ({info['architecture']})
🐍 Python: {info['python_version']}
📁 Data Path: {config['data_path']}
💾 Recommended Memory: {config['max_memory_usage']}

🔋 SUPPORTED FEATURES:
"""
        
        for feature, supported in config['supported_features'].items():
            status = "✅" if supported else "❌"
            summary += f"   {status} {feature.replace('_', ' ').title()}\n"
        
        if info['is_termux']:
            summary += f"""
🤖 TERMUX SPECIFIC:
   📦 Manager: {config['package_manager']}
   📋 Recommended Packages: {', '.join(config['recommended_packages'])}
"""
        
        return summary

# Global instance for application use
platform_detector = PlatformDetector()

def get_platform_detector() -> PlatformDetector:
    """Returns the platform detector instance"""
    return platform_detector

if __name__ == "__main__":
    detector = PlatformDetector()
    print(detector.get_platform_summary())
