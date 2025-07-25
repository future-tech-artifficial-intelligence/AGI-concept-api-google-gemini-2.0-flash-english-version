#!/usr/bin/env python3
"""
Automatic Checker and Starter for the Searx system
Ensures everything is ready before starting the main application
"""

import subprocess
import sys
import time
import logging
import os
from pathlib import Path

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SearxAutoStart')

class SearxAutoStarter:
    """Automatic starter for the Searx system"""
    
    def __init__(self):
        self.docker_ready = False
        self.searx_ready = False
        
    def check_docker(self):
        """Checks and starts Docker if necessary"""
        logger.info("🐳 Checking Docker...")
        
        try:
            # Check if Docker is installed
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                logger.error("❌ Docker is not installed")
                return False
            
            logger.info(f"✅ Docker installed: {result.stdout.strip()}")
            
            # Check if Docker daemon is active
            result = subprocess.run(['docker', 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("✅ Docker daemon active")
                self.docker_ready = True
                return True
            else:
                logger.warning("⚠️ Docker daemon not active - attempting to start...")
                return self._start_docker()
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Docker is not responding")
            return False
        except FileNotFoundError:
            logger.error("❌ Docker not found in PATH")
            return False
        except Exception as e:
            logger.error(f"❌ Docker error: {e}")
            return False
    
    def _start_docker(self):
        """Starts Docker Desktop"""
        try:
            logger.info("🚀 Starting Docker Desktop...")
            
            # Possible paths for Docker Desktop
            docker_paths = [
                "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe",
                "C:\\Program Files (x86)\\Docker\\Docker\\Docker Desktop.exe"
            ]
            
            docker_exe = None
            for path in docker_paths:
                if os.path.exists(path):
                    docker_exe = path
                    break
            
            if not docker_exe:
                logger.error("❌ Docker Desktop not found")
                return False
            
            # Start Docker Desktop
            subprocess.Popen([docker_exe], shell=True)
            logger.info("⏳ Docker Desktop is starting...")
            
            # Wait for Docker to be ready (max 60 seconds)
            max_wait = 60
            wait_time = 0
            
            while wait_time < max_wait:
                time.sleep(5)
                wait_time += 5
                
                try:
                    result = subprocess.run(['docker', 'ps'], 
                                          capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        logger.info("✅ Docker Desktop started successfully!")
                        self.docker_ready = True
                        return True
                    else:
                        logger.info(f"⏳ Waiting for Docker... ({wait_time}/{max_wait}s)")
                
                except:
                    logger.info(f"⏳ Waiting for Docker... ({wait_time}/{max_wait}s)")
            
            logger.warning("⚠️ Docker is taking longer than expected")
            return False
            
        except Exception as e:
            logger.error(f"❌ Docker startup error: {e}")
            return False
    
    def ensure_searx_ready(self):
        """Ensures Searx is ready"""
        logger.info("🔍 Preparing the Searx system...")
        
        try:
            # Import Searx modules
            from port_manager import PortManager
            from searx_interface import SearxInterface
            
            # Initialize the port manager
            port_manager = PortManager()
            
            # Check if there's already an instance
            current_url = port_manager.get_current_searx_url()
            
            if current_url:
                logger.info(f"✅ Existing Searx detected: {current_url}")
                
                # Check if it's working
                searx = SearxInterface(current_url)
                if searx.check_health():
                    logger.info("✅ Existing Searx operational")
                    self.searx_ready = True
                    return True
                else:
                    logger.info("🔄 Existing Searx not functional - cleaning up...")
                    port_manager.stop_all_searx_containers()
            
            # Start Searx if Docker is ready
            if self.docker_ready:
                logger.info("🚀 Starting Searx...")
                success, url = port_manager.start_searx_smart()
                
                if success:
                    logger.info(f"✅ Searx started: {url}")
                    
                    # Wait for Searx to be fully ready
                    self._wait_for_searx_health(url)
                    self.searx_ready = True
                    return True
                else:
                    logger.warning("⚠️ Searx startup failed")
                    return False
            else:
                logger.warning("⚠️ Docker not available - Searx will run in degraded mode")
                return False
                
        except Exception as e:
            logger.error(f"❌ Searx preparation error: {e}")
            return False
    
    def _wait_for_searx_health(self, url):
        """Waits for Searx to be healthy"""
        logger.info("⏳ Waiting for full Searx initialization...")
        
        try:
            from searx_interface import SearxInterface
            searx = SearxInterface(url)
            
            max_wait = 30
            wait_time = 0
            
            while wait_time < max_wait:
                if searx.check_health():
                    logger.info("✅ Searx fully operational!")
                    return True
                
                time.sleep(2)
                wait_time += 2
                logger.info(f"⏳ Checking Searx health... ({wait_time}/{max_wait}s)")
            
            logger.warning("⚠️ Searx is taking longer than expected to be ready")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Searx health check error: {e}")
            return False
    
    def start_complete_system(self):
        """Starts the complete system"""
        logger.info("🎯 STARTING THE COMPLETE ARTIFICIAL INTELLIGENCE API GOOGLE GEMINI 2.0 FLASH SYSTEM")
        logger.info("=" * 60)
        
        # Step 1: Check Docker
        docker_ok = self.check_docker()
        
        # Step 2: Prepare Searx
        searx_ok = self.ensure_searx_ready()
        
        # Step 3: System Status Summary
        logger.info("📊 SYSTEM STATUS:")
        logger.info(f"   🐳 Docker: {'✅ Operational' if docker_ok else '❌ Not Available'}")
        logger.info(f"   🔍 Searx: {'✅ Operational' if searx_ok else '⚠️ Degraded Mode'}")
        
        if docker_ok and searx_ok:
            logger.info("🎉 SYSTEM FULLY OPERATIONAL!")
            return True
        elif docker_ok:
            logger.info("⚠️ SYSTEM PARTIALLY OPERATIONAL")
            return True
        else:
            logger.info("❌ SYSTEM IN DEGRADED MODE")
            return False

def main():
    """Main function"""
    auto_starter = SearxAutoStarter()
    return auto_starter.start_complete_system()

if __name__ == "__main__":
    try:
        success = main()
        logger.info("=" * 60)
        
        if success:
            logger.info("✅ READY FOR APPLICATION STARTUP")
        else:
            logger.info("⚠️ STARTUP IN DEGRADED MODE")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("❌ Startup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
