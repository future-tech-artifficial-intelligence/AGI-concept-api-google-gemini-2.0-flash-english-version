#!/usr/bin/env python3
"""
Autostart manager for Searx
Launches and configures Searx when the application starts
"""

import logging
import time
import subprocess
import requests
import os
import sys
from typing import Optional

logger = logging.getLogger('SearxManager')

class SearxManager:
    """Manager for the Searx service"""
    
    def __init__(self, searx_url: str = "http://localhost:8080"):
        self.searx_url = searx_url
        self.is_running = False
        
    def check_docker_availability(self) -> bool:
        """Checks if Docker is available"""
        try:
            # Quick initial test
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Docker detected: {result.stdout.strip()}")
                
                # More in-depth test - check if Docker daemon is responding
                result2 = subprocess.run(['docker', 'info'], 
                                       capture_output=True, text=True, timeout=15)
                if result2.returncode == 0:
                    logger.info("Docker daemon is operational")
                    return True
                else:
                    logger.warning("Docker is installed but the daemon is not accessible")
                    logger.warning("Please start Docker Desktop")
                    return False
            else:
                logger.error("Docker is not available")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout during Docker check")
            return False
        except Exception as e:
            logger.error(f"Error checking Docker: {e}")
            return False
    
    def start_searx_service(self) -> bool:
        """Starts the Searx service with Docker Compose"""
        try:
            if not self.check_docker_availability():
                return False
            
            logger.info("Starting Searx service...")
            
            # Change to the project directory
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Start with Docker Compose
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 
                'up', '-d', '--remove-orphans'
            ], 
            capture_output=True, text=True, cwd=project_dir, timeout=120)
            
            if result.returncode == 0:
                logger.info("Docker Compose started successfully")
                
                # Wait for the service to be ready
                return self._wait_for_service_ready()
            else:
                logger.error(f"Docker Compose error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout during Searx startup")
            return False
        except Exception as e:
            logger.error(f"Error starting Searx: {e}")
            return False
    
    def _wait_for_service_ready(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """Waits for the Searx service to be ready to receive requests"""
        logger.info("Waiting for Searx to be ready...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.searx_url}/", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Searx is operational after {attempt + 1} attempts")
                    self.is_running = True
                    return True
                else:
                    logger.debug(f"Attempt {attempt + 1}: Status code {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"Attempt {attempt + 1}: {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(delay)
        
        logger.error("Searx did not become operational within the allotted time")
        return False
    
    def stop_searx_service(self) -> bool:
        """Stops the Searx service"""
        try:
            logger.info("Stopping Searx service...")
            
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 'down'
            ], 
            capture_output=True, text=True, cwd=project_dir, timeout=60)
            
            if result.returncode == 0:
                logger.info("Searx stopped successfully")
                self.is_running = False
                return True
            else:
                logger.error(f"Error during shutdown: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping Searx: {e}")
            return False
    
    def restart_searx_service(self) -> bool:
        """Restarts the Searx service"""
        logger.info("Restarting Searx...")
        if self.stop_searx_service():
            time.sleep(3)
            return self.start_searx_service()
        return False
    
    def get_service_status(self) -> dict:
        """Gets the status of the Searx service"""
        try:
            # Check Docker containers
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=ai_searx', '--format', 'table {{.Names}}\t{{.Status}}'
            ], capture_output=True, text=True, timeout=10)
            
            docker_status = "running" if "ai_searx" in result.stdout else "stopped"
            
            # Check HTTP connectivity
            try:
                response = requests.get(f"{self.searx_url}/", timeout=5)
                http_status = "accessible" if response.status_code == 200 else f"error_{response.status_code}"
            except:
                http_status = "inaccessible"
            
            return {
                "docker_status": docker_status,
                "http_status": http_status,
                "is_running": docker_status == "running" and http_status == "accessible",
                "url": self.searx_url
            }
            
        except Exception as e:
            logger.error(f"Error checking status: {e}")
            return {
                "docker_status": "unknown",
                "http_status": "unknown", 
                "is_running": False,
                "url": self.searx_url,
                "error": str(e)
            }
    
    def ensure_searx_running(self) -> bool:
        """Ensures Searx is running, starts it if necessary"""
        status = self.get_service_status()
        
        if status["is_running"]:
            logger.info("Searx is already running")
            self.is_running = True
            return True
        
        logger.info("Searx is not running, attempting to start...")
        return self.start_searx_service()

# Global instance
searx_manager = SearxManager()

def initialize_searx() -> bool:
    """Initializes Searx for the application"""
    logger.info("🔍 Initializing Searx search system...")
    
    success = searx_manager.ensure_searx_running()
    
    if success:
        logger.info("✅ Searx initialized successfully")
    else:
        logger.warning("⚠️ Searx initialization failed")
    
    return success

def get_searx_manager() -> SearxManager:
    """Returns the Searx manager instance"""
    return searx_manager

if __name__ == "__main__":
    # Test the manager
    manager = SearxManager()
    
    print("🔍 Testing Searx manager")
    print("="*50)
    
    # Check Docker
    if manager.check_docker_availability():
        print("✅ Docker available")
    else:
        print("❌ Docker not available")
        sys.exit(1)
    
    # Initial status
    status = manager.get_service_status()
    print(f"Initial status: {status}")
    
    # Start Searx
    if manager.ensure_searx_running():
        print("✅ Searx started successfully")
        
        # Test a simple search
        try:
            response = requests.get(f"{manager.searx_url}/search", 
                                  params={'q': 'test', 'format': 'json'}, 
                                  timeout=10)
            if response.status_code == 200:
                print("✅ Search test successful")
            else:
                print(f"⚠️ Search test failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Error during search test: {e}")
    else:
        print("❌ Failed to start Searx")
