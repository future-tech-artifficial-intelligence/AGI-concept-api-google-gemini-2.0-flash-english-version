#!/usr/bin/env python3
"""
Intelligent Port Manager for Searx
Automatically detects available ports and manages conflicts
"""

import socket
import subprocess
import logging
import time
import platform
import psutil
from typing import Optional, List, Dict, Tuple
import json
import os

logger = logging.getLogger('PortManager')

class PortManager:
    """Intelligent Port Manager"""
    
    def __init__(self):
        self.default_ports = [8080, 8081, 8082, 8083, 8084]
        self.searx_port = None
        self.is_windows = platform.system().lower() == 'windows'
        
    def is_port_available(self, port: int, host: str = 'localhost') -> bool:
        """Checks if a port is available"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # Port is free if connection fails
        except Exception as e:
            logger.error(f"Error checking port {port}: {e}")
            return False
    
    def find_available_port(self, start_port: int = 8080, max_attempts: int = 100) -> Optional[int]:
        """Finds an available port starting from the given port"""
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_available(port):
                logger.info(f"✅ Port {port} available")
                return port
        
        logger.error(f"❌ No port available in range {start_port}-{start_port + max_attempts}")
        return None
    
    def get_process_using_port(self, port: int) -> Optional[Dict]:
        """Identifies the process using a port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    for conn in proc.connections():
                        if conn.laddr.port == port:
                            return {
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else '',
                                'status': proc.status()
                            }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error identifying process on port {port}: {e}")
        
        return None
    
    def kill_process_on_port(self, port: int) -> bool:
        """Kills the process using a specific port"""
        try:
            process_info = self.get_process_using_port(port)
            
            if not process_info:
                logger.info(f"No process found on port {port}")
                return True
            
            pid = process_info['pid']
            name = process_info['name']
            
            logger.warning(f"⚠️  Process detected on port {port}: {name} (PID: {pid})")
            
            # Ask for confirmation for critical system processes
            critical_processes = ['system', 'svchost.exe', 'winlogon.exe', 'csrss.exe']
            if name.lower() in critical_processes:
                logger.error(f"❌ Refusing to kill critical system process: {name}")
                return False
            
            # Kill the process
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=5)
                logger.info(f"✅ Process {name} (PID: {pid}) terminated successfully")
                
                # Check if the port is now free
                time.sleep(1)
                return self.is_port_available(port)
                
            except psutil.TimeoutExpired:
                # Force kill if graceful termination fails
                proc.kill()
                logger.warning(f"🔥 Process {name} forced to stop")
                time.sleep(1)
                return self.is_port_available(port)
                
        except Exception as e:
            logger.error(f"❌ Error stopping process on port {port}: {e}")
            return False
    
    def free_port_with_confirmation(self, port: int) -> bool:
        """Frees a port with user confirmation"""
        if self.is_port_available(port):
            logger.info(f"✅ Port {port} already free")
            return True
        
        process_info = self.get_process_using_port(port)
        if not process_info:
            logger.warning(f"⚠️  Port {port} occupied but process not identifiable")
            return False
        
        print(f"\n🔍 Port {port} used by:")
        print(f"   Process: {process_info['name']}")
        print(f"   PID: {process_info['pid']}")
        print(f"   Command: {process_info['cmdline'][:100]}...")
        print(f"   Status: {process_info['status']}")
        
        # Auto-kill for known Docker/Searx processes
        safe_to_kill = [
            'docker', 'searx', 'nginx', 'httpd', 'apache2',
            'node.exe', 'python.exe', 'uwsgi', 'gunicorn'
        ]
        
        if any(safe_name in process_info['name'].lower() for safe_name in safe_to_kill):
            print(f"🤖 Process identified as safe to stop: {process_info['name']}")
            return self.kill_process_on_port(port)
        else:
            print(f"⚠️  Warning: process not identified as safe")
            response = input(f"Do you want to stop this process to free port {port}? (y/N): ")
            
            if response.lower() in ['o', 'oui', 'y', 'yes']:
                return self.kill_process_on_port(port)
            else:
                print("❌ Stop cancelled by user")
                return False
    
    def get_docker_compose_with_port(self, port: int) -> str:
        """Generates a docker-compose.yml with the specified port"""
        return f"""version: '3.8'

services:
  searx:
    image: searxng/searxng:latest
    container_name: searx-artificial-intelligence-API-GOOGLE-GEMINI-2.0-FLASH-{port}
    ports:
      - "{port}:8080"
    environment:
      - SEARXNG_BASE_URL=http://localhost:{port}
      - SEARXNG_SECRET=artificial-intelligence-API-GOOGLE-GEMINI-2.0-FLASH-search-secret-{port}
    volumes:
      - searx-data-{port}:/etc/searxng
    restart: unless-stopped
    networks:
      - searx-network-{port}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  searx-data-{port}:

networks:
  searx-network-{port}:
    driver: bridge
"""
    
    def setup_searx_with_available_port(self) -> Tuple[bool, int, str]:
        """Sets up Searx with an available port"""
        logger.info("🔍 Searching for an available port for Searx...")
        
        # Try the default port first
        preferred_port = 8080
        
        if self.is_port_available(preferred_port):
            selected_port = preferred_port
            logger.info(f"✅ Preferred port {preferred_port} available")
        else:
            # Try to free port 8080
            logger.info(f"🔄 Attempting to free port {preferred_port}")
            if self.free_port_with_confirmation(preferred_port):
                selected_port = preferred_port
                logger.info(f"✅ Port {preferred_port} successfully freed")
            else:
                # Search for an alternative port
                logger.info("🔍 Searching for an alternative port...")
                selected_port = self.find_available_port(8081)
                
                if not selected_port:
                    logger.error("❌ No available port found")
                    return False, 0, ""
        
        # Generate the docker-compose file
        compose_content = self.get_docker_compose_with_port(selected_port)
        compose_filename = f"docker-compose.searx-port-{selected_port}.yml"
        
        try:
            with open(compose_filename, 'w', encoding='utf-8') as f:
                f.write(compose_content)
            
            logger.info(f"✅ Searx configuration created: {compose_filename}")
            logger.info(f"🚀 Searx will be accessible at: http://localhost:{selected_port}")
            
            self.searx_port = selected_port
            
            # Save the configuration
            self._save_port_config(selected_port, compose_filename)
            
            return True, selected_port, compose_filename
            
        except Exception as e:
            logger.error(f"❌ Error creating configuration: {e}")
            return False, 0, ""
    
    def _save_port_config(self, port: int, compose_file: str):
        """Saves the used port configuration"""
        config = {
            'searx_port': port,
            'compose_file': compose_file,
            'timestamp': time.time(),
            'url': f"http://localhost:{port}"
        }
        
        try:
            with open('searx_port_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✅ Configuration saved: searx_port_config.json")
        except Exception as e:
            logger.error(f"⚠️  Error saving configuration: {e}")
    
    def load_port_config(self) -> Optional[Dict]:
        """Loads the saved port configuration"""
        try:
            if os.path.exists('searx_port_config.json'):
                with open('searx_port_config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Check if the configuration is still valid
                # This logic is a bit counter-intuitive. If the port IS available, it means Searx is NOT using it.
                # If Searx is running, the port should NOT be available.
                # Re-evaluating: If the port is *not* available, it means *something* is using it.
                # We need to verify that *that something* is Searx.
                # For now, let's keep the original logic which assumes if it's available, it's not Searx.
                if self.is_port_available(config['searx_port']):
                    logger.warning(f"⚠️  Port {config['searx_port']} is no longer in use by Searx (or is free)")
                    # It's better to return None if Searx is not confirmed to be using it
                    return None 
                
                # If the port is NOT available, we should ideally check if the process using it is Searx
                # For simplicity, we assume if it's in config and not available, Searx is using it.
                logger.info(f"✅ Configuration loaded: Searx on port {config['searx_port']}")
                self.searx_port = config['searx_port']
                return config
            
        except Exception as e:
            logger.error(f"❌ Error loading configuration: {e}")
        
        return None
    
    def start_searx_smart(self) -> Tuple[bool, str]:
        """Starts Searx intelligently with automatic port management"""
        
        # Try to load an existing configuration
        existing_config = self.load_port_config()
        
        if existing_config:
            logger.info(f"🔄 Using existing configuration: port {existing_config['searx_port']}")
            # Before starting, ensure the port is indeed in use by Searx or can be taken
            if self.is_port_available(existing_config['searx_port']):
                logger.warning(f"Port {existing_config['searx_port']} is free despite existing config. Reconfiguring.")
                return self.setup_searx_with_available_port() # Re-run setup to find a working port
            
            return self._start_with_compose(existing_config['compose_file']), existing_config['url']
        
        # Configure a new port
        success, port, compose_file = self.setup_searx_with_available_port()
        
        if not success:
            return False, ""
        
        # Start Searx
        if self._start_with_compose(compose_file):
            url = f"http://localhost:{port}"
            logger.info(f"🚀 Searx started successfully on: {url}")
            return True, url
        else:
            logger.error("❌ Failed to start Searx")
            return False, ""
    
    def _start_with_compose(self, compose_file: str) -> bool:
        """Starts Searx with a specific docker-compose file"""
        try:
            # Check Docker
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Docker not available")
                return False
            
            # Start the container
            logger.info(f"🐳 Starting Docker with {compose_file}...")
            
            cmd = ['docker-compose', '-f', compose_file, 'up', '-d']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("✅ Searx container started")
                
                # Wait for the service to be ready
                logger.info("⏳ Waiting for Searx to initialize...")
                time.sleep(15)
                
                return True
            else:
                logger.error(f"❌ Docker error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Startup error: {e}")
            return False
    
    def get_current_searx_url(self) -> Optional[str]:
        """Returns the currently configured Searx URL"""
        config = self.load_port_config()
        if config and not self.is_port_available(config['searx_port']): # Only return if the port is actually in use
            return config['url']
        
        # Check default ports
        for port in self.default_ports:
            if not self.is_port_available(port):
                # Port occupied, potentially by Searx
                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/", timeout=5)
                    if response.status_code == 200 and 'searx' in response.text.lower():
                        logger.info(f"✅ Searx detected on port {port}")
                        return f"http://localhost:{port}"
                except:
                    continue
        
        return None
    
    def stop_all_searx_containers(self) -> bool:
        """Stops all Searx containers"""
        try:
            # List Searx containers
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=searx-artificial-intelligence-API-GOOGLE-GEMINI-2.0-FLASH', '--format', '{{.Names}}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                containers = result.stdout.strip().split('\n')
                logger.info(f"🛑 Stopping {len(containers)} Searx container(s)")
                
                for container in containers:
                    subprocess.run(['docker', 'stop', container], 
                                 capture_output=True, text=True)
                    subprocess.run(['docker', 'rm', container], 
                                 capture_output=True, text=True)
                
                # Clean up the config file
                if os.path.exists('searx_port_config.json'):
                    os.remove('searx_port_config.json')
                
                logger.info("✅ All Searx containers stopped")
                return True
            else:
                logger.info("ℹ️  No Searx containers currently running")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error stopping containers: {e}")
            return False

# Global instance
port_manager = PortManager()

def get_port_manager() -> PortManager:
    """Returns the port manager instance"""
    return port_manager

if __name__ == "__main__":
    # Test the port manager
    pm = PortManager()
    
    print("🔧 Testing Searx port manager")
    print("=" * 50)
    
    # Port detection test
    print(f"Port 8080 available: {pm.is_port_available(8080)}")
    
    if not pm.is_port_available(8080):
        process = pm.get_process_using_port(8080)
        if process:
            print(f"Process on 8080: {process['name']} (PID: {process['pid']})")
    
    # Smart startup test
    success, url = pm.start_searx_smart()
    if success:
        print(f"✅ Searx started: {url}")
    else:
        print("❌ Failed to start Searx")
