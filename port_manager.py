#!/usr/bin/env python3
"""
Gestionnaire intelligent de ports pour Searx
Détecte automatiquement les ports disponibles et gère les conflits
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
    """Gestionnaire intelligent de ports"""
    
    def __init__(self):
        self.default_ports = [8080, 8081, 8082, 8083, 8084]
        self.searx_port = None
        self.is_windows = platform.system().lower() == 'windows'
        
    def is_port_available(self, port: int, host: str = 'localhost') -> bool:
        """Vérifie si un port est disponible"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                return result != 0  # Port libre si connexion échoue
        except Exception as e:
            logger.error(f"Erreur vérification port {port}: {e}")
            return False
    
    def find_available_port(self, start_port: int = 8080, max_attempts: int = 100) -> Optional[int]:
        """Trouve un port disponible à partir du port de départ"""
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_available(port):
                logger.info(f"✅ Port {port} disponible")
                return port
        
        logger.error(f"❌ Aucun port disponible dans la plage {start_port}-{start_port + max_attempts}")
        return None
    
    def get_process_using_port(self, port: int) -> Optional[Dict]:
        """Identifie le processus qui utilise un port"""
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
            logger.error(f"Erreur identification processus sur port {port}: {e}")
        
        return None
    
    def kill_process_on_port(self, port: int) -> bool:
        """Tue le processus qui utilise un port spécifique"""
        try:
            process_info = self.get_process_using_port(port)
            
            if not process_info:
                logger.info(f"Aucun processus trouvé sur le port {port}")
                return True
            
            pid = process_info['pid']
            name = process_info['name']
            
            logger.warning(f"⚠️  Processus détecté sur port {port}: {name} (PID: {pid})")
            
            # Demander confirmation pour des processus système critiques
            critical_processes = ['system', 'svchost.exe', 'winlogon.exe', 'csrss.exe']
            if name.lower() in critical_processes:
                logger.error(f"❌ Refus de tuer le processus système critique: {name}")
                return False
            
            # Tuer le processus
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=5)
                logger.info(f"✅ Processus {name} (PID: {pid}) terminé avec succès")
                
                # Vérifier que le port est maintenant libre
                time.sleep(1)
                return self.is_port_available(port)
                
            except psutil.TimeoutExpired:
                # Force kill si termination douce échoue
                proc.kill()
                logger.warning(f"🔥 Processus {name} forcé à s'arrêter")
                time.sleep(1)
                return self.is_port_available(port)
                
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'arrêt du processus sur port {port}: {e}")
            return False
    
    def free_port_with_confirmation(self, port: int) -> bool:
        """Libère un port avec confirmation utilisateur"""
        if self.is_port_available(port):
            logger.info(f"✅ Port {port} déjà libre")
            return True
        
        process_info = self.get_process_using_port(port)
        if not process_info:
            logger.warning(f"⚠️  Port {port} occupé mais processus non identifiable")
            return False
        
        print(f"\n🔍 Port {port} utilisé par:")
        print(f"   Processus: {process_info['name']}")
        print(f"   PID: {process_info['pid']}")
        print(f"   Commande: {process_info['cmdline'][:100]}...")
        print(f"   Statut: {process_info['status']}")
        
        # Auto-kill pour les processus Docker/Searx connus
        safe_to_kill = [
            'docker', 'searx', 'nginx', 'httpd', 'apache2',
            'node.exe', 'python.exe', 'uwsgi', 'gunicorn'
        ]
        
        if any(safe_name in process_info['name'].lower() for safe_name in safe_to_kill):
            print(f"🤖 Processus identifié comme sûr à arrêter: {process_info['name']}")
            return self.kill_process_on_port(port)
        else:
            print(f"⚠️  Attention: processus non identifié comme sûr")
            response = input(f"Voulez-vous arrêter ce processus pour libérer le port {port}? (o/N): ")
            
            if response.lower() in ['o', 'oui', 'y', 'yes']:
                return self.kill_process_on_port(port)
            else:
                print("❌ Arrêt annulé par l'utilisateur")
                return False
    
    def get_docker_compose_with_port(self, port: int) -> str:
        """Génère un docker-compose.yml avec le port spécifié"""
        return f"""version: '3.8'

services:
  searx:
    image: searxng/searxng:latest
    container_name: searx-ai-{port}
    ports:
      - "{port}:8080"
    environment:
      - SEARXNG_BASE_URL=http://localhost:{port}
      - SEARXNG_SECRET=ai-search-secret-{port}
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
        """Configure Searx avec un port disponible"""
        logger.info("🔍 Recherche d'un port disponible pour Searx...")
        
        # Essayer le port par défaut d'abord
        preferred_port = 8080
        
        if self.is_port_available(preferred_port):
            selected_port = preferred_port
            logger.info(f"✅ Port préféré {preferred_port} disponible")
        else:
            # Essayer de libérer le port 8080
            logger.info(f"🔄 Tentative de libération du port {preferred_port}")
            if self.free_port_with_confirmation(preferred_port):
                selected_port = preferred_port
                logger.info(f"✅ Port {preferred_port} libéré avec succès")
            else:
                # Chercher un port alternatif
                logger.info("🔍 Recherche d'un port alternatif...")
                selected_port = self.find_available_port(8081)
                
                if not selected_port:
                    logger.error("❌ Aucun port disponible trouvé")
                    return False, 0, ""
        
        # Générer le fichier docker-compose
        compose_content = self.get_docker_compose_with_port(selected_port)
        compose_filename = f"docker-compose.searx-port-{selected_port}.yml"
        
        try:
            with open(compose_filename, 'w', encoding='utf-8') as f:
                f.write(compose_content)
            
            logger.info(f"✅ Configuration Searx créée: {compose_filename}")
            logger.info(f"🚀 Searx sera accessible sur: http://localhost:{selected_port}")
            
            self.searx_port = selected_port
            
            # Sauvegarder la configuration
            self._save_port_config(selected_port, compose_filename)
            
            return True, selected_port, compose_filename
            
        except Exception as e:
            logger.error(f"❌ Erreur création configuration: {e}")
            return False, 0, ""
    
    def _save_port_config(self, port: int, compose_file: str):
        """Sauvegarde la configuration du port utilisé"""
        config = {
            'searx_port': port,
            'compose_file': compose_file,
            'timestamp': time.time(),
            'url': f"http://localhost:{port}"
        }
        
        try:
            with open('searx_port_config.json', 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            logger.info(f"✅ Configuration sauvegardée: searx_port_config.json")
        except Exception as e:
            logger.error(f"⚠️  Erreur sauvegarde configuration: {e}")
    
    def load_port_config(self) -> Optional[Dict]:
        """Charge la configuration du port sauvegardée"""
        try:
            if os.path.exists('searx_port_config.json'):
                with open('searx_port_config.json', 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Vérifier si la configuration est encore valide
                if self.is_port_available(config['searx_port']):
                    logger.warning(f"⚠️  Port {config['searx_port']} n'est plus utilisé par Searx")
                    return None
                
                logger.info(f"✅ Configuration chargée: Searx sur port {config['searx_port']}")
                self.searx_port = config['searx_port']
                return config
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement configuration: {e}")
        
        return None
    
    def start_searx_smart(self) -> Tuple[bool, str]:
        """Démarre Searx intelligemment avec gestion automatique des ports"""
        
        # Essayer de charger une configuration existante
        existing_config = self.load_port_config()
        
        if existing_config:
            logger.info(f"🔄 Utilisation configuration existante: port {existing_config['searx_port']}")
            return self._start_with_compose(existing_config['compose_file']), existing_config['url']
        
        # Configurer un nouveau port
        success, port, compose_file = self.setup_searx_with_available_port()
        
        if not success:
            return False, ""
        
        # Démarrer Searx
        if self._start_with_compose(compose_file):
            url = f"http://localhost:{port}"
            logger.info(f"🚀 Searx démarré avec succès sur: {url}")
            return True, url
        else:
            logger.error("❌ Échec du démarrage de Searx")
            return False, ""
    
    def _start_with_compose(self, compose_file: str) -> bool:
        """Démarre Searx avec un fichier docker-compose spécifique"""
        try:
            # Vérifier Docker
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("❌ Docker non disponible")
                return False
            
            # Démarrer le conteneur
            logger.info(f"🐳 Démarrage Docker avec {compose_file}...")
            
            cmd = ['docker-compose', '-f', compose_file, 'up', '-d']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode == 0:
                logger.info("✅ Conteneur Searx démarré")
                
                # Attendre que le service soit prêt
                logger.info("⏳ Attente de l'initialisation de Searx...")
                time.sleep(15)
                
                return True
            else:
                logger.error(f"❌ Erreur Docker: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur démarrage: {e}")
            return False
    
    def get_current_searx_url(self) -> Optional[str]:
        """Retourne l'URL Searx actuellement configurée"""
        config = self.load_port_config()
        if config:
            return config['url']
        
        # Vérifier les ports par défaut
        for port in self.default_ports:
            if not self.is_port_available(port):
                # Port occupé, potentiellement par Searx
                try:
                    import requests
                    response = requests.get(f"http://localhost:{port}/", timeout=5)
                    if response.status_code == 200 and 'searx' in response.text.lower():
                        logger.info(f"✅ Searx détecté sur port {port}")
                        return f"http://localhost:{port}"
                except:
                    continue
        
        return None
    
    def stop_all_searx_containers(self) -> bool:
        """Arrête tous les conteneurs Searx"""
        try:
            # Lister les conteneurs Searx
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=searx-ai', '--format', '{{.Names}}'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                containers = result.stdout.strip().split('\n')
                logger.info(f"🛑 Arrêt de {len(containers)} conteneur(s) Searx")
                
                for container in containers:
                    subprocess.run(['docker', 'stop', container], 
                                 capture_output=True, text=True)
                    subprocess.run(['docker', 'rm', container], 
                                 capture_output=True, text=True)
                
                # Nettoyer le fichier de config
                if os.path.exists('searx_port_config.json'):
                    os.remove('searx_port_config.json')
                
                logger.info("✅ Tous les conteneurs Searx arrêtés")
                return True
            else:
                logger.info("ℹ️  Aucun conteneur Searx en cours d'exécution")
                return True
                
        except Exception as e:
            logger.error(f"❌ Erreur arrêt conteneurs: {e}")
            return False

# Instance globale
port_manager = PortManager()

def get_port_manager() -> PortManager:
    """Retourne l'instance du gestionnaire de ports"""
    return port_manager

if __name__ == "__main__":
    # Test du gestionnaire de ports
    pm = PortManager()
    
    print("🔧 Test du gestionnaire de ports Searx")
    print("=" * 50)
    
    # Test de détection de port
    print(f"Port 8080 disponible: {pm.is_port_available(8080)}")
    
    if not pm.is_port_available(8080):
        process = pm.get_process_using_port(8080)
        if process:
            print(f"Processus sur 8080: {process['name']} (PID: {process['pid']})")
    
    # Test de démarrage intelligent
    success, url = pm.start_searx_smart()
    if success:
        print(f"✅ Searx démarré: {url}")
    else:
        print("❌ Échec démarrage Searx")
