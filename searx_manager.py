#!/usr/bin/env python3
"""
Gestionnaire de démarrage automatique pour Searx
Lance et configure Searx au démarrage de l'application
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
    """Gestionnaire pour le service Searx"""
    
    def __init__(self, searx_url: str = "http://localhost:8080"):
        self.searx_url = searx_url
        self.is_running = False
        
    def check_docker_availability(self) -> bool:
        """Vérifie si Docker est disponible"""
        try:
            # Test initial rapide
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Docker détecté: {result.stdout.strip()}")
                
                # Test plus approfondi - vérifier que le daemon Docker répond
                result2 = subprocess.run(['docker', 'info'], 
                                       capture_output=True, text=True, timeout=15)
                if result2.returncode == 0:
                    logger.info("Docker daemon est opérationnel")
                    return True
                else:
                    logger.warning("Docker est installé mais le daemon n'est pas accessible")
                    logger.warning("Veuillez démarrer Docker Desktop")
                    return False
            else:
                logger.error("Docker n'est pas disponible")
                return False
        except subprocess.TimeoutExpired:
            logger.error("Timeout lors de la vérification de Docker")
            return False
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de Docker: {e}")
            return False
    
    def start_searx_service(self) -> bool:
        """Démarre le service Searx avec Docker Compose"""
        try:
            if not self.check_docker_availability():
                return False
            
            logger.info("Démarrage du service Searx...")
            
            # Changer vers le répertoire du projet
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Démarrer avec Docker Compose
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 
                'up', '-d', '--remove-orphans'
            ], 
            capture_output=True, text=True, cwd=project_dir, timeout=120)
            
            if result.returncode == 0:
                logger.info("Docker Compose démarré avec succès")
                
                # Attendre que le service soit prêt
                return self._wait_for_service_ready()
            else:
                logger.error(f"Erreur Docker Compose: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Timeout lors du démarrage de Searx")
            return False
        except Exception as e:
            logger.error(f"Erreur lors du démarrage de Searx: {e}")
            return False
    
    def _wait_for_service_ready(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """Attend que le service Searx soit prêt à recevoir des requêtes"""
        logger.info("Attente que Searx soit prêt...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.searx_url}/", timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ Searx est opérationnel après {attempt + 1} tentatives")
                    self.is_running = True
                    return True
                else:
                    logger.debug(f"Tentative {attempt + 1}: Code de statut {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                logger.debug(f"Tentative {attempt + 1}: {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(delay)
        
        logger.error("Searx n'est pas devenu opérationnel dans le délai imparti")
        return False
    
    def stop_searx_service(self) -> bool:
        """Arrête le service Searx"""
        try:
            logger.info("Arrêt du service Searx...")
            
            project_dir = os.path.dirname(os.path.abspath(__file__))
            
            result = subprocess.run([
                'docker-compose', '-f', 'docker-compose.searx.yml', 'down'
            ], 
            capture_output=True, text=True, cwd=project_dir, timeout=60)
            
            if result.returncode == 0:
                logger.info("Searx arrêté avec succès")
                self.is_running = False
                return True
            else:
                logger.error(f"Erreur lors de l'arrêt: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur lors de l'arrêt de Searx: {e}")
            return False
    
    def restart_searx_service(self) -> bool:
        """Redémarre le service Searx"""
        logger.info("Redémarrage de Searx...")
        if self.stop_searx_service():
            time.sleep(3)
            return self.start_searx_service()
        return False
    
    def get_service_status(self) -> dict:
        """Obtient le statut du service Searx"""
        try:
            # Vérifier les conteneurs Docker
            result = subprocess.run([
                'docker', 'ps', '--filter', 'name=ai_searx', '--format', 'table {{.Names}}\t{{.Status}}'
            ], capture_output=True, text=True, timeout=10)
            
            docker_status = "running" if "ai_searx" in result.stdout else "stopped"
            
            # Vérifier la connectivité HTTP
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
            logger.error(f"Erreur lors de la vérification du statut: {e}")
            return {
                "docker_status": "unknown",
                "http_status": "unknown", 
                "is_running": False,
                "url": self.searx_url,
                "error": str(e)
            }
    
    def ensure_searx_running(self) -> bool:
        """S'assure que Searx est en cours d'exécution, le démarre si nécessaire"""
        status = self.get_service_status()
        
        if status["is_running"]:
            logger.info("Searx est déjà en cours d'exécution")
            self.is_running = True
            return True
        
        logger.info("Searx n'est pas en cours d'exécution, tentative de démarrage...")
        return self.start_searx_service()

# Instance globale
searx_manager = SearxManager()

def initialize_searx() -> bool:
    """Initialise Searx pour l'application"""
    logger.info("🔍 Initialisation du système de recherche Searx...")
    
    success = searx_manager.ensure_searx_running()
    
    if success:
        logger.info("✅ Searx initialisé avec succès")
    else:
        logger.warning("⚠️ Échec de l'initialisation de Searx")
    
    return success

def get_searx_manager() -> SearxManager:
    """Retourne l'instance du gestionnaire Searx"""
    return searx_manager

if __name__ == "__main__":
    # Test du gestionnaire
    manager = SearxManager()
    
    print("🔍 Test du gestionnaire Searx")
    print("="*50)
    
    # Vérifier Docker
    if manager.check_docker_availability():
        print("✅ Docker disponible")
    else:
        print("❌ Docker non disponible")
        sys.exit(1)
    
    # Status initial
    status = manager.get_service_status()
    print(f"Status initial: {status}")
    
    # Démarrer Searx
    if manager.ensure_searx_running():
        print("✅ Searx démarré avec succès")
        
        # Tester une recherche simple
        try:
            response = requests.get(f"{manager.searx_url}/search", 
                                  params={'q': 'test', 'format': 'json'}, 
                                  timeout=10)
            if response.status_code == 200:
                print("✅ Test de recherche réussi")
            else:
                print(f"⚠️ Test de recherche échoué: {response.status_code}")
        except Exception as e:
            print(f"⚠️ Erreur lors du test de recherche: {e}")
    else:
        print("❌ Échec du démarrage de Searx")
