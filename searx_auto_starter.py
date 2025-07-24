#!/usr/bin/env python3
"""
Vérificateur et démarreur automatique pour le système Searx
S'assure que tout est prêt avant le démarrage de l'application principale
"""

import subprocess
import sys
import time
import logging
import os
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SearxAutoStart')

class SearxAutoStarter:
    """Démarreur automatique du système Searx"""
    
    def __init__(self):
        self.docker_ready = False
        self.searx_ready = False
        
    def check_docker(self):
        """Vérifie et démarre Docker si nécessaire"""
        logger.info("🐳 Vérification de Docker...")
        
        try:
            # Vérifier si Docker est installé
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                logger.error("❌ Docker n'est pas installé")
                return False
            
            logger.info(f"✅ Docker installé: {result.stdout.strip()}")
            
            # Vérifier si Docker daemon est actif
            result = subprocess.run(['docker', 'ps'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                logger.info("✅ Docker daemon actif")
                self.docker_ready = True
                return True
            else:
                logger.warning("⚠️ Docker daemon non actif - tentative de démarrage...")
                return self._start_docker()
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Docker ne répond pas")
            return False
        except FileNotFoundError:
            logger.error("❌ Docker non trouvé dans le PATH")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur Docker: {e}")
            return False
    
    def _start_docker(self):
        """Démarre Docker Desktop"""
        try:
            logger.info("🚀 Démarrage de Docker Desktop...")
            
            # Chemins possibles pour Docker Desktop
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
                logger.error("❌ Docker Desktop non trouvé")
                return False
            
            # Démarrer Docker Desktop
            subprocess.Popen([docker_exe], shell=True)
            logger.info("⏳ Docker Desktop en cours de démarrage...")
            
            # Attendre que Docker soit prêt (max 60 secondes)
            max_wait = 60
            wait_time = 0
            
            while wait_time < max_wait:
                time.sleep(5)
                wait_time += 5
                
                try:
                    result = subprocess.run(['docker', 'ps'], 
                                          capture_output=True, text=True, timeout=5)
                    
                    if result.returncode == 0:
                        logger.info("✅ Docker Desktop démarré avec succès!")
                        self.docker_ready = True
                        return True
                    else:
                        logger.info(f"⏳ Attente Docker... ({wait_time}/{max_wait}s)")
                
                except:
                    logger.info(f"⏳ Attente Docker... ({wait_time}/{max_wait}s)")
            
            logger.warning("⚠️ Docker prend plus de temps que prévu")
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur démarrage Docker: {e}")
            return False
    
    def ensure_searx_ready(self):
        """S'assure que Searx est prêt"""
        logger.info("🔍 Préparation du système Searx...")
        
        try:
            # Importer les modules Searx
            from port_manager import PortManager
            from searx_interface import SearxInterface
            
            # Initialiser le gestionnaire de ports
            port_manager = PortManager()
            
            # Vérifier s'il y a déjà une instance
            current_url = port_manager.get_current_searx_url()
            
            if current_url:
                logger.info(f"✅ Searx existant détecté: {current_url}")
                
                # Vérifier s'il fonctionne
                searx = SearxInterface(current_url)
                if searx.check_health():
                    logger.info("✅ Searx existant opérationnel")
                    self.searx_ready = True
                    return True
                else:
                    logger.info("🔄 Searx existant non fonctionnel - nettoyage...")
                    port_manager.stop_all_searx_containers()
            
            # Démarrer Searx si Docker est prêt
            if self.docker_ready:
                logger.info("🚀 Démarrage de Searx...")
                success, url = port_manager.start_searx_smart()
                
                if success:
                    logger.info(f"✅ Searx démarré: {url}")
                    
                    # Attendre que Searx soit complètement prêt
                    self._wait_for_searx_health(url)
                    self.searx_ready = True
                    return True
                else:
                    logger.warning("⚠️ Échec démarrage Searx")
                    return False
            else:
                logger.warning("⚠️ Docker non disponible - Searx sera en mode dégradé")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur préparation Searx: {e}")
            return False
    
    def _wait_for_searx_health(self, url):
        """Attend que Searx soit en bonne santé"""
        logger.info("⏳ Attente de l'initialisation complète de Searx...")
        
        try:
            from searx_interface import SearxInterface
            searx = SearxInterface(url)
            
            max_wait = 30
            wait_time = 0
            
            while wait_time < max_wait:
                if searx.check_health():
                    logger.info("✅ Searx complètement opérationnel!")
                    return True
                
                time.sleep(2)
                wait_time += 2
                logger.info(f"⏳ Vérification santé Searx... ({wait_time}/{max_wait}s)")
            
            logger.warning("⚠️ Searx prend plus de temps que prévu à être prêt")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur vérification santé Searx: {e}")
            return False
    
    def start_complete_system(self):
        """Démarre le système complet"""
        logger.info("🎯 DÉMARRAGE DU SYSTÈME SEARX AI COMPLET")
        logger.info("=" * 60)
        
        # Étape 1: Vérifier Docker
        docker_ok = self.check_docker()
        
        # Étape 2: Préparer Searx
        searx_ok = self.ensure_searx_ready()
        
        # Étape 3: Résumé de l'état
        logger.info("📊 ÉTAT DU SYSTÈME:")
        logger.info(f"   🐳 Docker: {'✅ Opérationnel' if docker_ok else '❌ Non disponible'}")
        logger.info(f"   🔍 Searx: {'✅ Opérationnel' if searx_ok else '⚠️ Mode dégradé'}")
        
        if docker_ok and searx_ok:
            logger.info("🎉 SYSTÈME COMPLÈTEMENT OPÉRATIONNEL!")
            return True
        elif docker_ok:
            logger.info("⚠️ SYSTÈME PARTIELLEMENT OPÉRATIONNEL")
            return True
        else:
            logger.info("❌ SYSTÈME EN MODE DÉGRADÉ")
            return False

def main():
    """Fonction principale"""
    auto_starter = SearxAutoStarter()
    return auto_starter.start_complete_system()

if __name__ == "__main__":
    try:
        success = main()
        logger.info("=" * 60)
        
        if success:
            logger.info("✅ PRÊT POUR LE DÉMARRAGE DE L'APPLICATION")
        else:
            logger.info("⚠️ DÉMARRAGE EN MODE DÉGRADÉ")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("❌ Démarrage interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        sys.exit(1)
