#!/usr/bin/env python3
"""
Script de démarrage intelligent de Searx avec gestion automatique des ports
Résout automatiquement les conflits de ports et configure Searx de manière optimale
"""

import sys
import logging
import time
from pathlib import Path

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('searx_smart_start.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('SearxSmartStart')

def main():
    """Fonction principale de démarrage intelligent"""
    
    print("🚀 DÉMARRAGE INTELLIGENT DE SEARX")
    print("=" * 50)
    
    try:
        # Importer les modules nécessaires
        from port_manager import PortManager
        from searx_interface import SearxInterface
        
        # Initialiser le gestionnaire de ports
        logger.info("🔧 Initialisation du gestionnaire de ports...")
        port_manager = PortManager()
        
        # Vérifier l'état actuel
        current_url = port_manager.get_current_searx_url()
        if current_url:
            print(f"✅ Searx déjà actif sur: {current_url}")
            
            # Tester la connectivité
            searx = SearxInterface(current_url)
            if searx.check_health():
                print("🟢 Searx fonctionne correctement")
                print(f"🌐 Interface accessible: {current_url}")
                return True
            else:
                print("⚠️  Searx détecté mais non fonctionnel, redémarrage...")
                port_manager.stop_all_searx_containers()
        
        # Démarrage intelligent
        print("🔍 Analyse des ports disponibles...")
        success, url = port_manager.start_searx_smart()
        
        if not success:
            print("❌ Échec du démarrage intelligent")
            print("\n🔧 Solutions possibles:")
            print("1. Exécutez: python free_port_8080.py")
            print("2. Utilisez: docker-compose -f docker-compose.searx-alt.yml up -d")
            print("3. Redémarrez votre ordinateur pour libérer tous les ports")
            return False
        
        # Vérification finale
        print(f"⏳ Vérification de Searx sur {url}...")
        searx = SearxInterface(url)
        
        # Attendre que Searx soit prêt
        max_attempts = 12
        for attempt in range(max_attempts):
            if searx.check_health():
                print(f"✅ Searx opérationnel sur: {url}")
                break
            
            print(f"⏳ Tentative {attempt + 1}/{max_attempts} - Attente initialisation...")
            time.sleep(5)
        else:
            print("❌ Searx ne répond pas après 60 secondes")
            return False
        
        # Test de recherche
        print("\n🧪 Test de recherche...")
        results = searx.search("test intelligence artificielle", max_results=3)
        
        if results:
            print(f"✅ Test réussi: {len(results)} résultats trouvés")
            print(f"   Premier résultat: {results[0].title}")
        else:
            print("⚠️  Test de recherche échoué")
        
        # Informations finales
        print("\n" + "=" * 50)
        print("🎉 SEARX DÉMARRÉ AVEC SUCCÈS!")
        print(f"🌐 URL: {url}")
        print(f"📊 Interface de gestion: {url}")
        print("🔍 Prêt pour les recherches autonomes")
        print("=" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Module manquant: {e}")
        print("💡 Exécutez: pip install -r requirements.txt")
        return False
        
    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        print(f"❌ Erreur inattendue: {e}")
        return False

def show_status():
    """Affiche l'état actuel de Searx"""
    try:
        from port_manager import PortManager
        from searx_interface import SearxInterface
        
        pm = PortManager()
        
        print("📊 ÉTAT DE SEARX")
        print("=" * 30)
        
        # Vérifier la configuration sauvegardée
        config = pm.load_port_config()
        if config:
            print(f"📁 Configuration: {config['compose_file']}")
            print(f"🔌 Port: {config['searx_port']}")
            print(f"🌐 URL: {config['url']}")
            print(f"📅 Configuré: {time.ctime(config['timestamp'])}")
        else:
            print("❌ Aucune configuration sauvegardée")
        
        # Tester les ports courants
        print("\n🔍 Scan des ports:")
        for port in [8080, 8081, 8082, 8083]:
            status = "🟢 Libre" if pm.is_port_available(port) else "🔴 Occupé"
            print(f"   Port {port}: {status}")
            
            if not pm.is_port_available(port):
                process = pm.get_process_using_port(port)
                if process:
                    print(f"      → {process['name']} (PID: {process['pid']})")
        
        # Test de connectivité
        current_url = pm.get_current_searx_url()
        if current_url:
            print(f"\n🌐 Test de connectivité: {current_url}")
            searx = SearxInterface(current_url)
            if searx.check_health():
                print("✅ Searx fonctionne correctement")
            else:
                print("❌ Searx ne répond pas")
        else:
            print("\n❌ Aucune instance Searx détectée")
            
    except Exception as e:
        print(f"❌ Erreur lors de la vérification: {e}")

def stop_all():
    """Arrête toutes les instances Searx"""
    try:
        from port_manager import PortManager
        
        pm = PortManager()
        success = pm.stop_all_searx_containers()
        
        if success:
            print("✅ Toutes les instances Searx arrêtées")
        else:
            print("❌ Erreur lors de l'arrêt")
            
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "status":
            show_status()
        elif command == "stop":
            stop_all()
        elif command == "start":
            main()
        else:
            print("Commandes disponibles:")
            print("  python searx_smart_start.py start   # Démarre Searx")
            print("  python searx_smart_start.py status  # Affiche l'état")
            print("  python searx_smart_start.py stop    # Arrête tout")
    else:
        # Démarrage par défaut
        main()
