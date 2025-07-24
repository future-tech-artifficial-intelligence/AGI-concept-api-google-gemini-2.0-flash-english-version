#!/usr/bin/env python3
"""
Script de démarrage unifié pour l'application avec Searx intégré
Lance automatiquement tout le système puis démarre l'application Flask
"""

import sys
import logging
import time
import subprocess

# Configuration du logging pour le démarrage
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('searx_app_startup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('SearxAppStart')

def main():
    """Démarrage unifié de l'application avec Searx"""
    
    print("\n" + "=" * 80)
    print("🚀 DÉMARRAGE AUTOMATIQUE - APPLICATION SEARX AI")
    print("=" * 80)
    print()
    
    try:
        # Étape 1: Démarrage automatique de Searx
        logger.info("🔧 Phase 1: Préparation du système Searx...")
        
        try:
            from searx_auto_starter import SearxAutoStarter
            auto_starter = SearxAutoStarter()
            searx_ready = auto_starter.start_complete_system()
            
            if searx_ready:
                logger.info("✅ Système Searx prêt!")
            else:
                logger.warning("⚠️ Système Searx en mode dégradé")
                
        except Exception as e:
            logger.error(f"❌ Erreur préparation Searx: {e}")
            logger.info("🔄 Tentative de démarrage sans préparation automatique...")
        
        # Étape 2: Démarrage de l'application Flask
        logger.info("🌐 Phase 2: Démarrage de l'application Flask...")
        
        # Importer l'application Flask
        from app import app
        
        # Afficher les informations de démarrage
        print("\n" + "=" * 80)
        print("🎉 APPLICATION SEARX AI DÉMARRÉE!")
        print("=" * 80)
        print()
        print("🌐 Interface web accessible sur:")
        print("   - Local: http://127.0.0.1:5000")
        print("   - Réseau: http://localhost:5000")
        print()
        print("🔍 Fonctionnalités disponibles:")
        print("   ✅ Chat IA avec Gemini")
        print("   ✅ Recherches web autonomes (si Searx actif)")
        print("   ✅ Analyse d'images")
        print("   ✅ Capture visuelle des recherches")
        print("   ✅ Gestion intelligente des ports")
        print()
        print("💡 Pour arrêter: Ctrl+C dans cette fenêtre")
        print("=" * 80)
        
        # Démarrer l'application Flask
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Pas de debug pour éviter le double démarrage
            use_reloader=False  # Pas de reload automatique
        )
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Arrêt de l'application demandé par l'utilisateur")
        
        # Nettoyer Searx si nécessaire
        try:
            from port_manager import get_port_manager
            port_manager = get_port_manager()
            logger.info("🧹 Nettoyage des conteneurs Searx...")
            port_manager.stop_all_searx_containers()
            logger.info("✅ Nettoyage terminé")
        except:
            pass
        
        print("\n👋 Application arrêtée proprement")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"💥 Erreur fatale: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
