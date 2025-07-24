#!/usr/bin/env python3
"""
Lanceur automatique : Searx + Application
Lance automatiquement Searx puis démarre python app.py
"""

import sys
import os
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AutoLauncher')

def main():
    """Lance automatiquement Searx puis l'application"""
    
    print("\n🚀 LANCEUR AUTOMATIQUE - SEARX + APPLICATION")
    print("=" * 60)
    
    try:
        # Étape 1 : Démarrer Searx automatiquement
        logger.info("🔧 Phase 1: Démarrage automatique de Searx...")
        
        try:
            from searx_auto_starter import SearxAutoStarter
            auto_starter = SearxAutoStarter()
            searx_ready = auto_starter.start_complete_system()
            
            if searx_ready:
                logger.info("✅ Searx prêt!")
            else:
                logger.info("⚠️ Searx en mode dégradé")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur Searx: {e}")
            logger.info("🔄 Continuation sans Searx")
        
        # Étape 2 : Lancer l'application
        logger.info("🌐 Phase 2: Lancement de l'application...")
        
        print("\n" + "=" * 60)
        print("🎉 DÉMARRAGE DE L'APPLICATION")
        print("=" * 60)
        
        # Importer et lancer l'application Flask
        from app import app
        
        print("🌐 Interface web: http://localhost:5000")
        print("🤖 IA Gemini avec recherches autonomes")
        print("📸 Analyse visuelle disponible")
        print("🔧 Gestion intelligente des ports")
        print("=" * 60)
        print("💡 Pour arrêter: Ctrl+C")
        print()
        
        # Lancer l'application
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Arrêt demandé par l'utilisateur")
        
        # Nettoyer Searx
        try:
            from port_manager import get_port_manager
            pm = get_port_manager()
            pm.stop_all_searx_containers()
            logger.info("✅ Nettoyage Searx terminé")
        except:
            pass
        
        print("👋 Application arrêtée proprement!")
        
    except Exception as e:
        logger.error(f"💥 Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
