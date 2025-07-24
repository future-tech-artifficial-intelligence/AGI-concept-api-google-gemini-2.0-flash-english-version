"""
Intégration Simplifiée du Web Scraping Autonome
Ce module gère l'intégration directe du web scraping avec l'IA.
Toutes les données sont sauvegardées automatiquement dans des fichiers texte.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
from autonomous_web_scraper import autonomous_web_scraper

logger = logging.getLogger(__name__)

class SimpleWebLearningIntegration:
    """Intégration simplifiée pour l'apprentissage web autonome"""

    def __init__(self):
        self.integration_active = True
        self.auto_learning_enabled = True
        self.learning_interval = 300  # 5 minutes entre les sessions autonomes
        self.last_learning_session = None

        # Répertoire de monitoring
        self.monitor_dir = Path("data/web_learning_monitor")
        self.monitor_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Intégration Web-Apprentissage Simplifiée initialisée")

    def trigger_autonomous_learning_if_needed(self) -> Dict[str, Any]:
        """Déclenche l'apprentissage autonome si nécessaire"""

        if not self.auto_learning_enabled:
            return {"triggered": False, "reason": "Apprentissage automatique désactivé"}

        current_time = time.time()

        # L'IA peut maintenant apprendre quand elle le souhaite
        # Plus de limitation temporelle stricte
        if (self.last_learning_session and 
            current_time - self.last_learning_session < 60):  # Seulement 1 minute minimum
            return {
                "triggered": True,  # Autorisé même si récent
                "reason": "Apprentissage autonome autorisé en continu"
            }

        # Déclencher une session d'apprentissage
        logger.info("Déclenchement d'une session d'apprentissage autonome")

        session_result = autonomous_web_scraper.start_autonomous_learning()
        self.last_learning_session = current_time

        # Enregistrer l'activité
        self._log_learning_activity(session_result)

        return {
            "triggered": True,
            "session_result": session_result,
            "next_session_in": self.learning_interval
        }

    def force_learning_session(self) -> Dict[str, Any]:
        """Force le déclenchement d'une session d'apprentissage"""
        logger.info("Session d'apprentissage forcée")

        session_result = autonomous_web_scraper.start_autonomous_learning()
        self.last_learning_session = time.time()

        self._log_learning_activity(session_result)

        return {
            "forced": True,
            "session_result": session_result
        }

    def _log_learning_activity(self, session_result: Dict[str, Any]) -> None:
        """Enregistre l'activité d'apprentissage"""

        activity_file = self.monitor_dir / "learning_activity.txt"

        try:
            with open(activity_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"SESSION D'APPRENTISSAGE AUTONOME\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
                f.write(f"ID de session: {session_result.get('session_id', 'Non défini')}\n")
                f.write(f"Succès: {session_result.get('success', False)}\n")
                f.write(f"Pages traitées: {session_result.get('pages_processed', 0)}\n")
                f.write(f"Domaine principal: {session_result.get('domain_focus', 'Non spécifié')}\n")

                if session_result.get('files_created'):
                    f.write(f"Fichiers créés: {len(session_result['files_created'])}\n")
                    for file_path in session_result['files_created']:
                        f.write(f"  - {file_path}\n")

                if session_result.get('error'):
                    f.write(f"Erreur: {session_result['error']}\n")

                f.write(f"{'='*60}\n")

        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'activité: {str(e)}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Retourne le statut de l'intégration"""

        scraper_status = autonomous_web_scraper.get_learning_status()

        return {
            "integration_active": self.integration_active,
            "auto_learning_enabled": self.auto_learning_enabled,
            "learning_interval_hours": self.learning_interval / 3600,
            "last_session_timestamp": self.last_learning_session,
            "scraper_status": scraper_status,
            "monitor_directory": str(self.monitor_dir)
        }

    def enable_auto_learning(self) -> Dict[str, Any]:
        """Active l'apprentissage automatique"""
        self.auto_learning_enabled = True
        logger.info("Apprentissage automatique activé")
        return {"status": "Apprentissage automatique activé"}

    def disable_auto_learning(self) -> Dict[str, Any]:
        """Désactive l'apprentissage automatique"""
        self.auto_learning_enabled = False
        logger.info("Apprentissage automatique désactivé")
        return {"status": "Apprentissage automatique désactivé"}

    def set_learning_interval(self, hours: float) -> Dict[str, Any]:
        """Définit l'intervalle entre les sessions d'apprentissage"""
        self.learning_interval = int(hours * 3600)
        logger.info(f"Intervalle d'apprentissage défini à {hours} heures")
        return {"status": f"Intervalle défini à {hours} heures"}

# Instance globale
web_learning_integration = SimpleWebLearningIntegration()

# Fonctions d'interface publique
def trigger_autonomous_learning() -> Dict[str, Any]:
    """Interface publique pour déclencher l'apprentissage autonome"""
    return web_learning_integration.trigger_autonomous_learning_if_needed()

def force_web_learning_session() -> Dict[str, Any]:
    """Interface publique pour forcer une session d'apprentissage"""
    return web_learning_integration.force_learning_session()

def get_web_learning_integration_status() -> Dict[str, Any]:
    """Interface publique pour obtenir le statut d'intégration"""
    return web_learning_integration.get_integration_status()

def enable_autonomous_learning() -> Dict[str, Any]:
    """Interface publique pour activer l'apprentissage autonome"""
    return web_learning_integration.enable_auto_learning()

def disable_autonomous_learning() -> Dict[str, Any]:
    """Interface publique pour désactiver l'apprentissage autonome"""
    return web_learning_integration.disable_auto_learning()

if __name__ == "__main__":
    print("=== Test de l'Intégration Web-Apprentissage Simplifiée ===")

    # Forcer une session d'apprentissage
    result = force_web_learning_session()
    print(f"Session forcée: {result.get('forced', False)}")

    if result.get('session_result', {}).get('success'):
        session = result['session_result']
        print(f"✓ Pages traitées: {session['pages_processed']}")
        print(f"✓ Domaine: {session.get('domain_focus', 'Non spécifié')}")
        print(f"✓ Fichiers créés: {len(session.get('files_created', []))}")

    # Afficher le statut
    status = get_web_learning_integration_status()
    print(f"\n=== Statut de l'Intégration ===")
    print(f"Intégration active: {status['integration_active']}")
    print(f"Apprentissage auto: {status['auto_learning_enabled']}")
    print(f"Intervalle: {status['learning_interval_hours']} heures")
    print(f"Sessions completées: {status['scraper_status']['sessions_completed']}")