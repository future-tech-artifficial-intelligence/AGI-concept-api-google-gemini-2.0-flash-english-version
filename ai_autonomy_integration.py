"""
Système d'Autonomie IA Simplifié
Ce module gère l'autonomie de l'IA avec accès web direct sans programmation d'actions.
"""

import json
import logging
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Imports pour l'accès web autonome
try:
    from web_learning_integration import trigger_autonomous_learning, get_web_learning_integration_status
    from autonomous_web_scraper import get_autonomous_learning_status
    WEB_LEARNING_AVAILABLE = True
except ImportError:
    WEB_LEARNING_AVAILABLE = False

# Import pour l'accès aux fichiers
try:
    from direct_file_access import get_all_project_files, search_files
    FILE_ACCESS_AVAILABLE = True
except ImportError:
    FILE_ACCESS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleAIAutonomy:
    """Système d'autonomie IA simplifié avec accès web direct"""

    def __init__(self):
        self.autonomy_active = True
        self.web_access_enabled = True
        self.file_access_enabled = True

        # Répertoires de données
        self.data_dir = Path("data/ai_autonomy")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Mémoire d'interaction
        self.interaction_memory = {
            "total_interactions": 0,
            "web_learning_sessions": [],
            "file_access_requests": [],
            "autonomous_actions": [],
            "last_update": datetime.now().isoformat()
        }

        # Charger la mémoire existante
        self._load_interaction_memory()

        logger.info("Système d'autonomie IA simplifié initialisé")

    def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Traite l'entrée utilisateur avec autonomie complète"""

        self.interaction_memory["total_interactions"] += 1

        result = {
            "input_processed": True,
            "timestamp": datetime.now().isoformat(),
            "actions_taken": [],
            "web_access_triggered": False,
            "files_accessed": [],
            "autonomous_decisions": []
        }

        try:
            # Analyser l'entrée pour détecter les besoins
            needs_analysis = self._analyze_user_needs(user_input)

            # Accès web autonome si nécessaire
            if needs_analysis.get("needs_web_info") and self.web_access_enabled:
                web_result = self._trigger_autonomous_web_access()
                result["web_access_triggered"] = web_result.get("triggered", False)
                result["actions_taken"].append("web_access")

                if web_result.get("session_result"):
                    self.interaction_memory["web_learning_sessions"].append({
                        "timestamp": time.time(),
                        "trigger_reason": "user_input_analysis",
                        "user_input": user_input[:100],  # Limiter la taille
                        "session_data": web_result["session_result"]
                    })

            # Accès aux fichiers si nécessaire
            if needs_analysis.get("needs_file_access") and self.file_access_enabled:
                file_results = self._autonomous_file_access(needs_analysis.get("search_terms", []))
                result["files_accessed"] = file_results
                result["actions_taken"].append("file_access")

                self.interaction_memory["file_access_requests"].append({
                    "timestamp": time.time(),
                    "search_terms": needs_analysis.get("search_terms", []),
                    "files_found": len(file_results)
                })

            # Enregistrer les décisions autonomes prises
            result["autonomous_decisions"] = needs_analysis.get("decisions", [])

            # Sauvegarder la mémoire
            self._save_interaction_memory()

        except Exception as e:
            logger.error(f"Erreur lors du traitement autonome: {str(e)}")
            result["error"] = str(e)

        return result

    def _analyze_user_needs(self, user_input: str) -> Dict[str, Any]:
        """Analyse les besoins de l'utilisateur de manière autonome"""

        user_input_lower = user_input.lower()

        analysis = {
            "needs_web_info": False,
            "needs_file_access": False,
            "search_terms": [],
            "decisions": []
        }

        # Détecter le besoin d'informations web
        web_indicators = [
            "information", "recherche", "actualité", "nouveau", "récent",
            "qu'est-ce que", "comment", "pourquoi", "tendance", "innovation"
        ]

        if any(indicator in user_input_lower for indicator in web_indicators):
            analysis["needs_web_info"] = True
            analysis["decisions"].append("Détection du besoin d'informations web")

        # Détecter le besoin d'accès aux fichiers
        file_indicators = [
            "fichier", "code", "fonction", "module", "projet", "système",
            "comment fonctionne", "où se trouve", "montre-moi"
        ]

        if any(indicator in user_input_lower for indicator in file_indicators):
            analysis["needs_file_access"] = True
            analysis["decisions"].append("Détection du besoin d'accès aux fichiers")

            # Extraire les termes de recherche
            import re
            # Mots significatifs pour la recherche
            words = re.findall(r'\b\w{3,}\b', user_input_lower)
            analysis["search_terms"] = [w for w in words if w not in [
                "comment", "fonctionne", "montre", "moi", "que", "est", "une", "des"
            ]][:5]

        return analysis

    def _trigger_autonomous_web_access(self) -> Dict[str, Any]:
        """Déclenche l'accès web autonome"""

        if not WEB_LEARNING_AVAILABLE:
            return {"triggered": False, "reason": "Module web non disponible"}

        try:
            result = trigger_autonomous_learning()
            logger.info(f"Accès web autonome: {result.get('triggered', False)}")
            return result
        except Exception as e:
            logger.error(f"Erreur lors de l'accès web autonome: {str(e)}")
            return {"triggered": False, "error": str(e)}

    def _autonomous_file_access(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Accès autonome aux fichiers du projet"""

        if not FILE_ACCESS_AVAILABLE:
            return []

        try:
            file_results = []

            # Rechercher pour chaque terme
            for term in search_terms:
                search_result = search_files(term)
                if search_result.get("results"):
                    file_results.extend(search_result["results"][:3])  # Limiter à 3 par terme

            # Supprimer les doublons
            seen_files = set()
            unique_results = []
            for result in file_results:
                file_path = result.get("file_path", "")
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    unique_results.append(result)

            logger.info(f"Accès fichiers autonome: {len(unique_results)} fichiers trouvés")
            return unique_results[:10]  # Limiter à 10 résultats

        except Exception as e:
            logger.error(f"Erreur lors de l'accès aux fichiers: {str(e)}")
            return []

    def get_autonomy_status(self) -> Dict[str, Any]:
        """Retourne le statut du système d'autonomie"""

        # Statut du web learning si disponible
        web_status = {}
        if WEB_LEARNING_AVAILABLE:
            try:
                web_status = get_web_learning_integration_status()
            except:
                web_status = {"error": "Module web inaccessible"}

        return {
            "autonomy_active": self.autonomy_active,
            "web_access_enabled": self.web_access_enabled,
            "file_access_enabled": self.file_access_enabled,
            "total_interactions": self.interaction_memory["total_interactions"],
            "web_sessions_count": len(self.interaction_memory["web_learning_sessions"]),
            "file_requests_count": len(self.interaction_memory["file_access_requests"]),
            "web_learning_status": web_status,
            "last_update": self.interaction_memory["last_update"]
        }

    def _load_interaction_memory(self) -> None:
        """Charge la mémoire d'interaction"""
        memory_file = self.data_dir / "interaction_memory.json"

        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    loaded_memory = json.load(f)
                    self.interaction_memory.update(loaded_memory)
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la mémoire: {str(e)}")

    def _save_interaction_memory(self) -> None:
        """Sauvegarde la mémoire d'interaction"""
        memory_file = self.data_dir / "interaction_memory.json"

        try:
            self.interaction_memory["last_update"] = datetime.now().isoformat()
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.interaction_memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la mémoire: {str(e)}")

    def enable_web_access(self) -> Dict[str, Any]:
        """Active l'accès web autonome"""
        self.web_access_enabled = True
        logger.info("Accès web autonome activé")
        return {"status": "Accès web autonome activé"}

    def disable_web_access(self) -> Dict[str, Any]:
        """Désactive l'accès web autonome"""
        self.web_access_enabled = False
        logger.info("Accès web autonome désactivé")
        return {"status": "Accès web autonome désactivé"}

# Instance globale
ai_autonomy = SimpleAIAutonomy()

# Fonctions d'interface publique
def process_input(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Interface publique pour traiter l'entrée utilisateur avec autonomie"""
    return ai_autonomy.process_user_input(user_input, context)

def get_status() -> Dict[str, Any]:
    """Interface publique pour obtenir le statut d'autonomie"""
    return ai_autonomy.get_autonomy_status()

def enable_autonomous_web_access() -> Dict[str, Any]:
    """Interface publique pour activer l'accès web autonome"""
    return ai_autonomy.enable_web_access()

def disable_autonomous_web_access() -> Dict[str, Any]:
    """Interface publique pour désactiver l'accès web autonome"""
    return ai_autonomy.disable_web_access()

if __name__ == "__main__":
    print("=== Test du Système d'Autonomie IA Simplifié ===")

    # Test de traitement d'entrée
    test_input = "Comment fonctionne l'intelligence artificielle moderne ?"
    result = process_input(test_input)

    print(f"Entrée traitée: {result['input_processed']}")
    print(f"Actions prises: {result['actions_taken']}")
    print(f"Accès web déclenché: {result['web_access_triggered']}")
    print(f"Fichiers accédés: {len(result['files_accessed'])}")

    # Statut du système
    status = get_status()
    print(f"\n=== Statut du Système ===")
    print(f"Autonomie active: {status['autonomy_active']}")
    print(f"Accès web activé: {status['web_access_enabled']}")
    print(f"Total interactions: {status['total_interactions']}")
    print(f"Sessions web: {status['web_sessions_count']}")