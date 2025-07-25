"""
**Simplified artificial intelligence API GOOGLE GEMINI 2.0 FLASH Autonomy System**
This module manages the artificial intelligence API GOOGLE GEMINI 2.0 FLASH's autonomy with direct web access without programming actions.
"""

import json
import logging
import time
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Imports for autonomous web access
try:
    from web_learning_integration import trigger_autonomous_learning, get_web_learning_integration_status
    from autonomous_web_scraper import get_autonomous_learning_status
    WEB_LEARNING_AVAILABLE = True
except ImportError:
    WEB_LEARNING_AVAILABLE = False

# Import for file access
try:
    from direct_file_access import get_all_project_files, search_files
    FILE_ACCESS_AVAILABLE = True
except ImportError:
    FILE_ACCESS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleAIAutonomy:
    """Simplified AI Autonomy System with direct web access"""

    def __init__(self):
        self.autonomy_active = True
        self.web_access_enabled = True
        self.file_access_enabled = True

        # Data directories
        self.data_dir = Path("data/ai_autonomy")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Interaction memory
        self.interaction_memory = {
            "total_interactions": 0,
            "web_learning_sessions": [],
            "file_access_requests": [],
            "autonomous_actions": [],
            "last_update": datetime.now().isoformat()
        }

        # Load existing memory
        self._load_interaction_memory()

        logger.info("Simplified AI autonomy system initialized")

    def process_user_input(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Processes user input with full autonomy"""

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
            # Analyze input to detect needs
            needs_analysis = self._analyze_user_needs(user_input)

            # Autonomous web access if needed
            if needs_analysis.get("needs_web_info") and self.web_access_enabled:
                web_result = self._trigger_autonomous_web_access()
                result["web_access_triggered"] = web_result.get("triggered", False)
                result["actions_taken"].append("web_access")

                if web_result.get("session_result"):
                    self.interaction_memory["web_learning_sessions"].append({
                        "timestamp": time.time(),
                        "trigger_reason": "user_input_analysis",
                        "user_input": user_input[:100],  # Limit size
                        "session_data": web_result["session_result"]
                    })

            # File access if needed
            if needs_analysis.get("needs_file_access") and self.file_access_enabled:
                file_results = self._autonomous_file_access(needs_analysis.get("search_terms", []))
                result["files_accessed"] = file_results
                result["actions_taken"].append("file_access")

                self.interaction_memory["file_access_requests"].append({
                    "timestamp": time.time(),
                    "search_terms": needs_analysis.get("search_terms", []),
                    "files_found": len(file_results)
                })

            # Record autonomous decisions made
            result["autonomous_decisions"] = needs_analysis.get("decisions", [])

            # Save memory
            self._save_interaction_memory()

        except Exception as e:
            logger.error(f"Error during autonomous processing: {str(e)}")
            result["error"] = str(e)

        return result

    def _analyze_user_needs(self, user_input: str) -> Dict[str, Any]:
        """Analyzes user needs autonomously"""

        user_input_lower = user_input.lower()

        analysis = {
            "needs_web_info": False,
            "needs_file_access": False,
            "search_terms": [],
            "decisions": []
        }

        # Detect need for web information
        web_indicators = [
            "information", "research", "news", "new", "recent",
            "what is", "how", "why", "trend", "innovation"
        ]

        if any(indicator in user_input_lower for indicator in web_indicators):
            analysis["needs_web_info"] = True
            analysis["decisions"].append("Detection of need for web information")

        # Detect need for file access
        file_indicators = [
            "file", "code", "function", "module", "project", "system",
            "how it works", "where is", "show me"
        ]

        if any(indicator in user_input_lower for indicator in file_indicators):
            analysis["needs_file_access"] = True
            analysis["decisions"].append("Detection of need for file access")

            # Extract search terms
            import re
            # Significant words for search
            words = re.findall(r'\b\w{3,}\b', user_input_lower)
            analysis["search_terms"] = [w for w in words if w not in [
                "how", "works", "show", "me", "what", "is", "a", "of"
            ]][:5]

        return analysis

    def _trigger_autonomous_web_access(self) -> Dict[str, Any]:
        """Triggers autonomous web access"""

        if not WEB_LEARNING_AVAILABLE:
            return {"triggered": False, "reason": "Web module not available"}

        try:
            result = trigger_autonomous_learning()
            logger.info(f"Autonomous web access: {result.get('triggered', False)}")
            return result
        except Exception as e:
            logger.error(f"Error during autonomous web access: {str(e)}")
            return {"triggered": False, "error": str(e)}

    def _autonomous_file_access(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Autonomous access to project files"""

        if not FILE_ACCESS_AVAILABLE:
            return []

        try:
            file_results = []

            # Search for each term
            for term in search_terms:
                search_result = search_files(term)
                if search_result.get("results"):
                    file_results.extend(search_result["results"][:3])  # Limit to 3 per term

            # Remove duplicates
            seen_files = set()
            unique_results = []
            for result in file_results:
                file_path = result.get("file_path", "")
                if file_path not in seen_files:
                    seen_files.add(file_path)
                    unique_results.append(result)

            logger.info(f"Autonomous file access: {len(unique_results)} files found")
            return unique_results[:10]  # Limit to 10 results

        except Exception as e:
            logger.error(f"Error during file access: {str(e)}")
            return []

    def get_autonomy_status(self) -> Dict[str, Any]:
        """Returns the autonomy system status"""

        # Web learning status if available
        web_status = {}
        if WEB_LEARNING_AVAILABLE:
            try:
                web_status = get_web_learning_integration_status()
            except:
                web_status = {"error": "Web module inaccessible"}

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
        """Loads interaction memory"""
        memory_file = self.data_dir / "interaction_memory.json"

        if memory_file.exists():
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    loaded_memory = json.load(f)
                    self.interaction_memory.update(loaded_memory)
            except Exception as e:
                logger.error(f"Error loading memory: {str(e)}")

    def _save_interaction_memory(self) -> None:
        """Saves interaction memory"""
        memory_file = self.data_dir / "interaction_memory.json"

        try:
            self.interaction_memory["last_update"] = datetime.now().isoformat()
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.interaction_memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")

    def enable_web_access(self) -> Dict[str, Any]:
        """Enables autonomous web access"""
        self.web_access_enabled = True
        logger.info("Autonomous web access enabled")
        return {"status": "Autonomous web access enabled"}

    def disable_web_access(self) -> Dict[str, Any]:
        """Disables autonomous web access"""
        self.web_access_enabled = False
        logger.info("Autonomous web access disabled")
        return {"status": "Autonomous web access disabled"}

# Global instance
ai_autonomy = SimpleAIAutonomy()

# Public interface functions
def process_input(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Public interface to process user input with autonomy"""
    return ai_autonomy.process_user_input(user_input, context)

def get_status() -> Dict[str, Any]:
    """Public interface to get autonomy status"""
    return ai_autonomy.get_autonomy_status()

def enable_autonomous_web_access() -> Dict[str, Any]:
    """Public interface to enable autonomous web access"""
    return ai_autonomy.enable_web_access()

def disable_autonomous_web_access() -> Dict[str, Any]:
    """Public interface to disable autonomous web access"""
    return ai_autonomy.disable_web_access()

if __name__ == "__main__":
    print("=== Simplified AI Autonomy System Test ===")

    # Test input processing
    test_input = "How does modern artificial intelligence work?"
    result = process_input(test_input)

    print(f"Input processed: {result['input_processed']}")
    print(f"Actions taken: {result['actions_taken']}")
    print(f"Web access triggered: {result['web_access_triggered']}")
    print(f"Files accessed: {len(result['files_accessed'])}")

    # System status
    status = get_status()
    print(f"\n=== System Status ===")
    print(f"Autonomy active: {status['autonomy_active']}")
    print(f"Web access enabled: {status['web_access_enabled']}")
    print(f"Total interactions: {status['total_interactions']}")
    print(f"Web sessions: {status['web_sessions_count']}")
