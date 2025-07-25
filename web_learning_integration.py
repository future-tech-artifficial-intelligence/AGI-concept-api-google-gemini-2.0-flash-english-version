"""
Simplified Autonomous Web Scraping Integration
This module manages the direct integration of web scraping with artificial intelligence API GOOGLE GEMINI 2.0 FLASH
All data is automatically saved to text files. artificial intelligence API GOOGLE GEMINI 2.0 FLASH no
longer uses this web scraping module. artificial intelligence API GOOGLE GEMINI 2.0 FLASH uses the Searx engine, which allows artificial intelligence API GOOGLE GEMINI 2.0 FLASH
to learn autonomously during its searches. The Searx search engine is more powerful than simple HTML page extraction like web scraping;
it allows artificial intelligence API GOOGLE GEMINI 2.0 FLASH to search on search engines like Google, Bing, and others
for searches with better accuracy.
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
    """Simplified integration for autonomous web learning"""

    def __init__(self):
        self.integration_active = True
        self.auto_learning_enabled = True
        self.learning_interval = 300  # 5 minutes between autonomous sessions
        self.last_learning_session = None

        # Monitoring directory
        self.monitor_dir = Path("data/web_learning_monitor")
        self.monitor_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Simplified Web-Learning Integration initialized")

    def trigger_autonomous_learning_if_needed(self) -> Dict[str, Any]:
        """Triggers autonomous learning if needed"""

        if not self.auto_learning_enabled:
            return {"triggered": False, "reason": "Automatic learning disabled"}

        current_time = time.time()

        # The AI can now learn whenever it wishes
        # No more strict time limitation
        if (self.last_learning_session and
            current_time - self.last_learning_session < 60):  # Only 1 minute minimum
            return {
                "triggered": True,  # Allowed even if recent
                "reason": "Autonomous learning allowed continuously"
            }

        # Trigger a learning session
        logger.info("Triggering an autonomous learning session")

        session_result = autonomous_web_scraper.start_autonomous_learning()
        self.last_learning_session = current_time

        # Log activity
        self._log_learning_activity(session_result)

        return {
            "triggered": True,
            "session_result": session_result,
            "next_session_in": self.learning_interval
        }

    def force_learning_session(self) -> Dict[str, Any]:
        """Forces the triggering of a learning session"""
        logger.info("Forcing learning session")

        session_result = autonomous_web_scraper.start_autonomous_learning()
        self.last_learning_session = time.time()

        self._log_learning_activity(session_result)

        return {
            "forced": True,
            "session_result": session_result
        }

    def _log_learning_activity(self, session_result: Dict[str, Any]) -> None:
        """Logs learning activity"""

        activity_file = self.monitor_dir / "learning_activity.txt"

        try:
            with open(activity_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"AUTONOMOUS LEARNING SESSION\n")
                f.write(f"Date: {datetime.now().isoformat()}\n")
                f.write(f"Session ID: {session_result.get('session_id', 'Not defined')}\n")
                f.write(f"Success: {session_result.get('success', False)}\n")
                f.write(f"Pages processed: {session_result.get('pages_processed', 0)}\n")
                f.write(f"Main domain: {session_result.get('domain_focus', 'Not specified')}\n")

                if session_result.get('files_created'):
                    f.write(f"Files created: {len(session_result['files_created'])}\n")
                    for file_path in session_result['files_created']:
                        f.write(f"  - {file_path}\n")

                if session_result.get('error'):
                    f.write(f"Error: {session_result['error']}\n")

                f.write(f"{'='*60}\n")

        except Exception as e:
            logger.error(f"Error logging activity: {str(e)}")

    def get_integration_status(self) -> Dict[str, Any]:
        """Returns the integration status"""

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
        """Enables automatic learning"""
        self.auto_learning_enabled = True
        logger.info("Automatic learning enabled")
        return {"status": "Automatic learning enabled"}

    def disable_auto_learning(self) -> Dict[str, Any]:
        """Disables automatic learning"""
        self.auto_learning_enabled = False
        logger.info("Automatic learning disabled")
        return {"status": "Automatic learning disabled"}

    def set_learning_interval(self, hours: float) -> Dict[str, Any]:
        """Sets the interval between learning sessions"""
        self.learning_interval = int(hours * 3600)
        logger.info(f"Learning interval set to {hours} hours")
        return {"status": f"Interval set to {hours} hours"}

# Global instance
web_learning_integration = SimpleWebLearningIntegration()

# Public interface functions
def trigger_autonomous_learning() -> Dict[str, Any]:
    """Public interface to trigger autonomous learning"""
    return web_learning_integration.trigger_autonomous_learning_if_needed()

def force_web_learning_session() -> Dict[str, Any]:
    """Public interface to force a learning session"""
    return web_learning_integration.force_learning_session()

def get_web_learning_integration_status() -> Dict[str, Any]:
    """Public interface to get integration status"""
    return web_learning_integration.get_integration_status()

def enable_autonomous_learning() -> Dict[str, Any]:
    """Public interface to enable autonomous learning"""
    return web_learning_integration.enable_auto_learning()

def disable_autonomous_learning() -> Dict[str, Any]:
    """Public interface to disable autonomous learning"""
    return web_learning_integration.disable_auto_learning()

if __name__ == "__main__":
    print("=== Simple Web-Learning Integration Test ===")

    # Force a learning session
    result = force_web_learning_session()
    print(f"Session forced: {result.get('forced', False)}")

    if result.get('session_result', {}).get('success'):
        session = result['session_result']
        print(f"✓ Pages processed: {session['pages_processed']}")
        print(f"✓ Domain: {session.get('domain_focus', 'Not specified')}")
        print(f"✓ Files created: {len(session.get('files_created', []))}")

    # Display status
    status = get_web_learning_integration_status()
    print(f"\n=== Integration Status ===")
    print(f"Integration active: {status['integration_active']}")
    print(f"Auto learning: {status['auto_learning_enabled']}")
    print(f"Interval: {status['learning_interval_hours']} hours")
    print(f"Sessions completed: {status['scraper_status']['sessions_completed']}")
