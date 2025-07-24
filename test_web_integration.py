#!/usr/bin/env python3
"""
Script de test pour vérifier l'intégration du système de web scraping autonome
"""

def test_web_integration():
    """Test de base pour l'intégration web"""
    try:
        # Test de déclenchement d'une session d'apprentissage web
        from web_learning_integration import SimpleWebLearningIntegration
        from autonomous_web_scraper import AutonomousWebScraper
        from intelligent_web_navigator import SimpleWebNavigator
        
        # Test du scraper autonome
        scraper = AutonomousWebScraper()
        print("✅ AutonomousWebScraper initialisé avec succès")

        # Test du système d'apprentissage web
        learning_system = SimpleWebLearningIntegration()
        print("✅ SimpleWebLearningIntegration initialisé avec succès")

        # Test du navigateur intelligent
        navigator = SimpleWebNavigator(scraper)
        print("✅ SimpleWebNavigator initialisé avec succès")

        print("🌐 Système de Web Scraping Autonome opérationnel")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du test d'intégration web: {e}")
        return False

if __name__ == "__main__":
    test_web_integration()