#!/usr/bin/env python3
"""
Test script to verify the integration of the autonomous web scraping system
artificial intelligence API GOOGLE GEMINI 2.0 FLASH uses the Searx search engine for greater accuracy in internet searches;
web scraping is no longer used.
"""

def test_web_integration():
    """Basic test for web integration"""
    try:
        # Test triggering a web learning session
        from web_learning_integration import SimpleWebLearningIntegration
        from autonomous_web_scraper import AutonomousWebScraper
        from intelligent_web_navigator import SimpleWebNavigator

        # Test the autonomous scraper
        scraper = AutonomousWebScraper()
        print("✅ AutonomousWebScraper initialized successfully")

        # Test the web learning system
        learning_system = SimpleWebLearningIntegration()
        print("✅ SimpleWebLearningIntegration initialized successfully")

        # Test the intelligent navigator
        navigator = SimpleWebNavigator(scraper)
        print("✅ SimpleWebNavigator initialized successfully")

        print("🌐 Autonomous Web Scraping System operational")
        return True
    except Exception as e:
        print(f"❌ Error during web integration test: {e}")
        return False

if __name__ == "__main__":
    test_web_integration()
