#!/usr/bin/env python3
"""
Autonomous Web Scraping System Integration Test artificial intelligence API GOOGLE GEMINI 2.0 FLASH uses the Searx search engine for greater accuracy in internet searches;
web scraping is no longer used
"""

def test_web_scraping_integration():
    """Comprehensive test of web scraping integration"""
    print("🧪 Autonomous web scraping system integration test")
    
    try:
        # Test 1: Module imports
        from autonomous_web_scraper import start_autonomous_web_learning, get_autonomous_learning_status
        from web_learning_integration import trigger_autonomous_learning, force_web_learning_session
        print("✅ Web scraping modules imported successfully")
        
        # Test 2: Check system status
        status = get_autonomous_learning_status()
        print(f"✅ System status: {status.get('autonomous_learning_active', False)}")
        
        # Test 3: Learning session test (short)
        print("🔍 Triggering a test session...")
        result = force_web_learning_session()
        
        if result.get("forced") and result.get("session_result", {}).get("success"):
            session = result["session_result"]
            print(f"✅ Session successful:")
            print(f"   - Pages processed: {session.get('pages_processed', 0)}")
            print(f"   - Domain: {session.get('domain_focus', 'Not specified')}")
            print(f"   - Files created: {len(session.get('files_created', []))}")
            return True
        else:
            print("⚠️ Inconclusive test session")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_web_scraping_integration()
    if success:
        print("\n🎉 The autonomous web scraping system is operational!")
        print("The AI can now perform autonomous web searches.")
    else:
        print("\n⚠️ Issues detected with the web scraping system.")
