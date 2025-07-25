#!/usr/bin/env python3
"""
Test of the artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter with Searx
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api_adapter import GeminiAPI

def test_adapter():
    """Tests the adapter with Searx"""
    
    print("🔧 Testing the artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter with Searx")
    print("=" * 60)
    
    # Create an adapter instance
    gemini_adapter = GeminiAPI()
    
    # Test with a question that should trigger Searx
    print("\n1. Test with question requiring Searx:")
    test_question = "What are the latest news in artificial intelligence?"
    
    print(f"   Question: {test_question}")
    print("   Processing...")
    
    response = gemini_adapter.get_response(test_question, user_id=1, session_id="test_session")
    
    print(f"   Status: {response['status']}")
    print(f"   Response length: {len(response['response'])} characters")
    print(f"   Start of response: {response['response'][:300]}...")
    
    # Check if actual URLs are present
    if "https://" in response['response'] or "http://" in response['response']:
        print("\n   ✅ URLs detected in the response")
    else:
        print("\n   ⚠️ No URLs detected in the response")
    
    print("\n✅ Adapter test finished!")

if __name__ == "__main__":
    test_adapter()
