#!/usr/bin/env python3
"""
Test script for Searx integration in the artificial intelligence API GOOGLE GEMINI 2.0 FLASH
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api import get_gemini_response, get_searx_status, trigger_searx_search_session

def test_searx_integration():
    """Test Searx integration"""
    print("🔧 Testing Searx integration in the artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
    print("=" * 60)

    # Test 1: Searx status
    print("\n1. Searx status test:")
    status = get_searx_status()
    print(status)

    # Test 2: Manual search
    print("\n2. Manual search test:")
    search_result = trigger_searx_search_session("latest AI news")
    print(search_result)

    # Test 3: Request requiring automatic search
    print("\n3. Request test with automatic search:")
    response = get_gemini_response("What is the latest news in artificial intelligence?")
    print(f"Response: {response['response'][:200]}...")
    print(f"Status: {response['status']}")

    # Test 4: Technical query
    print("\n4. Technical query test:")
    response = get_gemini_response("What are the new features of Python 3.12?")
    print(f"Response: {response['response'][:200]}...")
    print(f"Status: {response['status']}")

    print("\n✅ Tests completed!")

if __name__ == "__main__":
    test_searx_integration()
