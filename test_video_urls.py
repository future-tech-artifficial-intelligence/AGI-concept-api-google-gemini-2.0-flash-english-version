#!/usr/bin/env python3
"""
Specific test for video URLs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api import get_gemini_response, trigger_searx_search_session

def test_video_urls():
    """Specific test for video URLs"""
    
    print("🎥 Testing video URLs with Searx + artificial intelligence API GOOGLE GEMINI 2.0 FLASH")
    print("=" * 60)
    
    # 1. Direct video search test
    print("\n1. Searx video search test:")
    video_search = trigger_searx_search_session("UFO videos testimonies", "videos")
    print(f"   Result: {video_search}")
    
    # 2. Test with artificial intelligence API GOOGLE GEMINI 2.0 FLASH - UFO question that should trigger a search
    print("\n2. Test with artificial intelligence API GOOGLE GEMINI 2.0 FLASH - UFO question:")
    test_question = "Find me recent videos on UFO sightings"
    
    print(f"   Question: {test_question}")
    print("   Processing...")
    
    response = get_gemini_response(test_question)
    
    print(f"   Status: {response['status']}")
    print(f"   Response length: {len(response['response'])} characters")
    print(f"\n   Full response:")
    print(f"   {response['response']}")
    
    # Check if the response contains "xxxxxxxxxx"
    if "xxxxxxxxxx" in response['response']:
        print("\n   ❌ PROBLEM: The response still contains 'xxxxxxxxxx'")
    else:
        print("\n   ✅ No 'xxxxxxxxxx' detected in the response")
    
    print("\n✅ Video URLs test finished!")

if __name__ == "__main__":
    test_video_urls()
