#!/usr/bin/env python3
"""
Simple test script to validate the Searx system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_api import get_gemini_response, get_searx_status

def test_simple_integration():
    """Simple test of Searx integration"""

    print("🔧 Simple validation test of Searx integration")
    print("=" * 60)

    # 1. Check status
    print("\n1. Searx Status:")
    status = get_searx_status()
    print(f"   {status}")

    # 2. Test a simple question that triggers Searx
    print("\n2. Test with Searx-triggering question:")
    test_question = "What's the latest news in technology?"

    print(f"   Question: {test_question}")
    print("   Processing...")

    response = get_gemini_response(test_question)

    print(f"   Status: {response['status']}")
    print(f"   Response length: {len(response['response'])} characters")
    print(f"   Start of response: {response['response'][:200]}...")

    # 3. Test a normal question without Searx
    print("\n3. Test with normal question (without Searx):")
    normal_question = "How are you today?"

    print(f"   Question: {normal_question}")
    response2 = get_gemini_response(normal_question)

    print(f"   Status: {response2['status']}")
    print(f"   Response length: {len(response2['response'])} characters")
    print(f"   Start of response: {response2['response'][:200]}...")

    print("\n✅ Validation tests completed!")

if __name__ == "__main__":
    test_simple_integration()
