#!/usr/bin/env python3
"""
Searx Interface Robustness Test
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from searx_interface import SearxInterface
import time

def test_searx_robustness():
    """Tests Searx robustness"""
    
    print("🔧 Searx Interface Robustness Test")
    print("=" * 60)
    
    # Create a SearxInterface instance
    searx = SearxInterface()
    
    # 1. Basic status test
    print("\n1. Searx status test:")
    status = searx.check_health()
    print(f"   Searx operational: {'✅ Yes' if status else '❌ No'}")
    
    if not status:
        print("   Attempting to start Searx...")
        start_result = searx.start_searx()
        print(f"   Startup successful: {'✅ Yes' if start_result else '❌ No'}")
        
        if start_result:
            print("   Waiting 10 seconds for Searx to be ready...")
            time.sleep(10)
            status = searx.check_health()
            print(f"   Searx operational after startup: {'✅ Yes' if status else '❌ No'}")
    
    # 2. Simple search test
    print("\n2. Simple search test:")
    query = "simple test"
    print(f"   Search: '{query}'")
    
    results = searx.search(query, max_results=3)
    print(f"   Results obtained: {len(results)}")
    
    for i, result in enumerate(results[:2], 1):
        print(f"   Result {i}: {result.title[:50]}...")
        print(f"   URL: {result.url}")
    
    # 3. Search with retry test
    print("\n3. Search with retry test:")
    query2 = "artificial intelligence news"
    print(f"   Search: '{query2}'")
    
    start_time = time.time()
    results2 = searx.search(query2, category="general", max_results=5, retry_count=3)
    end_time = time.time()
    
    print(f"   Results obtained: {len(results2)}")
    print(f"   Search time: {end_time - start_time:.2f} seconds")
    
    if results2:
        print("   ✅ Search with retry successful")
        for i, result in enumerate(results2[:2], 1):
            print(f"   Result {i}: {result.title[:50]}...")
            print(f"   URL: {result.url[:80]}...")
    else:
        print("   ❌ No results obtained even with retry")
    
    print("\n✅ Robustness test completed!")

if __name__ == "__main__":
    test_searx_robustness()
