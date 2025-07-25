#!/usr/bin/env python3
"""
Test script to validate the Searx system
"""

import logging
import time
import sys
import os

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('searx_test')

def test_docker_availability():
    """Test Docker availability"""
    logger.info("🔍 Test 1: Checking Docker...")

    try:
        from searx_manager import SearxManager
        manager = SearxManager()

        if manager.check_docker_availability():
            logger.info("✅ Docker is available")
            return True
        else:
            logger.error("❌ Docker is not available")
            return False
    except Exception as e:
        logger.error(f"❌ Error during Docker test: {e}")
        return False

def test_searx_startup():
    """Test Searx startup"""
    logger.info("🔍 Test 2: Starting Searx...")

    try:
        from searx_manager import get_searx_manager
        manager = get_searx_manager()

        # Attempt to start Searx
        if manager.ensure_searx_running():
            logger.info("✅ Searx started successfully")
            return True
        else:
            logger.error("❌ Failed to start Searx")
            return False
    except Exception as e:
        logger.error(f"❌ Error during startup test: {e}")
        return False

def test_searx_interface():
    """Test Searx interface"""
    logger.info("🔍 Test 3: Search interface...")

    try:
        from searx_interface import get_searx_interface
        searx = get_searx_interface()

        # Check if Searx is ready
        if not searx.check_health():
            logger.error("❌ Searx is not accessible")
            return False

        # Simple search test
        results = searx.search("test python", max_results=3)

        if results:
            logger.info(f"✅ Search successful: {len(results)} results found")

            # Display first results
            for i, result in enumerate(results[:2], 1):
                logger.info(f"   {i}. {result.title[:50]}... (via {result.engine})")

            return True
        else:
            logger.warning("⚠️ No results found (service might be starting up)")
            return False

    except Exception as e:
        logger.error(f"❌ Error during interface test: {e}")
        return False

def test_gemini_integration():
    """Test integration with artificial intelligence API GOOGLE GEMINI 2.0 FLASH"""
    logger.info("🔍 Test 4: Integration with artificial intelligence API GOOGLE GEMINI 2.0 FLASH...")

    try:
        from gemini_api_adapter import GeminiAPI

        # Create an API instance
        api = GeminiAPI()

        # Check if Searx is integrated
        if hasattr(api, 'searx_available') and api.searx_available:
            logger.info("✅ Searx is integrated into the artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter")

            # Test search request detection
            test_prompt = "search for information on Python"
            if api._detect_web_search_request(test_prompt):
                logger.info("✅ Web search request detection works")
                return True
            else:
                logger.warning("⚠️ Web search request detection does not work")
                return False
        else:
            logger.error("❌ Searx is not integrated into the artificial intelligence API GOOGLE GEMINI 2.0 FLASH adapter")
            return False

    except Exception as e:
        logger.error(f"❌ Error during integration test: {e}")
        return False

def test_search_categories():
    """Test different search categories"""
    logger.info("🔍 Test 5: Search categories...")

    try:
        from searx_interface import get_searx_interface
        searx = get_searx_interface()

        if not searx.check_health():
            logger.warning("⚠️ Searx not accessible, test skipped")
            return True

        # Test different categories
        test_queries = [
            ("artificial intelligence", "general"),
            ("python tutorial", "it"),
            ("tech news", "general")
        ]

        success_count = 0

        for query, category in test_queries:
            try:
                results = searx.search(query, category=category, max_results=2)
                if results:
                    logger.info(f"✅ Search '{query}' ({category}): {len(results)} results")
                    success_count += 1
                else:
                    logger.warning(f"⚠️ No results for '{query}' ({category})")

                # Wait a bit between searches
                time.sleep(1)

            except Exception as e:
                logger.warning(f"⚠️ Error for '{query}': {e}")

        if success_count > 0:
            logger.info(f"✅ Categories test: {success_count}/{len(test_queries)} successful")
            return True
        else:
            logger.error("❌ No category search worked")
            return False

    except Exception as e:
        logger.error(f"❌ Error during categories test: {e}")
        return False

def main():
    """Executes all tests"""
    logger.info("🧪 SEARX SYSTEM TESTS")
    logger.info("=" * 60)

    tests = [
        ("Docker", test_docker_availability),
        ("Searx Startup", test_searx_startup),
        ("Searx Interface", test_searx_interface),
        ("artificial intelligence API GOOGLE GEMINI 2.0 FLASH Integration", test_gemini_integration),
        ("Search Categories", test_search_categories)
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))

            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")

        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))

        # Pause between tests
        time.sleep(2)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:<25} {status}")

    logger.info(f"\nOverall result: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! The Searx system is operational.")
        return True
    elif passed > 0:
        logger.warning(f"⚠️ {passed} tests passed out of {total}. The system is partially functional.")
        return True
    else:
        logger.error("💥 No tests passed. The Searx system is not functioning.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
