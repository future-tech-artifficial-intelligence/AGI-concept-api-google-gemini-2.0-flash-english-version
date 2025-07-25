#!/usr/bin/env python3
"""
Test script for the Searx visual capture system
"""

import logging
import time
import sys
import os

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('searx_visual_test')

def test_dependencies():
    """Tests necessary dependencies"""
    logger.info("🔍 Test 1: Checking dependencies...")
    
    missing_deps = []
    
    try:
        import selenium
        logger.info("✅ Selenium available")
    except ImportError:
        missing_deps.append("selenium")
    
    try:
        from PIL import Image
        logger.info("✅ Pillow available")
    except ImportError:
        missing_deps.append("Pillow")
    
    try:
        import requests
        logger.info("✅ Requests available")
    except ImportError:
        missing_deps.append("requests")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        return False
    
    logger.info("✅ All dependencies are available")
    return True

def test_webdriver_initialization():
    """Tests WebDriver initialization"""
    logger.info("🔍 Test 2: Initializing WebDriver...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        
        if capture._initialize_webdriver():
            logger.info("✅ WebDriver initialized successfully")
            capture.close()
            return True
        else:
            logger.error("❌ WebDriver initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during WebDriver test: {e}")
        return False

def test_searx_accessibility():
    """Tests Searx accessibility"""
    logger.info("🔍 Test 3: Searx accessibility...")
    
    try:
        import requests
        
        response = requests.get("http://localhost:8080/", timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Searx is accessible")
            return True
        else:
            logger.warning(f"⚠️ Searx responds with code: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Searx unreachable: {e}")
        logger.error("Make sure Searx is started with: start_with_searx.bat")
        return False

def test_visual_capture():
    """Tests visual capture"""
    logger.info("🔍 Test 4: Visual capture...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        
        # Simple capture test
        result = capture.capture_search_results("test python", category="general")
        
        if result and result.get('success'):
            logger.info(f"✅ Capture successful: {result['screenshot_path']}")
            
            # Check if the file exists
            if os.path.exists(result['screenshot_path']):
                logger.info("✅ Capture file created")
                
                # Check file size
                file_size = os.path.getsize(result['screenshot_path'])
                logger.info(f"📄 File size: {file_size} bytes")
                
                if file_size > 1000:  # At least 1KB
                    logger.info("✅ Valid capture file")
                    capture.close()
                    return True
                else:
                    logger.warning("⚠️ Capture file too small")
            else:
                logger.error("❌ Capture file not created")
        else:
            logger.error(f"❌ Capture failed: {result.get('error', 'Unknown error')}")
        
        capture.close()
        return False
        
    except Exception as e:
        logger.error(f"❌ Error during capture test: {e}")
        return False

def test_visual_annotation():
    """Tests visual annotations"""
    logger.info("🔍 Test 5: Visual annotations...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        
        # Test capture with annotations
        result = capture.capture_with_annotations("artificial intelligence", category="general")
        
        if result and result.get('success') and result.get('has_annotations'):
            logger.info("✅ Capture with annotations successful")
            
            if result.get('annotated_image'):
                logger.info("✅ Annotated image generated (base64)")
            
            capture.close()
            return True
        else:
            logger.error("❌ Capture with annotations failed")
            capture.close()
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during annotation test: {e}")
        return False

def test_integration_with_searx_interface():
    """Tests integration with SearxInterface"""
    logger.info("🔍 Test 6: Integration with SearxInterface...")
    
    try:
        from searx_interface import get_searx_interface
        
        searx = get_searx_interface()
        
        # Test search with visual
        if hasattr(searx, 'search_with_visual'):
            result = searx.search_with_visual("test search", category="general", max_results=3)
            
            if result.get('has_visual'):
                logger.info("✅ Search with visual capture successful")
                
                # Test summary
                summary = searx.get_visual_search_summary(result)
                if summary and len(summary) > 100:
                    logger.info("✅ Visual summary generated")
                    return True
                else:
                    logger.warning("⚠️ Visual summary too short")
            else:
                logger.warning("⚠️ No visual data in result")
        else:
            logger.error("❌ search_with_visual method not available")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during integration test: {e}")
        return False

def test_cleanup():
    """Tests cleanup"""
    logger.info("🔍 Test 7: Cleanup...")
    
    try:
        from searx_visual_capture import SearxVisualCapture
        
        capture = SearxVisualCapture()
        capture.cleanup_old_screenshots(max_age_hours=0)  # Clean up everything
        
        # Check directory
        screenshots_dir = capture.screenshots_dir
        if os.path.exists(screenshots_dir):
            remaining_files = len([f for f in os.listdir(screenshots_dir) 
                                 if f.endswith('.png')])
            logger.info(f"📁 Remaining files: {remaining_files}")
        
        logger.info("✅ Cleanup complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")
        return False

def main():
    """Executes all tests"""
    logger.info("🧪 SEARX VISUAL CAPTURE SYSTEM TESTS")
    logger.info("=" * 70)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("WebDriver", test_webdriver_initialization),
        ("Searx Accessibility", test_searx_accessibility),
        ("Visual Capture", test_visual_capture),
        ("Annotations", test_visual_annotation),
        ("Integration", test_integration_with_searx_interface),
        ("Cleanup", test_cleanup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✅ {test_name}: SUCCESS")
            else:
                logger.error(f"❌ {test_name}: FAIL")
                
        except Exception as e:
            logger.error(f"💥 {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        # Pause between tests
        time.sleep(1)
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAIL"
        logger.info(f"{test_name:<25} {status}")
    
    logger.info(f"\nOverall result: {passed}/{total} tests successful")
    
    if passed == total:
        logger.info("🎉 All tests passed! The visual capture system is operational.")
        logger.info("\n💡 The AI can now see search results like a human!")
        return True
    elif passed > total // 2:
        logger.warning(f"⚠️ {passed} tests successful out of {total}. The system is partially functional.")
        return True
    else:
        logger.error("💥 Critical failure. The visual capture system is not working.")
        logger.error("\n🔧 Possible solutions:")
        logger.error("1. Install dependencies: python install_searx_visual_deps.py")
        logger.error("2. Start Searx: start_with_searx.bat")
        logger.error("3. Verify that Chrome or Edge is installed")
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
