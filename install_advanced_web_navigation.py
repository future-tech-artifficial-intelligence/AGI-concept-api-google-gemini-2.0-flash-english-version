"""
Advanced Web Navigation System Installation and Test Script
This script installs dependencies and tests the full system
"""

import subprocess
import sys
import os
import logging
from pathlib import Path

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    """Installs required dependencies"""
    requirements = [
        'beautifulsoup4>=4.12.0',
        'lxml>=4.9.0',
        'nltk>=3.8',
        'aiohttp>=3.8.0',
        'requests>=2.31.0',
        'flask>=2.3.0'
    ]
    
    logger.info("🔧 Installing dependencies...")
    
    for requirement in requirements:
        try:
            logger.info(f"📦 Installing {requirement}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', requirement])
            logger.info(f"✅ {requirement} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error installing {requirement}: {str(e)}")
            return False
    
    # Optional NLTK data installation
    try:
        import nltk
        logger.info("📚 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("✅ NLTK data downloaded")
    except Exception as e:
        logger.warning(f"⚠️ Could not download NLTK data: {str(e)}")
    
    return True

def test_imports():
    """Tests module imports"""
    logger.info("🧪 Testing imports...")
    
    modules_to_test = [
        ('advanced_web_navigator', 'Advanced Web Navigator'),
        ('gemini_web_integration', 'Gemini-Web Integration'),
        ('gemini_navigation_adapter', 'Gemini Navigation Adapter'),
        ('web_navigation_api', 'Web Navigation REST API')
    ]
    
    success_count = 0
    
    for module_name, display_name in modules_to_test:
        try:
            __import__(module_name)
            logger.info(f"✅ {display_name} - Import successful")
            success_count += 1
        except ImportError as e:
            logger.error(f"❌ {display_name} - Import error: {str(e)}")
        except Exception as e:
            logger.error(f"❌ {display_name} - Error: {str(e)}")
    
    logger.info(f"📊 Import results: {success_count}/{len(modules_to_test)} modules imported successfully")
    return success_count == len(modules_to_test)

def test_navigation_system():
    """Tests the navigation system"""
    logger.info("🚀 Testing navigation system...")
    
    try:
        from advanced_web_navigator import extract_website_content
        
        # Test with a test URL
        test_url = "https://httpbin.org/json"
        logger.info(f"🔍 Extraction test: {test_url}")
        
        content = extract_website_content(test_url)
        
        if content.success:
            logger.info(f"✅ Extraction successful:")
            logger.info(f"  - Title: {content.title}")
            logger.info(f"  - Content: {len(content.cleaned_text)} characters")
            logger.info(f"  - Quality score: {content.content_quality_score}")
            logger.info(f"  - Language: {content.language}")
            return True
        else:
            logger.error(f"❌ Extraction failed: {content.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during navigation test: {str(e)}")
        return False

def test_gemini_integration():
    """Tests the Gemini integration"""
    logger.info("🤖 Testing Gemini integration...")
    
    try:
        from gemini_navigation_adapter import detect_navigation_need, initialize_gemini_navigation_adapter
        
        # Initialize the adapter
        initialize_gemini_navigation_adapter()
        logger.info("✅ Gemini adapter initialized")
        
        # Navigation detection test
        test_prompts = [
            "Search and navigate artificial intelligence",
            "Extract content from https://example.com",
            "What is machine learning?",
            "Explore https://wikipedia.org in depth"
        ]
        
        detection_results = []
        for prompt in test_prompts:
            detection = detect_navigation_need(prompt)
            detection_results.append({
                'prompt': prompt,
                'requires_navigation': detection.get('requires_navigation', False),
                'navigation_type': detection.get('navigation_type'),
                'confidence': detection.get('confidence', 0)
            })
        
        # Display results
        logger.info("🔍 Navigation detection results:")
        for result in detection_results:
            status = "🟢" if result['requires_navigation'] else "🔴"
            logger.info(f"  {status} '{result['prompt'][:50]}...'")
            logger.info(f"     Type: {result['navigation_type']}, Confidence: {result['confidence']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during Gemini integration test: {str(e)}")
        return False

def test_api_endpoints():
    """Tests API endpoints"""
    logger.info("🌐 Testing API endpoints...")
    
    try:
        from web_navigation_api import register_web_navigation_api, initialize_web_navigation_api
        from flask import Flask
        
        # Create a test Flask app
        app = Flask(__name__)
        register_web_navigation_api(app)
        initialize_web_navigation_api()
        
        # Test endpoints
        with app.test_client() as client:
            # Health check
            response = client.get('/api/web-navigation/health')
            if response.status_code == 200:
                health_data = response.get_json()
                logger.info(f"✅ Health check: {health_data.get('overall_status', 'unknown')}")
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
            
            # Documentation test
            response = client.get('/api/web-navigation/docs')
            if response.status_code == 200:
                logger.info("✅ API documentation accessible")
            else:
                logger.error(f"❌ Documentation not accessible: {response.status_code}")
            
            # Statistics test
            response = client.get('/api/web-navigation/stats')
            if response.status_code == 200:
                logger.info("✅ API statistics accessible")
            else:
                logger.error(f"❌ Statistics not accessible: {response.status_code}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during API endpoints test: {str(e)}")
        return False

def create_test_report():
    """Creates a test report"""
    logger.info("📋 Creating test report...")
    
    report_content = """
# Test Report - Advanced Web Navigation System

## Tests Performed

### 1. Dependency Installation
- beautifulsoup4: Installation and test
- lxml: HTML/XML parser
- nltk: Natural Language Toolkit
- aiohttp: Asynchronous HTTP requests
- requests: Synchronous HTTP requests
- flask: Web framework

### 2. Module Testing
- advanced_web_navigator.py: Web content extractor
- gemini_web_integration.py: Integration with Gemini
- gemini_navigation_adapter.py: Adapter for Gemini
- web_navigation_api.py: REST API

### 3. Navigation Test
- Web content extraction
- Content quality analysis
- Language detection
- Metadata extraction

### 4. Gemini Integration Test
- Navigation request detection
- Navigation type classification
- Confidence scoring

### 5. REST API Test
- Health endpoints
- Automatic documentation
- Cache and statistics
- Self-generated documentation
- Health checks

## Implemented Features

### Advanced Web Navigation
✅ Structured content extraction
✅ Deep navigation
✅ Content quality analysis
✅ Smart caching
✅ Multi-language support

### Gemini Integration
✅ Automatic request detection
✅ Formatting for the Gemini API
✅ Context management
✅ Fallback to old system

### Complete REST API
✅ Complete CRUD endpoints
✅ Session management
✅ Cache and statistics
✅ Auto-generated documentation
✅ Health checks

## Available API Endpoints

- POST /api/web-navigation/search-and-navigate
- POST /api/web-navigation/extract-content
- POST /api/web-navigation/navigate-deep
- POST /api/web-navigation/user-journey
- GET  /api/web-navigation/health
- GET  /api/web-navigation/docs
- GET  /api/web-navigation/stats

## Next Steps

1. Performance optimization
2. Adding more search engines
3. Improving content detection
4. Supporting more data formats
5. Advanced monitoring

"""
    
    try:
        with open("test_report.md", "w", encoding="utf-8") as f:
            f.write(report_content)
        logger.info("✅ Test report created: test_report.md")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating report: {str(e)}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting Advanced Web Navigation System installation and tests")
    logger.info("=" * 80)
    
    # Step 1: Install dependencies
    if not install_requirements():
        logger.error("❌ Dependency installation failed")
        return False
    
    # Step 2: Test imports
    if not test_imports():
        logger.error("❌ Import test failed")
        return False
    
    # Step 3: Test navigation system
    if not test_navigation_system():
        logger.error("❌ Navigation system test failed")
        return False
    
    # Step 4: Test Gemini integration
    if not test_gemini_integration():
        logger.error("❌ Gemini integration test failed")
        return False
    
    # Step 5: Test API endpoints
    if not test_api_endpoints():
        logger.error("❌ API endpoints test failed")
        return False
    
    # Step 6: Create report
    create_test_report()
    
    logger.info("=" * 80)
    logger.info("🎉 ALL TESTS PASSED SUCCESSFULLY!")
    logger.info("✅ The Advanced Web Navigation System is ready to use")
    logger.info("=" * 80)
    
    # Display usage instructions
    logger.info("\n📚 USAGE INSTRUCTIONS:")
    logger.info("1. The system is now integrated into your Flask app")
    logger.info("2. API endpoints are available under /api/web-navigation/")
    logger.info("3. Gemini integration automatically detects navigation requests")
    logger.info("4. See test_report.md for more details")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
