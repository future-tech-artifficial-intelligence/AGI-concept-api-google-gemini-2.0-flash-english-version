"""
**Advanced Web Navigation System Test and Demonstration Script**
This script showcases the system's capabilities with concrete examples
"""

import logging
import json
import time
from datetime import datetime

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_header(title):
    """Displays a demonstration header"""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def demo_content_extraction():
    """Content extraction demonstration"""
    demo_header("WEB CONTENT EXTRACTION")
    
    try:
        from advanced_web_navigator import extract_website_content
        
        test_urls = [
            "https://httpbin.org/json",
            "https://fr.wikipedia.org/wiki/Intelligence_artificielle",
            "https://www.python.org"
        ]
        
        for url in test_urls:
            print(f"\n🔍 Extracting from: {url}")
            
            start_time = time.time()
            content = extract_website_content(url)
            extraction_time = time.time() - start_time
            
            if content.success:
                print(f"✅ Extraction successful in {extraction_time:.2f}s")
                print(f"  📄 Title: {content.title}")
                print(f"  📊 Content: {len(content.cleaned_text)} characters")
                print(f"  🏆 Quality score: {content.content_quality_score}/10")
                print(f"  🌐 Language: {content.language}")
                print(f"  🔗 Links found: {len(content.links)}")
                print(f"  🖼️ Images: {len(content.images)}")
                print(f"  🔑 Keywords: {', '.join(content.keywords[:5])}")
                
                if content.summary:
                    print(f"  📝 Summary: {content.summary[:150]}...")
            else:
                print(f"❌ Extraction failed: {content.error_message}")
            
            time.sleep(1)  # Delay between requests
            
    except Exception as e:
        logger.error(f"Error during extraction demo: {str(e)}")

def demo_navigation_detection():
    """Navigation detection demonstration"""
    demo_header("AUTOMATIC NAVIGATION DETECTION")
    
    try:
        from gemini_navigation_adapter import detect_navigation_need, initialize_gemini_navigation_adapter
        
        # Initialize the adapter
        initialize_gemini_navigation_adapter()
        
        test_prompts = [
            "Search and navigate artificial intelligence",
            "Extract content from https://example.com/article",
            "Deeply explore https://wikipedia.org with 3 levels",
            "Simulate a purchase journey on https://shop.example.com",
            "What is machine learning?",
            "Can you help me with Python?",
            "Navigate the Python documentation site",
            "How to do a Google search?",
            "Analyze this webpage completely",
            "Find information about electric cars"
        ]
        
        print("🧪 Testing navigation detection on different prompts:\n")
        
        for i, prompt in enumerate(test_prompts, 1):
            detection = detect_navigation_need(prompt)
            
            requires_nav = detection.get('requires_navigation', False)
            nav_type = detection.get('navigation_type', 'none')
            confidence = detection.get('confidence', 0)
            params = detection.get('extracted_params', {})
            
            status_icon = "🟢" if requires_nav else "🔴"
            confidence_bar = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            
            print(f"{i:2}. {status_icon} '{prompt}'")
            print(f"     Type: {nav_type} | Confidence: {confidence:.1f} [{confidence_bar}]")
            
            if params:
                print(f"     Parameters: {params}")
            print()
            
    except Exception as e:
        logger.error(f"Error during detection demo: {str(e)}")

def demo_deep_navigation():
    """Deep navigation demonstration"""
    demo_header("DEEP WEBSITE NAVIGATION")
    
    try:
        from advanced_web_navigator import navigate_website_deep
        
        # Test with a simple site
        test_url = "https://httpbin.org"
        
        print(f"🚀 Deep navigation from: {test_url}")
        print("   Parameters: max_depth=2, max_pages=5")
        
        start_time = time.time()
        nav_path = navigate_website_deep(test_url, max_depth=2, max_pages=5)
        navigation_time = time.time() - start_time
        
        print(f"\n✅ Navigation finished in {navigation_time:.2f}s")
        print(f"📊 Navigation statistics:")
        print(f"  - Pages visited: {len(nav_path.visited_pages)}")
        print(f"  - Depth reached: {nav_path.navigation_depth}")
        print(f"  - Total content extracted: {nav_path.total_content_extracted} characters")
        print(f"  - Strategy: {nav_path.navigation_strategy}")
        print(f"  - Session ID: {nav_path.session_id}")
        
        if nav_path.visited_pages:
            print(f"\n📄 Explored pages:")
            for i, page in enumerate(nav_path.visited_pages, 1):
                print(f"  {i}. {page.title} (Score: {page.content_quality_score:.1f})")
                print(f"     URL: {page.url}")
                print(f"     Content: {len(page.cleaned_text)} characters")
                if page.keywords:
                    print(f"     Keywords: {', '.join(page.keywords[:3])}")
                print()
                
    except Exception as e:
        logger.error(f"Error during navigation demo: {str(e)}")

def demo_gemini_integration():
    """Gemini integration demonstration"""
    demo_header("FULL GEMINI INTEGRATION")
    
    try:
        from gemini_web_integration import search_web_for_gemini, initialize_gemini_web_integration
        
        # Initialize integration
        initialize_gemini_web_integration()
        
        # Simple search test
        query = "artificial intelligence 2024"
        user_context = "developer looking for the latest trends"
        
        print(f"🔍 Searching and navigating for artificial intelligence API GOOGLE GEMINI 2.0 FLASH:")
        print(f"   Query: '{query}'")
        print(f"   Context: {user_context}")
        
        start_time = time.time()
        result = search_web_for_gemini(query, user_context)
        processing_time = time.time() - start_time
        
        print(f"\n⏱️ Processing finished in {processing_time:.2f}s")
        
        if result.get('success', False):
            print("✅ Search and navigation successful!")
            
            search_summary = result.get('search_summary', {})
            print(f"\n📊 Search summary:")
            print(f"  - Sites searched: {search_summary.get('sites_searched', 0)}")
            print(f"  - Sites navigated: {search_summary.get('sites_navigated', 0)}")
            print(f"  - Total pages visited: {search_summary.get('total_pages_visited', 0)}")
            print(f"  - High quality pages: {search_summary.get('high_quality_pages', 0)}")
            
            if 'content_synthesis' in result:
                print(f"\n📝 Content synthesis:")
                synthesis = result['content_synthesis']
                print(f"   {synthesis[:300]}{'...' if len(synthesis) > 300 else ''}")
            
            if 'aggregated_keywords' in result and result['aggregated_keywords']:
                keywords = ', '.join(result['aggregated_keywords'][:10])
                print(f"\n🔑 Identified keywords: {keywords}")
            
            if 'navigation_insights' in result:
                print(f"\n💡 Navigation insights:")
                for insight in result['navigation_insights'][:3]:
                    print(f"   • {insight}")
            
            if 'recommended_actions' in result:
                print(f"\n💭 Recommendations:")
                for action in result['recommended_actions'][:3]:
                    print(f"   • {action}")
                    
        else:
            print(f"❌ Search failed: {result.get('reason', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error during Gemini integration demo: {str(e)}")

def demo_api_endpoints():
    """API endpoints demonstration"""
    demo_header("REST API - ENDPOINTS")
    
    try:
        from web_navigation_api import register_web_navigation_api, initialize_web_navigation_api
        from flask import Flask
        
        # Create a test Flask app
        app = Flask(__name__)
        register_web_navigation_api(app)
        initialize_web_navigation_api()
        
        print("🌐 Testing REST API endpoints:\n")
        
        with app.test_client() as client:
            # Test 1: Health check
            print("1. 🏥 Health Check")
            response = client.get('/api/web-navigation/health')
            if response.status_code == 200:
                health_data = response.get_json()
                status = health_data.get('overall_status', 'unknown')
                print(f"   Status: {status}")
                
                components = health_data.get('components', {})
                for component, comp_status in components.items():
                    if component != 'timestamp':
                        icon = "✅" if comp_status == 'healthy' else "⚠️"
                        print(f"   {icon} {component}: {comp_status}")
            else:
                print(f"   ❌ Error: {response.status_code}")
            
            # Test 2: Documentation
            print("\n2. 📚 API Documentation")
            response = client.get('/api/web-navigation/docs')
            if response.status_code == 200:
                docs = response.get_json()
                print(f"   ✅ Documentation available")
                print(f"   API: {docs.get('api_name', 'N/A')}")
                print(f"   Version: {docs.get('version', 'N/A')}")
                
                endpoints = docs.get('endpoints', {})
                print(f"   Available endpoints: {len(endpoints)}")
            else:
                print(f"   ❌ Error: {response.status_code}")
            
            # Test 3: Statistics
            print("\n3. 📊 Statistics")
            response = client.get('/api/web-navigation/stats')
            if response.status_code == 200:
                stats_data = response.get_json()
                if stats_data.get('success', False):
                    stats = stats_data.get('stats', {})
                    api_stats = stats.get('api_stats', {})
                    
                    print(f"   ✅ Statistics retrieved")
                    print(f"   Active sessions: {stats.get('active_sessions', 0)}")
                    print(f"   Cache size: {stats.get('cache_size', 0)}")
                    print(f"   Cache hit rate: {stats.get('cache_hit_rate', 0):.1f}%")
                    print(f"   Total searches: {api_stats.get('total_searches', 0)}")
                else:
                    print(f"   ❌ Error in data")
            else:
                print(f"   ❌ Error: {response.status_code}")
            
            # Test 4: Session creation
            print("\n4. 🔐 Session Creation")
            session_data = {
                "user_id": "demo_user",
                "config": {
                    "max_depth": 2,
                    "max_pages": 5
                }
            }
            
            response = client.post('/api/web-navigation/create-session', json=session_data)
            if response.status_code == 200:
                result = response.get_json()
                if result.get('success', False):
                    session_id = result.get('session_id')
                    print(f"   ✅ Session created: {session_id}")
                    print(f"   Configuration: {result.get('config', {})}")
                    
                    # Test 5: Session info
                    print("\n5. ℹ️ Session Information")
                    response = client.get(f'/api/web-navigation/session/{session_id}')
                    if response.status_code == 200:
                        session_info = response.get_json()
                        if session_info.get('success', False):
                            print(f"   ✅ Session found")
                            print(f"   User: {session_info.get('user_id')}")
                            print(f"   Created at: {session_info.get('created_at', 'N/A')[:19]}")
                            print(f"   Requests: {session_info.get('requests_count', 0)}")
                            print(f"   Active: {session_info.get('is_active', False)}")
                        else:
                            print(f"   ❌ Session not found")
                    else:
                        print(f"   ❌ Error: {response.status_code}")
                else:
                    print(f"   ❌ Error: {result.get('error', 'Unknown')}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
    except Exception as e:
        logger.error(f"Error during API demo: {str(e)}")

def demo_performance_test():
    """Simple performance test"""
    demo_header("PERFORMANCE TEST")
    
    try:
        from advanced_web_navigator import extract_website_content
        
        test_urls = [
            "https://httpbin.org/json",
            "https://httpbin.org/html",
            "https://httpbin.org/robots.txt"
        ]
        
        print("⚡ Performance test on multiple URLs:\n")
        
        total_start = time.time()
        results = []
        
        for i, url in enumerate(test_urls, 1):
            print(f"{i}. Test: {url}")
            
            start_time = time.time()
            content = extract_website_content(url)
            end_time = time.time()
            
            processing_time = end_time - start_time
            results.append({
                'url': url,
                'success': content.success,
                'time': processing_time,
                'content_length': len(content.cleaned_text) if content.success else 0,
                'quality_score': content.content_quality_score if content.success else 0
            })
            
            status = "✅" if content.success else "❌"
            print(f"   {status} Time: {processing_time:.2f}s | "
                  f"Content: {len(content.cleaned_text)} chars | "
                  f"Quality: {content.content_quality_score:.1f}")
            
            time.sleep(0.5)  # Small delay between requests
        
        total_time = time.time() - total_start
        
        print(f"\n📊 Performance summary:")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per URL: {total_time/len(test_urls):.2f}s")
        
        successful_results = [r for r in results if r['success']]
        if successful_results:
            avg_content = sum(r['content_length'] for r in successful_results) / len(successful_results)
            avg_quality = sum(r['quality_score'] for r in successful_results) / len(successful_results)
            print(f"   Average content: {avg_content:.0f} characters")
            print(f"   Average quality: {avg_quality:.1f}/10")
        
        success_rate = (len(successful_results) / len(results)) * 100
        print(f"   Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"Error during performance test: {str(e)}")

def main():
    """Main demonstration function"""
    print("🌟 ADVANCED WEB NAVIGATION SYSTEM DEMONSTRATION")
    print("=" * 70)
    print(f"⏰ Started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    demos = [
        ("Content Extraction", demo_content_extraction),
        ("Navigation Detection", demo_navigation_detection),
        ("Deep Navigation", demo_deep_navigation),
        ("Gemini Integration", demo_gemini_integration),
        ("API Endpoints", demo_api_endpoints),
        ("Performance Test", demo_performance_test)
    ]
    
    print("\n🎯 Available Demonstrations:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 70)
    
    try:
        choice = input("\n🔢 Choose a demo (1-6) or 'all' for all: ").strip().lower()
        
        if choice == 'all':
            for name, demo_func in demos:
                print(f"\n🚀 Starting: {name}")
                demo_func()
                print(f"✅ Finished: {name}")
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            demo_index = int(choice) - 1
            name, demo_func = demos[demo_index]
            print(f"\n🚀 Starting: {name}")
            demo_func()
            print(f"✅ Finished: {name}")
        else:
            print("❌ Invalid choice. Running all demos...")
            for name, demo_func in demos:
                demo_func()
    
    except KeyboardInterrupt:
        print("\n\n⏹️ Demonstration interrupted by user")
    
    except Exception as e:
        logger.error(f"Error during demonstration: {str(e)}")
    
    finally:
        print("\n" + "=" * 70)
        print("🎉 DEMONSTRATION COMPLETE")
        print("📚 Consult ADVANCED_WEB_NAVIGATION_DOCUMENTATION.md for more info")
        print("=" * 70)

if __name__ == "__main__":
    main()
