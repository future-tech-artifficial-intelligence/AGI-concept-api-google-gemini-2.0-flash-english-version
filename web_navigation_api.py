"""
REST API for the Advanced Web Navigation System
This API exposes all web navigation functionalities for the artificial intelligence API GOOGLE GEMINI 2.0 FLASH
"""

from flask import Flask, request, jsonify, Blueprint
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import asyncio
from pathlib import Path

from gemini_web_integration import (
    initialize_gemini_web_integration,
    search_web_for_gemini,
    extract_content_for_gemini,
    simulate_user_journey,
    gemini_web_integration
)
from advanced_web_navigator import (
    navigate_website_deep,
    extract_website_content,
    advanced_navigator
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WebNavigationAPI')

# Blueprint for the API
web_nav_bp = Blueprint('web_navigation', __name__, url_prefix='/api/web-navigation')

class WebNavigationAPIManager:
    """Web navigation API manager"""
    
    def __init__(self):
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 hour
        self.max_concurrent_sessions = 10
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'total_pages_extracted': 0,
            'total_content_characters': 0,
            'successful_navigations': 0,
            'failed_navigations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Recent results cache
        self.result_cache = {}
        self.cache_max_size = 100
        
        logger.info("✅ Web Navigation API Manager initialized")
    
    def create_session(self, user_id: str, session_config: Dict[str, Any] = None) -> str:
        """Creates a new navigation session"""
        session_id = f"nav_session_{user_id}_{int(time.time())}"
        
        # Default configuration
        default_config = {
            'max_depth': 3,
            'max_pages': 10,
            'quality_threshold': 3.0,
            'timeout': 30,
            'enable_cache': True
        }
        
        if session_config:
            default_config.update(session_config)
        
        self.active_sessions[session_id] = {
            'user_id': user_id,
            'created_at': datetime.now(),
            'config': default_config,
            'requests_count': 0,
            'last_activity': datetime.now()
        }
        
        # Clean up old sessions
        self._cleanup_old_sessions()
        
        logger.info(f"🆕 Session created: {session_id} for user {user_id}")
        return session_id
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session information"""
        return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """Updates session activity"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_activity'] = datetime.now()
            self.active_sessions[session_id]['requests_count'] += 1
    
    def _cleanup_old_sessions(self):
        """Cleans up expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            if (current_time - session_data['last_activity']).seconds > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️ Expired session deleted: {session_id}")
    
    def get_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generates a cache key"""
        cache_data = {
            'query': query,
            'params': sorted(params.items())
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieves a result from cache"""
        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            # Cache valid for 30 minutes
            if (datetime.now() - timestamp).seconds < 1800:
                self.stats['cache_hits'] += 1
                return result
            else:
                del self.result_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        return None
    
    def save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Saves a result to cache"""
        if len(self.result_cache) >= self.cache_max_size:
            # Remove the oldest
            oldest_key = min(self.result_cache.keys(), 
                           key=lambda k: self.result_cache[k][1])
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = (result, datetime.now())
    
    def update_stats(self, operation: str, **kwargs):
        """Updates statistics"""
        if operation == 'search':
            self.stats['total_searches'] += 1
            if kwargs.get('success', False):
                self.stats['successful_navigations'] += 1
                self.stats['total_pages_extracted'] += kwargs.get('pages_extracted', 0)
                self.stats['total_content_characters'] += kwargs.get('content_characters', 0)
            else:
                self.stats['failed_navigations'] += 1

# Global manager instance
api_manager = WebNavigationAPIManager()

@web_nav_bp.route('/create-session', methods=['POST'])
def create_navigation_session():
    """Creates a new navigation session"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'anonymous')
        session_config = data.get('config', {})
        
        session_id = api_manager.create_session(user_id, session_config)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'config': api_manager.get_session_info(session_id)['config'],
            'message': 'Navigation session created successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ Session creation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/search-and-navigate', methods=['POST'])
def search_and_navigate():
    """Searches and navigates websites"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'JSON data required'}), 400
        
        query = data.get('query')
        if not query:
            return jsonify({'success': False, 'error': '"query" parameter required'}), 400
        
        session_id = data.get('session_id')
        user_context = data.get('user_context', '')
        use_cache = data.get('use_cache', True)
        
        # Check session if provided
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Invalid session'}), 400
            api_manager.update_session_activity(session_id)
        
        # Check cache
        cache_key = api_manager.get_cache_key(query, {
            'user_context': user_context,
            'operation': 'search_and_navigate'
        })
        
        if use_cache:
            cached_result = api_manager.get_from_cache(cache_key)
            if cached_result:
                logger.info(f"📋 Result retrieved from cache for: {query}")
                return jsonify(cached_result)
        
        # Initialize integration if necessary
        if not gemini_web_integration:
            initialize_gemini_web_integration()
        
        # Perform search and navigation
        logger.info(f"🔍 Starting search and navigation: {query}")
        start_time = time.time()
        
        result = search_web_for_gemini(query, user_context)
        
        processing_time = time.time() - start_time
        
        # Enrich the result with API metadata
        api_result = {
            'api_version': '1.0',
            'processing_time': round(processing_time, 2),
            'session_id': session_id,
            'cache_used': False,
            'timestamp': datetime.now().isoformat(),
            **result
        }
        
        # Update statistics
        api_manager.update_stats(
            'search',
            success=result.get('success', False),
            pages_extracted=result.get('search_summary', {}).get('total_pages_visited', 0),
            content_characters=len(str(result.get('content_synthesis', '')))
        )
        
        # Save to cache
        if use_cache and result.get('success', False):
            api_manager.save_to_cache(cache_key, api_result)
        
        logger.info(f"✅ Search completed in {processing_time:.2f}s")
        return jsonify(api_result)
        
    except Exception as e:
        logger.error(f"❌ Search and navigation error: {str(e)}")
        api_manager.update_stats('search', success=False)
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/extract-content', methods=['POST'])
def extract_specific_content():
    """Extracts specific content from a URL"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'JSON data required'}), 400
        
        url = data.get('url')
        if not url:
            return jsonify({'success': False, 'error': '"url" parameter required'}), 400
        
        requirements = data.get('requirements', ['summary', 'details', 'links'])
        session_id = data.get('session_id')
        use_cache = data.get('use_cache', True)
        
        # Check session
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Invalid session'}), 400
            api_manager.update_session_activity(session_id)
        
        # Check cache
        cache_key = api_manager.get_cache_key(url, {
            'requirements': requirements,
            'operation': 'extract_content'
        })
        
        if use_cache:
            cached_result = api_manager.get_from_cache(cache_key)
            if cached_result:
                logger.info(f"📋 Extraction retrieved from cache for: {url}")
                return jsonify(cached_result)
        
        # Extract content
        logger.info(f"🎯 Content extraction: {url}")
        start_time = time.time()
        
        result = extract_content_for_gemini(url, requirements)
        
        processing_time = time.time() - start_time
        
        # Enrich the result
        api_result = {
            'api_version': '1.0',
            'processing_time': round(processing_time, 2),
            'session_id': session_id,
            'cache_used': False,
            'requirements_requested': requirements,
            'timestamp': datetime.now().isoformat(),
            **result
        }
        
        # Save to cache
        if use_cache and result.get('success', False):
            api_manager.save_to_cache(cache_key, api_result)
        
        logger.info(f"✅ Extraction completed in {processing_time:.2f}s")
        return jsonify(api_result)
        
    except Exception as e:
        logger.error(f"❌ Content extraction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/navigate-deep', methods=['POST'])
def navigate_website_deep_api():
    """Deep navigation into a website"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'JSON data required'}), 400
        
        start_url = data.get('start_url')
        if not start_url:
            return jsonify({'success': False, 'error': '"start_url" parameter required'}), 400
        
        max_depth = data.get('max_depth', 3)
        max_pages = data.get('max_pages', 10)
        session_id = data.get('session_id')
        
        # Validate parameters
        if max_depth > 5:
            return jsonify({'success': False, 'error': 'max_depth cannot exceed 5'}), 400
        if max_pages > 50:
            return jsonify({'success': False, 'error': 'max_pages cannot exceed 50'}), 400
        
        # Check session
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Invalid session'}), 400
            api_manager.update_session_activity(session_id)
        
        # Deep navigation
        logger.info(f"🚀 Deep navigation: {start_url} (depth: {max_depth}, pages: {max_pages})")
        start_time = time.time()
        
        nav_path = navigate_website_deep(start_url, max_depth, max_pages)
        
        processing_time = time.time() - start_time
        
        # Prepare the response
        result = {
            'success': True,
            'api_version': '1.0',
            'processing_time': round(processing_time, 2),
            'session_id': session_id,
            'navigation_summary': {
                'start_url': nav_path.start_url,
                'pages_visited': len(nav_path.visited_pages),
                'navigation_depth': nav_path.navigation_depth,
                'total_content_extracted': nav_path.total_content_extracted,
                'navigation_strategy': nav_path.navigation_strategy,
                'session_id': nav_path.session_id
            },
            'visited_pages': [
                {
                    'url': page.url,
                    'title': page.title,
                    'summary': page.summary,
                    'content_quality_score': page.content_quality_score,
                    'keywords': page.keywords[:10],  # Top 10 keywords
                    'language': page.language,
                    'links_count': len(page.links),
                    'images_count': len(page.images)
                }
                for page in nav_path.visited_pages
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Deep navigation completed in {processing_time:.2f}s - {len(nav_path.visited_pages)} pages")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Deep navigation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/user-journey', methods=['POST'])
def simulate_user_journey_api():
    """Simulates a user journey on a website"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'JSON data required'}), 400
        
        start_url = data.get('start_url')
        user_intent = data.get('user_intent', 'explore')
        session_id = data.get('session_id')
        
        if not start_url:
            return jsonify({'success': False, 'error': '"start_url" parameter required'}), 400
        
        # Validate intent
        valid_intents = ['buy', 'learn', 'contact', 'explore']
        if user_intent not in valid_intents:
            return jsonify({
                'success': False, 
                'error': f'user_intent must be one of: {valid_intents}'
            }), 400
        
        # Check session
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Invalid session'}), 400
            api_manager.update_session_activity(session_id)
        
        # Simulate user journey
        logger.info(f"👤 User journey simulation: {user_intent} from {start_url}")
        start_time = time.time()
        
        result = simulate_user_journey(start_url, user_intent)
        
        processing_time = time.time() - start_time
        
        # Enrich the result
        api_result = {
            'api_version': '1.0',
            'processing_time': round(processing_time, 2),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            **result
        }
        
        logger.info(f"✅ User journey completed in {processing_time:.2f}s")
        return jsonify(api_result)
        
    except Exception as e:
        logger.error(f"❌ User journey error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/session/<session_id>', methods=['GET'])
def get_session_info_api(session_id: str):
    """Retrieves session information"""
    try:
        session_info = api_manager.get_session_info(session_id)
        
        if not session_info:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        # Prepare session information
        response_data = {
            'success': True,
            'session_id': session_id,
            'user_id': session_info['user_id'],
            'created_at': session_info['created_at'].isoformat(),
            'last_activity': session_info['last_activity'].isoformat(),
            'requests_count': session_info['requests_count'],
            'config': session_info['config'],
            'is_active': (datetime.now() - session_info['last_activity']).seconds < api_manager.session_timeout
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Session retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/session/<session_id>', methods=['DELETE'])
def delete_session_api(session_id: str):
    """Deletes a session"""
    try:
        if session_id in api_manager.active_sessions:
            del api_manager.active_sessions[session_id]
            logger.info(f"🗑️ Session deleted: {session_id}")
            return jsonify({
                'success': True,
                'message': f'Session {session_id} deleted successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
            
    except Exception as e:
        logger.error(f"❌ Session deletion error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/stats', methods=['GET'])
def get_api_stats():
    """Retrieves API statistics"""
    try:
        stats = {
            'api_stats': api_manager.stats.copy(),
            'active_sessions': len(api_manager.active_sessions),
            'cache_size': len(api_manager.result_cache),
            'cache_hit_rate': (api_manager.stats['cache_hits'] / 
                             max(api_manager.stats['cache_hits'] + api_manager.stats['cache_misses'], 1)) * 100,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        logger.error(f"❌ Statistics retrieval error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    try:
        # Check component status
        health_status = {
            'api': 'healthy',
            'navigator': 'healthy' if advanced_navigator else 'unavailable',
            'integration': 'healthy' if gemini_web_integration else 'unavailable',
            'cache': 'healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        # Simple navigation test
        try:
            test_content = extract_website_content('https://httpbin.org/json')
            if test_content.success:
                health_status['connectivity'] = 'healthy'
            else:
                health_status['connectivity'] = 'limited'
        except:
            health_status['connectivity'] = 'offline'
        
        overall_status = 'healthy' if all(
            status in ['healthy', 'limited'] for status in health_status.values() 
            if status != health_status['timestamp']
        ) else 'unhealthy'
        
        return jsonify({
            'success': True,
            'overall_status': overall_status,
            'components': health_status
        })
        
    except Exception as e:
        logger.error(f"❌ Health check error: {str(e)}")
        return jsonify({
            'success': False,
            'overall_status': 'unhealthy',
            'error': str(e)
        }), 500

@web_nav_bp.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clears the API cache"""
    try:
        cache_size_before = len(api_manager.result_cache)
        api_manager.result_cache.clear()
        
        logger.info(f"🧹 Cache cleared: {cache_size_before} entries removed")
        
        return jsonify({
            'success': True,
            'message': f'Cache cleared successfully ({cache_size_before} entries removed)',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Cache clearing error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# API Documentation Route
@web_nav_bp.route('/docs', methods=['GET'])
def api_documentation():
    """API Documentation"""
    docs = {
        'api_name': 'Advanced Web Navigation API for artificial intelligence API GOOGLE GEMINI 2.0 FLASH',
        'version': '1.0',
        'description': 'API allowing artificial intelligence API GOOGLE GEMINI 2.0 FLASH to navigate websites and extract structured content',
        'endpoints': {
            'POST /api/web-navigation/create-session': {
                'description': 'Creates a new navigation session',
                'parameters': {
                    'user_id': 'string (optional)',
                    'config': 'object (optional) - Session configuration'
                }
            },
            'POST /api/web-navigation/search-and-navigate': {
                'description': 'Searches and navigates websites',
                'parameters': {
                    'query': 'string (required) - Search query',
                    'session_id': 'string (optional)',
                    'user_context': 'string (optional)',
                    'use_cache': 'boolean (optional, default: true)'
                }
            },
            'POST /api/web-navigation/extract-content': {
                'description': 'Extracts specific content from a URL',
                'parameters': {
                    'url': 'string (required)',
                    'requirements': 'array (optional) - Types of content to extract',
                    'session_id': 'string (optional)',
                    'use_cache': 'boolean (optional, default: true)'
                }
            },
            'POST /api/web-navigation/navigate-deep': {
                'description': 'Deep navigation into a site',
                'parameters': {
                    'start_url': 'string (required)',
                    'max_depth': 'integer (optional, default: 3, max: 5)',
                    'max_pages': 'integer (optional, default: 10, max: 50)',
                    'session_id': 'string (optional)'
                }
            },
            'POST /api/web-navigation/user-journey': {
                'description': 'Simulates a user journey',
                'parameters': {
                    'start_url': 'string (required)',
                    'user_intent': 'string (required) - buy, learn, contact, explore',
                    'session_id': 'string (optional)'
                }
            },
            'GET /api/web-navigation/session/<session_id>': {
                'description': 'Retrieves session information'
            },
            'DELETE /api/web-navigation/session/<session_id>': {
                'description': 'Deletes a session'
            },
            'GET /api/web-navigation/stats': {
                'description': 'Retrieves API statistics'
            },
            'GET /api/web-navigation/health': {
                'description': 'API health check'
            },
            'POST /api/web-navigation/clear-cache': {
                'description': 'Clears the API cache'
            }
        },
        'examples': {
            'search_request': {
                'query': 'artificial intelligence machine learning',
                'user_context': 'developer user interested in AI',
                'use_cache': True
            },
            'extract_request': {
                'url': 'https://example.com/article',
                'requirements': ['summary', 'details', 'links', 'images']
            },
            'navigate_request': {
                'start_url': 'https://example.com',
                'max_depth': 2,
                'max_pages': 5
            }
        }
    }
    
    return jsonify(docs)

def register_web_navigation_api(app: Flask):
    """Registers the web navigation API in the Flask application"""
    app.register_blueprint(web_nav_bp)
    logger.info("🔌 Web Navigation API registered")

# Initialization function for integration with the main app
def initialize_web_navigation_api(searx_interface=None):
    """Initializes the web navigation API"""
    try:
        # Initialize Gemini-Web integration
        initialize_gemini_web_integration(searx_interface)
        
        logger.info("🚀 Web Navigation API initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ API initialization error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Web Navigation API Test ===")
    
    # Create a test Flask app
    app = Flask(__name__)
    register_web_navigation_api(app)
    
    print("✅ API registered successfully")
    print("📡 Available Endpoints:")
    print("  - POST /api/web-navigation/search-and-navigate")
    print("  - POST /api/web-navigation/extract-content")
    print("  - POST /api/web-navigation/navigate-deep")
    print("  - POST /api/web-navigation/user-journey")
    print("  - GET  /api/web-navigation/docs")
    print("  - GET  /api/web-navigation/health")
    
    # Health test
    with app.test_client() as client:
        response = client.get('/api/web-navigation/health')
        if response.status_code == 200:
            print("✅ Health test successful")
        else:
            print("❌ Health test failed")
