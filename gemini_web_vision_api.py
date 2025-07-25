"""
REST API for artificial intelligence Google Gemini 2.0 Flash  Web Visual Capabilities
Endpoints for navigation with integrated vision
"""

from flask import Flask, request, jsonify, send_file
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import vision systems
try:
    from gemini_web_vision_integration import initialize_gemini_web_vision, get_gemini_web_vision
    from gemini_visual_adapter import get_gemini_visual_adapter
    from intelligent_web_capture import get_intelligent_capture
    VISION_SYSTEMS_AVAILABLE = True
except ImportError as e:
    VISION_SYSTEMS_AVAILABLE = False
    logging.error(f"❌ Vision systems not available: {e}")

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Google Gemini 2.0 Flash AI WebVisionAPI')

class GeminiWebVisionAPI:
    """REST API for web visual capabilities"""
    
    def __init__(self, app: Flask = None):
        """
        Initializes the Web Vision API
        
        Args:
            app: Optional Flask instance
        """
        self.app = app or Flask(__name__)
        self.vision_integration = None
        
        if VISION_SYSTEMS_AVAILABLE:
            try:
                self.vision_integration = initialize_gemini_web_vision()
                logger.info("✅ Vision systems initialized for the API")
            except Exception as e:
                logger.error(f"❌ Error initializing vision systems: {e}")
        
        # Configure routes
        self._setup_routes()
        
        logger.info("🚀 Google Gemini 2.0 Flash AI Web Vision API initialized")
    
    def _setup_routes(self):
        """Configures all API routes"""
        
        @self.app.route('/api/vision/health', methods=['GET'])
        def health_check():
            """Vision API health check"""
            return jsonify({
                'status': 'healthy',
                'vision_systems_available': VISION_SYSTEMS_AVAILABLE,
                'timestamp': datetime.now().isoformat(),
                'components': {
                    'vision_integration': self.vision_integration is not None,
                    'visual_adapter': get_gemini_visual_adapter() is not None,
                    'capture_system': get_intelligent_capture() is not None
                }
            })
        
        @self.app.route('/api/vision/create-session', methods=['POST'])
        def create_vision_session():
            """Creates a new navigation session with vision"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'JSON data required'}), 400
                
                session_id = data.get('session_id')
                user_query = data.get('user_query')
                navigation_goals = data.get('navigation_goals', ['extract_content', 'analyze_ui', 'capture_visuals'])
                
                if not session_id or not user_query:
                    return jsonify({'error': 'session_id and user_query required'}), 400
                
                if not self.vision_integration:
                    return jsonify({'error': 'Vision system not available'}), 503
                
                result = self.vision_integration.create_vision_navigation_session(
                    session_id=session_id,
                    user_query=user_query,
                    navigation_goals=navigation_goals
                )
                
                if result['success']:
                    return jsonify(result), 201
                else:
                    return jsonify(result), 500
                    
            except Exception as e:
                logger.error(f"❌ Session creation error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/navigate', methods=['POST'])
        def navigate_with_vision():
            """Navigation with visual capture and analysis"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'JSON data required'}), 400
                
                session_id = data.get('session_id')
                url = data.get('url')
                navigation_type = data.get('navigation_type', 'smart_exploration')
                capture_config = data.get('capture_config')
                
                if not session_id or not url:
                    return jsonify({'error': 'session_id and url required'}), 400
                
                if not self.vision_integration:
                    return jsonify({'error': 'Vision system not available'}), 503
                
                result = self.vision_integration.navigate_with_vision(
                    session_id=session_id,
                    url=url,
                    navigation_type=navigation_type,
                    capture_config=capture_config
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Navigation with vision error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/capture', methods=['POST'])
        def capture_website():
            """Intelligent website capture"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'JSON data required'}), 400
                
                url = data.get('url')
                capture_type = data.get('capture_type', 'full_page')
                viewport = data.get('viewport', 'desktop')
                analyze_elements = data.get('analyze_elements', True)
                
                if not url:
                    return jsonify({'error': 'url required'}), 400
                
                capture_system = get_intelligent_capture()
                if not capture_system:
                    return jsonify({'error': 'Capture system not available'}), 503
                
                result = capture_system.capture_website_intelligent(
                    url=url,
                    capture_type=capture_type,
                    viewport=viewport,
                    analyze_elements=analyze_elements
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Website capture error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/analyze', methods=['POST'])
        def analyze_visual():
            """Visual analysis of a screenshot"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'JSON data required'}), 400
                
                image_path = data.get('image_path')
                analysis_prompt = data.get('analysis_prompt', 'Analyze this website screenshot')
                context = data.get('context')
                
                if not image_path:
                    return jsonify({'error': 'image_path required'}), 400
                
                visual_adapter = get_gemini_visual_adapter()
                if not visual_adapter:
                    return jsonify({'error': 'Visual adapter not available'}), 503
                
                result = visual_adapter.analyze_website_screenshot(
                    image_path=image_path,
                    analysis_prompt=analysis_prompt,
                    context=context
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Visual analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/compare', methods=['POST'])
        def compare_sites():
            """Visual comparison of two sites"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'JSON data required'}), 400
                
                session_id = data.get('session_id', f'comparison_{int(datetime.now().timestamp())}')
                url1 = data.get('url1')
                url2 = data.get('url2')
                comparison_focus = data.get('comparison_focus', 'general')
                
                if not url1 or not url2:
                    return jsonify({'error': 'url1 and url2 required'}), 400
                
                if not self.vision_integration:
                    return jsonify({'error': 'Vision system not available'}), 503
                
                result = self.vision_integration.analyze_site_comparison(
                    session_id=session_id,
                    url1=url1,
                    url2=url2,
                    comparison_focus=comparison_focus
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Site comparison error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/ui-analysis', methods=['POST'])
        def ui_analysis():
            """Specialized UI elements analysis"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'JSON data required'}), 400
                
                image_path = data.get('image_path')
                element_types = data.get('element_types', ['buttons', 'forms', 'navigation', 'content'])
                
                if not image_path:
                    return jsonify({'error': 'image_path required'}), 400
                
                visual_adapter = get_gemini_visual_adapter()
                if not visual_adapter:
                    return jsonify({'error': 'Visual adapter not available'}), 503
                
                result = visual_adapter.analyze_ui_elements(
                    image_path=image_path,
                    element_types=element_types
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ UI analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/session/<session_id>', methods=['GET'])
        def get_session_info(session_id):
            """Gets session information"""
            try:
                if not self.vision_integration:
                    return jsonify({'error': 'Vision system not available'}), 503
                
                session_info = self.vision_integration.active_sessions.get(session_id)
                
                if not session_info:
                    return jsonify({'error': f'Session {session_id} not found'}), 404
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'session_info': session_info
                }), 200
                
            except Exception as e:
                logger.error(f"❌ Session retrieval error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/session/<session_id>', methods=['DELETE'])
        def close_session(session_id):
            """Closes a navigation session"""
            try:
                if not self.vision_integration:
                    return jsonify({'error': 'Vision system not available'}), 503
                
                result = self.vision_integration.close_session(session_id)
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Session closing error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/statistics', methods=['GET'])
        def get_statistics():
            """Gets vision system statistics"""
            try:
                stats = {}
                
                if self.vision_integration:
                    stats['integration'] = self.vision_integration.get_statistics()
                
                visual_adapter = get_gemini_visual_adapter()
                if visual_adapter:
                    stats['visual_adapter'] = visual_adapter.get_statistics()
                
                capture_system = get_intelligent_capture()
                if capture_system:
                    stats['capture_system'] = capture_system.get_statistics()
                
                return jsonify({
                    'success': True,
                    'statistics': stats,
                    'timestamp': datetime.now().isoformat()
                }), 200
                
            except Exception as e:
                logger.error(f"❌ Statistics retrieval error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/image/<path:image_path>', methods=['GET'])
        def serve_image(image_path):
            """Serves captured images"""
            try:
                # Check that the path is secure
                if '..' in image_path or image_path.startswith('/'):
                    return jsonify({'error': 'Unauthorized path'}), 403
                
                # Build the full path
                full_path = Path('intelligent_screenshots') / image_path
                
                if not full_path.exists():
                    return jsonify({'error': 'Image not found'}), 404
                
                return send_file(str(full_path))
                
            except Exception as e:
                logger.error(f"❌ Image serving error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/docs', methods=['GET'])
        def get_documentation():
            """Vision API Documentation"""
            docs = {
                'title': 'Google Gemini 2.0 Flash AI Web Vision API',
                'version': '1.0.0',
                'description': 'API for web navigation with Google Gemini 2.0 Flash AI visual capabilities',
                'endpoints': {
                    'GET /api/vision/health': 'API health check',
                    'POST /api/vision/create-session': 'Create a navigation session with vision',
                    'POST /api/vision/navigate': 'Navigate with visual capture and analysis',
                    'POST /api/vision/capture': 'Intelligently capture a website',
                    'POST /api/vision/analyze': 'Visually analyze a capture',
                    'POST /api/vision/compare': 'Visually compare two sites',
                    'POST /api/vision/ui-analysis': 'Specialized UI elements analysis',
                    'GET /api/vision/session/<id>': 'Get session info',
                    'DELETE /api/vision/session/<id>': 'Close a session',
                    'GET /api/vision/statistics': 'System statistics',
                    'GET /api/vision/image/<path>': 'Serve captured images',
                },
                'examples': {
                    'create_session': {
                        'method': 'POST',
                        'url': '/api/vision/create-session',
                        'body': {
                            'session_id': 'my_session_123',
                            'user_query': 'Analyze the UX of this e-commerce site',
                            'navigation_goals': ['extract_content', 'analyze_ui', 'capture_visuals']
                        }
                    },
                    'navigate_with_vision': {
                        'method': 'POST',
                        'url': '/api/vision/navigate',
                        'body': {
                            'session_id': 'my_session_123',
                            'url': 'https://example.com',
                            'navigation_type': 'smart_exploration',
                            'capture_config': {
                                'capture_type': 'full_page',
                                'viewport': 'desktop',
                                'analyze_elements': True
                            }
                        }
                    }
                }
            }
            
            return jsonify(docs), 200

# Global instance of the API
vision_api = None

def create_vision_api(app: Flask = None) -> GeminiWebVisionAPI:
    """
    Creates the Web Vision API instance
    
    Args:
        app: Optional Flask instance
        
    Returns:
        API instance
    """
    global vision_api
    
    if vision_api is None:
        vision_api = GeminiWebVisionAPI(app)
        logger.info("🚀 Google Gemini 2.0 Flash AI Web Vision API created")
    
    return vision_api

def get_vision_api() -> Optional[GeminiWebVisionAPI]:
    """
    Returns the global Vision API instance
    
    Returns:
        API instance or None
    """
    global vision_api
    return vision_api

def register_vision_routes(app: Flask):
    """
    Registers vision routes in an existing Flask app
    
    Args:
        app: Flask instance
    """
    api = create_vision_api(app)
    logger.info("📡 Vision routes registered in the Flask application")

if __name__ == "__main__":
    # Standalone API test
    app = Flask(__name__)
    api = create_vision_api(app)
    
    print("🧪 Google Gemini 2.0 Flash AI Web Vision API ready for tests")
    print("🌐 Available Endpoints:")
    print("  - GET  /api/vision/health")
    print("  - POST /api/vision/create-session")
    print("  - POST /api/vision/navigate")
    print("  - POST /api/vision/capture")
    print("  - POST /api/vision/analyze")
    print("  - POST /api/vision/compare")
    print("  - GET  /api/vision/docs")
    
    app.run(debug=True, port=5001)
