"""
API REST pour les Capacités Visuelles de Gemini Web
Endpoints pour navigation avec vision intégrée
"""

from flask import Flask, request, jsonify, send_file
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

# Import des systèmes de vision
try:
    from gemini_web_vision_integration import initialize_gemini_web_vision, get_gemini_web_vision
    from gemini_visual_adapter import get_gemini_visual_adapter
    from intelligent_web_capture import get_intelligent_capture
    VISION_SYSTEMS_AVAILABLE = True
except ImportError as e:
    VISION_SYSTEMS_AVAILABLE = False
    logging.error(f"❌ Systèmes de vision non disponibles: {e}")

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GeminiWebVisionAPI')

class GeminiWebVisionAPI:
    """API REST pour les capacités visuelles de navigation web"""
    
    def __init__(self, app: Flask = None):
        """
        Initialise l'API Vision Web
        
        Args:
            app: Instance Flask optionnelle
        """
        self.app = app or Flask(__name__)
        self.vision_integration = None
        
        if VISION_SYSTEMS_AVAILABLE:
            try:
                self.vision_integration = initialize_gemini_web_vision()
                logger.info("✅ Systèmes de vision initialisés pour l'API")
            except Exception as e:
                logger.error(f"❌ Erreur initialisation systèmes de vision: {e}")
        
        # Configurer les routes
        self._setup_routes()
        
        logger.info("🚀 API Gemini Web Vision initialisée")
    
    def _setup_routes(self):
        """Configure toutes les routes de l'API"""
        
        @self.app.route('/api/vision/health', methods=['GET'])
        def health_check():
            """Vérification de l'état de l'API Vision"""
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
            """Crée une nouvelle session de navigation avec vision"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Données JSON requises'}), 400
                
                session_id = data.get('session_id')
                user_query = data.get('user_query')
                navigation_goals = data.get('navigation_goals', ['extract_content', 'analyze_ui', 'capture_visuals'])
                
                if not session_id or not user_query:
                    return jsonify({'error': 'session_id et user_query requis'}), 400
                
                if not self.vision_integration:
                    return jsonify({'error': 'Système de vision non disponible'}), 503
                
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
                logger.error(f"❌ Erreur création session: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/navigate', methods=['POST'])
        def navigate_with_vision():
            """Navigation avec capture et analyse visuelle"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Données JSON requises'}), 400
                
                session_id = data.get('session_id')
                url = data.get('url')
                navigation_type = data.get('navigation_type', 'smart_exploration')
                capture_config = data.get('capture_config')
                
                if not session_id or not url:
                    return jsonify({'error': 'session_id et url requis'}), 400
                
                if not self.vision_integration:
                    return jsonify({'error': 'Système de vision non disponible'}), 503
                
                result = self.vision_integration.navigate_with_vision(
                    session_id=session_id,
                    url=url,
                    navigation_type=navigation_type,
                    capture_config=capture_config
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Erreur navigation avec vision: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/capture', methods=['POST'])
        def capture_website():
            """Capture intelligente d'un site web"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Données JSON requises'}), 400
                
                url = data.get('url')
                capture_type = data.get('capture_type', 'full_page')
                viewport = data.get('viewport', 'desktop')
                analyze_elements = data.get('analyze_elements', True)
                
                if not url:
                    return jsonify({'error': 'url requise'}), 400
                
                capture_system = get_intelligent_capture()
                if not capture_system:
                    return jsonify({'error': 'Système de capture non disponible'}), 503
                
                result = capture_system.capture_website_intelligent(
                    url=url,
                    capture_type=capture_type,
                    viewport=viewport,
                    analyze_elements=analyze_elements
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Erreur capture site: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/analyze', methods=['POST'])
        def analyze_visual():
            """Analyse visuelle d'une capture d'écran"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Données JSON requises'}), 400
                
                image_path = data.get('image_path')
                analysis_prompt = data.get('analysis_prompt', 'Analysez cette capture d\'écran de site web')
                context = data.get('context')
                
                if not image_path:
                    return jsonify({'error': 'image_path requis'}), 400
                
                visual_adapter = get_gemini_visual_adapter()
                if not visual_adapter:
                    return jsonify({'error': 'Adaptateur visuel non disponible'}), 503
                
                result = visual_adapter.analyze_website_screenshot(
                    image_path=image_path,
                    analysis_prompt=analysis_prompt,
                    context=context
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Erreur analyse visuelle: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/compare', methods=['POST'])
        def compare_sites():
            """Comparaison visuelle de deux sites"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Données JSON requises'}), 400
                
                session_id = data.get('session_id', f'comparison_{int(datetime.now().timestamp())}')
                url1 = data.get('url1')
                url2 = data.get('url2')
                comparison_focus = data.get('comparison_focus', 'general')
                
                if not url1 or not url2:
                    return jsonify({'error': 'url1 et url2 requis'}), 400
                
                if not self.vision_integration:
                    return jsonify({'error': 'Système de vision non disponible'}), 503
                
                result = self.vision_integration.analyze_site_comparison(
                    session_id=session_id,
                    url1=url1,
                    url2=url2,
                    comparison_focus=comparison_focus
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Erreur comparaison sites: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/ui-analysis', methods=['POST'])
        def ui_analysis():
            """Analyse spécialisée des éléments UI"""
            try:
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'Données JSON requises'}), 400
                
                image_path = data.get('image_path')
                element_types = data.get('element_types', ['buttons', 'forms', 'navigation', 'content'])
                
                if not image_path:
                    return jsonify({'error': 'image_path requis'}), 400
                
                visual_adapter = get_gemini_visual_adapter()
                if not visual_adapter:
                    return jsonify({'error': 'Adaptateur visuel non disponible'}), 503
                
                result = visual_adapter.analyze_ui_elements(
                    image_path=image_path,
                    element_types=element_types
                )
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Erreur analyse UI: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/session/<session_id>', methods=['GET'])
        def get_session_info(session_id):
            """Obtient les informations d'une session"""
            try:
                if not self.vision_integration:
                    return jsonify({'error': 'Système de vision non disponible'}), 503
                
                session_info = self.vision_integration.active_sessions.get(session_id)
                
                if not session_info:
                    return jsonify({'error': f'Session {session_id} non trouvée'}), 404
                
                return jsonify({
                    'success': True,
                    'session_id': session_id,
                    'session_info': session_info
                }), 200
                
            except Exception as e:
                logger.error(f"❌ Erreur récupération session: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/session/<session_id>', methods=['DELETE'])
        def close_session(session_id):
            """Ferme une session de navigation"""
            try:
                if not self.vision_integration:
                    return jsonify({'error': 'Système de vision non disponible'}), 503
                
                result = self.vision_integration.close_session(session_id)
                
                return jsonify(result), 200 if result['success'] else 500
                
            except Exception as e:
                logger.error(f"❌ Erreur fermeture session: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/statistics', methods=['GET'])
        def get_statistics():
            """Obtient les statistiques du système de vision"""
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
                logger.error(f"❌ Erreur récupération statistiques: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/image/<path:image_path>', methods=['GET'])
        def serve_image(image_path):
            """Sert les images capturées"""
            try:
                # Vérifier que le chemin est sécurisé
                if '..' in image_path or image_path.startswith('/'):
                    return jsonify({'error': 'Chemin non autorisé'}), 403
                
                # Construire le chemin complet
                full_path = Path('intelligent_screenshots') / image_path
                
                if not full_path.exists():
                    return jsonify({'error': 'Image non trouvée'}), 404
                
                return send_file(str(full_path))
                
            except Exception as e:
                logger.error(f"❌ Erreur service image: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/vision/docs', methods=['GET'])
        def get_documentation():
            """Documentation de l'API Vision"""
            docs = {
                'title': 'API Gemini Web Vision',
                'version': '1.0.0',
                'description': 'API pour la navigation web avec capacités visuelles Gemini',
                'endpoints': {
                    'GET /api/vision/health': 'Vérification de l\'état de l\'API',
                    'POST /api/vision/create-session': 'Créer une session de navigation avec vision',
                    'POST /api/vision/navigate': 'Naviguer avec capture et analyse visuelle',
                    'POST /api/vision/capture': 'Capturer un site web intelligemment',
                    'POST /api/vision/analyze': 'Analyser visuellement une capture',
                    'POST /api/vision/compare': 'Comparer visuellement deux sites',
                    'POST /api/vision/ui-analysis': 'Analyse spécialisée des éléments UI',
                    'GET /api/vision/session/<id>': 'Obtenir les infos d\'une session',
                    'DELETE /api/vision/session/<id>': 'Fermer une session',
                    'GET /api/vision/statistics': 'Statistiques du système',
                    'GET /api/vision/image/<path>': 'Servir les images capturées',
                },
                'examples': {
                    'create_session': {
                        'method': 'POST',
                        'url': '/api/vision/create-session',
                        'body': {
                            'session_id': 'ma_session_123',
                            'user_query': 'Analyser l\'UX de ce site e-commerce',
                            'navigation_goals': ['extract_content', 'analyze_ui', 'capture_visuals']
                        }
                    },
                    'navigate_with_vision': {
                        'method': 'POST',
                        'url': '/api/vision/navigate',
                        'body': {
                            'session_id': 'ma_session_123',
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

# Instance globale de l'API
vision_api = None

def create_vision_api(app: Flask = None) -> GeminiWebVisionAPI:
    """
    Crée l'instance de l'API Vision Web
    
    Args:
        app: Instance Flask optionnelle
        
    Returns:
        Instance de l'API
    """
    global vision_api
    
    if vision_api is None:
        vision_api = GeminiWebVisionAPI(app)
        logger.info("🚀 API Gemini Web Vision créée")
    
    return vision_api

def get_vision_api() -> Optional[GeminiWebVisionAPI]:
    """
    Retourne l'instance globale de l'API Vision
    
    Returns:
        Instance de l'API ou None
    """
    global vision_api
    return vision_api

def register_vision_routes(app: Flask):
    """
    Enregistre les routes de vision dans une app Flask existante
    
    Args:
        app: Instance Flask
    """
    api = create_vision_api(app)
    logger.info("📡 Routes de vision enregistrées dans l'application Flask")

if __name__ == "__main__":
    # Test standalone de l'API
    app = Flask(__name__)
    api = create_vision_api(app)
    
    print("🧪 API Gemini Web Vision prête pour les tests")
    print("🌐 Endpoints disponibles:")
    print("  - GET  /api/vision/health")
    print("  - POST /api/vision/create-session")
    print("  - POST /api/vision/navigate")
    print("  - POST /api/vision/capture")
    print("  - POST /api/vision/analyze")
    print("  - POST /api/vision/compare")
    print("  - GET  /api/vision/docs")
    
    app.run(debug=True, port=5001)
