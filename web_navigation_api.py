"""
API REST pour le Système de Navigation Web Avancé
Cette API expose toutes les fonctionnalités de navigation web pour l'API Gemini
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

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WebNavigationAPI')

# Blueprint pour l'API
web_nav_bp = Blueprint('web_navigation', __name__, url_prefix='/api/web-navigation')

class WebNavigationAPIManager:
    """Gestionnaire de l'API de navigation web"""
    
    def __init__(self):
        self.active_sessions = {}
        self.session_timeout = 3600  # 1 heure
        self.max_concurrent_sessions = 10
        
        # Statistiques
        self.stats = {
            'total_searches': 0,
            'total_pages_extracted': 0,
            'total_content_characters': 0,
            'successful_navigations': 0,
            'failed_navigations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Cache des résultats récents
        self.result_cache = {}
        self.cache_max_size = 100
        
        logger.info("✅ API Manager de Navigation Web initialisé")
    
    def create_session(self, user_id: str, session_config: Dict[str, Any] = None) -> str:
        """Crée une nouvelle session de navigation"""
        session_id = f"nav_session_{user_id}_{int(time.time())}"
        
        # Configuration par défaut
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
        
        # Nettoyer les anciennes sessions
        self._cleanup_old_sessions()
        
        logger.info(f"🆕 Session créée: {session_id} pour utilisateur {user_id}")
        return session_id
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'une session"""
        return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """Met à jour l'activité d'une session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['last_activity'] = datetime.now()
            self.active_sessions[session_id]['requests_count'] += 1
    
    def _cleanup_old_sessions(self):
        """Nettoie les sessions expirées"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            if (current_time - session_data['last_activity']).seconds > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info(f"🗑️ Session expirée supprimée: {session_id}")
    
    def get_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Génère une clé de cache"""
        cache_data = {
            'query': query,
            'params': sorted(params.items())
        }
        return str(hash(json.dumps(cache_data, sort_keys=True)))
    
    def get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Récupère un résultat du cache"""
        if cache_key in self.result_cache:
            result, timestamp = self.result_cache[cache_key]
            # Cache valide pendant 30 minutes
            if (datetime.now() - timestamp).seconds < 1800:
                self.stats['cache_hits'] += 1
                return result
            else:
                del self.result_cache[cache_key]
        
        self.stats['cache_misses'] += 1
        return None
    
    def save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Sauvegarde un résultat en cache"""
        if len(self.result_cache) >= self.cache_max_size:
            # Supprimer le plus ancien
            oldest_key = min(self.result_cache.keys(), 
                           key=lambda k: self.result_cache[k][1])
            del self.result_cache[oldest_key]
        
        self.result_cache[cache_key] = (result, datetime.now())
    
    def update_stats(self, operation: str, **kwargs):
        """Met à jour les statistiques"""
        if operation == 'search':
            self.stats['total_searches'] += 1
            if kwargs.get('success', False):
                self.stats['successful_navigations'] += 1
                self.stats['total_pages_extracted'] += kwargs.get('pages_extracted', 0)
                self.stats['total_content_characters'] += kwargs.get('content_characters', 0)
            else:
                self.stats['failed_navigations'] += 1

# Instance globale du manager
api_manager = WebNavigationAPIManager()

@web_nav_bp.route('/create-session', methods=['POST'])
def create_navigation_session():
    """Crée une nouvelle session de navigation"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'anonymous')
        session_config = data.get('config', {})
        
        session_id = api_manager.create_session(user_id, session_config)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'config': api_manager.get_session_info(session_id)['config'],
            'message': 'Session de navigation créée avec succès'
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur création session: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/search-and-navigate', methods=['POST'])
def search_and_navigate():
    """Recherche et navigue dans les sites web"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Données JSON requises'}), 400
        
        query = data.get('query')
        if not query:
            return jsonify({'success': False, 'error': 'Paramètre "query" requis'}), 400
        
        session_id = data.get('session_id')
        user_context = data.get('user_context', '')
        use_cache = data.get('use_cache', True)
        
        # Vérifier la session si fournie
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Session invalide'}), 400
            api_manager.update_session_activity(session_id)
        
        # Vérifier le cache
        cache_key = api_manager.get_cache_key(query, {
            'user_context': user_context,
            'operation': 'search_and_navigate'
        })
        
        if use_cache:
            cached_result = api_manager.get_from_cache(cache_key)
            if cached_result:
                logger.info(f"📋 Résultat récupéré du cache pour: {query}")
                return jsonify(cached_result)
        
        # Initialiser l'intégration si nécessaire
        if not gemini_web_integration:
            initialize_gemini_web_integration()
        
        # Effectuer la recherche et navigation
        logger.info(f"🔍 Début recherche et navigation: {query}")
        start_time = time.time()
        
        result = search_web_for_gemini(query, user_context)
        
        processing_time = time.time() - start_time
        
        # Enrichir le résultat avec des métadonnées API
        api_result = {
            'api_version': '1.0',
            'processing_time': round(processing_time, 2),
            'session_id': session_id,
            'cache_used': False,
            'timestamp': datetime.now().isoformat(),
            **result
        }
        
        # Mettre à jour les statistiques
        api_manager.update_stats(
            'search',
            success=result.get('success', False),
            pages_extracted=result.get('search_summary', {}).get('total_pages_visited', 0),
            content_characters=len(str(result.get('content_synthesis', '')))
        )
        
        # Sauvegarder en cache
        if use_cache and result.get('success', False):
            api_manager.save_to_cache(cache_key, api_result)
        
        logger.info(f"✅ Recherche terminée en {processing_time:.2f}s")
        return jsonify(api_result)
        
    except Exception as e:
        logger.error(f"❌ Erreur recherche et navigation: {str(e)}")
        api_manager.update_stats('search', success=False)
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/extract-content', methods=['POST'])
def extract_specific_content():
    """Extrait le contenu spécifique d'une URL"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Données JSON requises'}), 400
        
        url = data.get('url')
        if not url:
            return jsonify({'success': False, 'error': 'Paramètre "url" requis'}), 400
        
        requirements = data.get('requirements', ['summary', 'details', 'links'])
        session_id = data.get('session_id')
        use_cache = data.get('use_cache', True)
        
        # Vérifier la session
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Session invalide'}), 400
            api_manager.update_session_activity(session_id)
        
        # Vérifier le cache
        cache_key = api_manager.get_cache_key(url, {
            'requirements': requirements,
            'operation': 'extract_content'
        })
        
        if use_cache:
            cached_result = api_manager.get_from_cache(cache_key)
            if cached_result:
                logger.info(f"📋 Extraction récupérée du cache pour: {url}")
                return jsonify(cached_result)
        
        # Extraire le contenu
        logger.info(f"🎯 Extraction de contenu: {url}")
        start_time = time.time()
        
        result = extract_content_for_gemini(url, requirements)
        
        processing_time = time.time() - start_time
        
        # Enrichir le résultat
        api_result = {
            'api_version': '1.0',
            'processing_time': round(processing_time, 2),
            'session_id': session_id,
            'cache_used': False,
            'requirements_requested': requirements,
            'timestamp': datetime.now().isoformat(),
            **result
        }
        
        # Sauvegarder en cache
        if use_cache and result.get('success', False):
            api_manager.save_to_cache(cache_key, api_result)
        
        logger.info(f"✅ Extraction terminée en {processing_time:.2f}s")
        return jsonify(api_result)
        
    except Exception as e:
        logger.error(f"❌ Erreur extraction contenu: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/navigate-deep', methods=['POST'])
def navigate_website_deep_api():
    """Navigation en profondeur dans un site web"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Données JSON requises'}), 400
        
        start_url = data.get('start_url')
        if not start_url:
            return jsonify({'success': False, 'error': 'Paramètre "start_url" requis'}), 400
        
        max_depth = data.get('max_depth', 3)
        max_pages = data.get('max_pages', 10)
        session_id = data.get('session_id')
        
        # Valider les paramètres
        if max_depth > 5:
            return jsonify({'success': False, 'error': 'max_depth ne peut pas dépasser 5'}), 400
        if max_pages > 50:
            return jsonify({'success': False, 'error': 'max_pages ne peut pas dépasser 50'}), 400
        
        # Vérifier la session
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Session invalide'}), 400
            api_manager.update_session_activity(session_id)
        
        # Navigation en profondeur
        logger.info(f"🚀 Navigation profonde: {start_url} (profondeur: {max_depth}, pages: {max_pages})")
        start_time = time.time()
        
        nav_path = navigate_website_deep(start_url, max_depth, max_pages)
        
        processing_time = time.time() - start_time
        
        # Préparer la réponse
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
                    'keywords': page.keywords[:10],  # Top 10 mots-clés
                    'language': page.language,
                    'links_count': len(page.links),
                    'images_count': len(page.images)
                }
                for page in nav_path.visited_pages
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Navigation profonde terminée en {processing_time:.2f}s - {len(nav_path.visited_pages)} pages")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Erreur navigation profonde: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/user-journey', methods=['POST'])
def simulate_user_journey_api():
    """Simule un parcours utilisateur sur un site"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Données JSON requises'}), 400
        
        start_url = data.get('start_url')
        user_intent = data.get('user_intent', 'explore')
        session_id = data.get('session_id')
        
        if not start_url:
            return jsonify({'success': False, 'error': 'Paramètre "start_url" requis'}), 400
        
        # Valider l'intention
        valid_intents = ['buy', 'learn', 'contact', 'explore']
        if user_intent not in valid_intents:
            return jsonify({
                'success': False, 
                'error': f'user_intent doit être un de: {valid_intents}'
            }), 400
        
        # Vérifier la session
        if session_id:
            session_info = api_manager.get_session_info(session_id)
            if not session_info:
                return jsonify({'success': False, 'error': 'Session invalide'}), 400
            api_manager.update_session_activity(session_id)
        
        # Simuler le parcours utilisateur
        logger.info(f"👤 Simulation parcours utilisateur: {user_intent} depuis {start_url}")
        start_time = time.time()
        
        result = simulate_user_journey(start_url, user_intent)
        
        processing_time = time.time() - start_time
        
        # Enrichir le résultat
        api_result = {
            'api_version': '1.0',
            'processing_time': round(processing_time, 2),
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            **result
        }
        
        logger.info(f"✅ Parcours utilisateur terminé en {processing_time:.2f}s")
        return jsonify(api_result)
        
    except Exception as e:
        logger.error(f"❌ Erreur parcours utilisateur: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@web_nav_bp.route('/session/<session_id>', methods=['GET'])
def get_session_info_api(session_id: str):
    """Récupère les informations d'une session"""
    try:
        session_info = api_manager.get_session_info(session_id)
        
        if not session_info:
            return jsonify({'success': False, 'error': 'Session non trouvée'}), 404
        
        # Préparer les informations de session
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
        logger.error(f"❌ Erreur récupération session: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/session/<session_id>', methods=['DELETE'])
def delete_session_api(session_id: str):
    """Supprime une session"""
    try:
        if session_id in api_manager.active_sessions:
            del api_manager.active_sessions[session_id]
            logger.info(f"🗑️ Session supprimée: {session_id}")
            return jsonify({
                'success': True,
                'message': f'Session {session_id} supprimée avec succès'
            })
        else:
            return jsonify({'success': False, 'error': 'Session non trouvée'}), 404
            
    except Exception as e:
        logger.error(f"❌ Erreur suppression session: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/stats', methods=['GET'])
def get_api_stats():
    """Récupère les statistiques de l'API"""
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
        logger.error(f"❌ Erreur récupération statistiques: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@web_nav_bp.route('/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    try:
        # Vérifier l'état des composants
        health_status = {
            'api': 'healthy',
            'navigator': 'healthy' if advanced_navigator else 'unavailable',
            'integration': 'healthy' if gemini_web_integration else 'unavailable',
            'cache': 'healthy',
            'timestamp': datetime.now().isoformat()
        }
        
        # Test simple de navigation
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
        logger.error(f"❌ Erreur vérification santé: {str(e)}")
        return jsonify({
            'success': False,
            'overall_status': 'unhealthy',
            'error': str(e)
        }), 500

@web_nav_bp.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Vide le cache de l'API"""
    try:
        cache_size_before = len(api_manager.result_cache)
        api_manager.result_cache.clear()
        
        logger.info(f"🧹 Cache vidé: {cache_size_before} entrées supprimées")
        
        return jsonify({
            'success': True,
            'message': f'Cache vidé avec succès ({cache_size_before} entrées supprimées)',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur vidage cache: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Route de documentation API
@web_nav_bp.route('/docs', methods=['GET'])
def api_documentation():
    """Documentation de l'API"""
    docs = {
        'api_name': 'API de Navigation Web Avancée pour Gemini',
        'version': '1.0',
        'description': 'API permettant à Gemini de naviguer dans les sites web et d\'extraire du contenu structuré',
        'endpoints': {
            'POST /api/web-navigation/create-session': {
                'description': 'Crée une nouvelle session de navigation',
                'parameters': {
                    'user_id': 'string (optionnel)',
                    'config': 'object (optionnel) - Configuration de la session'
                }
            },
            'POST /api/web-navigation/search-and-navigate': {
                'description': 'Recherche et navigue dans les sites web',
                'parameters': {
                    'query': 'string (requis) - Requête de recherche',
                    'session_id': 'string (optionnel)',
                    'user_context': 'string (optionnel)',
                    'use_cache': 'boolean (optionnel, défaut: true)'
                }
            },
            'POST /api/web-navigation/extract-content': {
                'description': 'Extrait le contenu spécifique d\'une URL',
                'parameters': {
                    'url': 'string (requis)',
                    'requirements': 'array (optionnel) - Types de contenu à extraire',
                    'session_id': 'string (optionnel)',
                    'use_cache': 'boolean (optionnel, défaut: true)'
                }
            },
            'POST /api/web-navigation/navigate-deep': {
                'description': 'Navigation en profondeur dans un site',
                'parameters': {
                    'start_url': 'string (requis)',
                    'max_depth': 'integer (optionnel, défaut: 3, max: 5)',
                    'max_pages': 'integer (optionnel, défaut: 10, max: 50)',
                    'session_id': 'string (optionnel)'
                }
            },
            'POST /api/web-navigation/user-journey': {
                'description': 'Simule un parcours utilisateur',
                'parameters': {
                    'start_url': 'string (requis)',
                    'user_intent': 'string (requis) - buy, learn, contact, explore',
                    'session_id': 'string (optionnel)'
                }
            },
            'GET /api/web-navigation/session/<session_id>': {
                'description': 'Récupère les informations d\'une session'
            },
            'DELETE /api/web-navigation/session/<session_id>': {
                'description': 'Supprime une session'
            },
            'GET /api/web-navigation/stats': {
                'description': 'Récupère les statistiques de l\'API'
            },
            'GET /api/web-navigation/health': {
                'description': 'Vérification de l\'état de l\'API'
            },
            'POST /api/web-navigation/clear-cache': {
                'description': 'Vide le cache de l\'API'
            }
        },
        'examples': {
            'search_request': {
                'query': 'intelligence artificielle apprentissage automatique',
                'user_context': 'utilisateur développeur intéressé par l\'IA',
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
    """Enregistre l'API de navigation web dans l'application Flask"""
    app.register_blueprint(web_nav_bp)
    logger.info("🔌 API de Navigation Web enregistrée")

# Fonction d'initialisation pour l'intégration avec l'app principale
def initialize_web_navigation_api(searx_interface=None):
    """Initialise l'API de navigation web"""
    try:
        # Initialiser l'intégration Gemini-Web
        initialize_gemini_web_integration(searx_interface)
        
        logger.info("🚀 API de Navigation Web initialisée avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation API: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Test de l'API de Navigation Web ===")
    
    # Créer une app Flask de test
    app = Flask(__name__)
    register_web_navigation_api(app)
    
    print("✅ API enregistrée avec succès")
    print("📡 Endpoints disponibles:")
    print("  - POST /api/web-navigation/search-and-navigate")
    print("  - POST /api/web-navigation/extract-content")
    print("  - POST /api/web-navigation/navigate-deep")
    print("  - POST /api/web-navigation/user-journey")
    print("  - GET  /api/web-navigation/docs")
    print("  - GET  /api/web-navigation/health")
    
    # Test de santé
    with app.test_client() as client:
        response = client.get('/api/web-navigation/health')
        if response.status_code == 200:
            print("✅ Test de santé réussi")
        else:
            print("❌ Test de santé échoué")
