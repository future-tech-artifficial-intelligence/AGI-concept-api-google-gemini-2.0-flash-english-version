"""
**API Route for Configuring Which Artificial Intelligence API to Use.**
This module exposes a REST API to configure the artificial intelligence API to be used.
"""

from flask import Blueprint, request, jsonify, session
from ai_api_manager import get_ai_api_manager
import logging

# Logger configuration
logger = logging.getLogger(__name__)

# Create a Flask Blueprint
api_config_bp = Blueprint('api_config', __name__)

@api_config_bp.route('/api/config/apis', methods=['GET'])
def get_available_apis():
    """Retrieves the list of available AI APIs."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    try:
        api_manager = get_ai_api_manager()
        available_apis = api_manager.list_available_apis()
        current_api = api_manager.get_current_api_name()
        
        return jsonify({
            'available_apis': available_apis,
            'current_api': current_api
        })
    except Exception as e:
        logger.error(f"Error retrieving available APIs: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

@api_config_bp.route('/api/config/apis/current', methods=['GET', 'POST'])
def manage_current_api():
    """Retrieves or modifies the active AI API."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    api_manager = get_ai_api_manager()
    
    # Retrieve the active API
    if request.method == 'GET':
        try:
            current_api = api_manager.get_current_api_name()
            return jsonify({'current_api': current_api})
        except Exception as e:
            logger.error(f"Error retrieving the active API: {str(e)}")
            return jsonify({'error': 'An error occurred'}), 500
    
    # Change the active API
    elif request.method == 'POST':
        try:
            data = request.json
            new_api = data.get('api_name')
            
            if not new_api:
                return jsonify({'error': 'API name missing'}), 400
            
            available_apis = api_manager.list_available_apis()
            if new_api not in available_apis:
                return jsonify({'error': f'API "{new_api}" not available'}), 400
            
            result = api_manager.set_api(new_api)
            if result:
                return jsonify({
                    'success': True, 
                    'message': f'API changed to "{new_api}"',
                    'current_api': new_api
                })
            else:
                logger.error(f"Failed to change API to {new_api}")
                return jsonify({'error': 'Failed to change API'}), 500
        except Exception as e:
            logger.error(f"Error changing API: {str(e)}")
            return jsonify({'error': 'An error occurred'}), 500

@api_config_bp.route('/api/config/apis/<api_name>', methods=['GET', 'POST'])
def configure_api(api_name):
    """Retrieves or modifies the configuration of a specific API."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401
    
    api_manager = get_ai_api_manager()
    
    # Verify that the API exists
    available_apis = api_manager.list_available_apis()
    if api_name not in available_apis:
        return jsonify({'error': f'API "{api_name}" not available'}), 404
      # Retrieve API configuration
    if request.method == 'GET':
        try:
            # Retrieve current configuration
            config = api_manager.get_api_config(api_name) if hasattr(api_manager, 'get_api_config') else {}
            
            # If the method does not exist, try to directly access the configuration
            if not config and hasattr(api_manager, 'config') and 'apis' in api_manager.config:
                config = api_manager.config['apis'].get(api_name, {})
                
            return jsonify({
                'api_name': api_name,
                'config': config
            })
        except Exception as e:
            logger.error(f"Error retrieving configuration for API {api_name}: {str(e)}")
            return jsonify({'error': 'An error occurred'}), 500
    
    # Modify API configuration
    elif request.method == 'POST':
        try:
            data = request.json
            api_config = data.get('config', {})
            
            result = api_manager.configure_api(api_name, api_config)
            if result:
                return jsonify({
                    'success': True, 
                    'message': f'Configuration for API "{api_name}" updated'
                })
            else:
                logger.error(f"Failed to update configuration for API {api_name}")
                return jsonify({'error': 'Failed to update configuration'}), 500
        except Exception as e:
            logger.error(f"Error updating configuration for API {api_name}: {str(e)}")
            return jsonify({'error': 'An error occurred'}), 500
