"""
**API Routes for configuring and managing API keys of various integrations.**
"""

from flask import Blueprint, request, jsonify, session
from ai_api_manager import get_ai_api_manager
import logging
import json
import os

# Configure the logger
logger = logging.getLogger(__name__)

# Create a Flask Blueprint
api_keys_bp = Blueprint('api_keys', __name__)

# Path to the API keys file
API_KEYS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_keys.json')

def load_api_keys():
    """Loads API keys from a JSON file."""
    try:
        if os.path.exists(API_KEYS_PATH):
            with open(API_KEYS_PATH, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading API keys: {str(e)}")
        return {}

def save_api_keys(api_keys):
    """Saves API keys to a JSON file."""
    try:
        with open(API_KEYS_PATH, 'w') as f:
            json.dump(api_keys, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error saving API keys: {str(e)}")
        return False

@api_keys_bp.route('/api/keys', methods=['GET'])
def get_api_keys():
    """Retrieves the list of configured API keys."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401

    try:
        api_keys = load_api_keys()

        # For security reasons, do not return complete keys
        secured_keys = {}
        for api_name, key in api_keys.items():
            # Mask the key, show only the last 4 characters
            if key and len(key) > 4:
                secured_keys[api_name] = '•' * (len(key) - 4) + key[-4:]
            else:
                secured_keys[api_name] = None

        return jsonify({'api_keys': secured_keys})
    except Exception as e:
        logger.error(f"Error retrieving API keys: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

@api_keys_bp.route('/api/keys/<api_name>', methods=['POST'])
def set_api_key(api_name):
    """Sets or updates the API key for a specific API."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401

    try:
        data = request.json
        api_key = data.get('api_key')

        if not api_key:
            return jsonify({'error': 'Missing API key'}), 400

        # Check if the API exists
        api_manager = get_ai_api_manager()
        available_apis = api_manager.list_available_apis()

        if api_name not in available_apis:
            return jsonify({'error': f'API "{api_name}" not available'}), 404

        # Load existing keys
        api_keys = load_api_keys()

        # Update the key
        api_keys[api_name] = api_key

        # Save the keys
        if save_api_keys(api_keys):
            # Update the API configuration
            api_manager.configure_api(api_name, {'api_key': api_key})

            return jsonify({
                'success': True,
                'message': f'API key for "{api_name}" updated'
            })
        else:
            return jsonify({'error': 'Could not save API key'}), 500
    except Exception as e:
        logger.error(f"Error updating API key for {api_name}: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

@api_keys_bp.route('/api/keys/<api_name>', methods=['DELETE'])
def delete_api_key(api_name):
    """Deletes the API key for a specific API."""
    if not session.get('logged_in'):
        return jsonify({'error': 'Authentication required'}), 401

    try:
        # Check if the API exists
        api_manager = get_ai_api_manager()
        available_apis = api_manager.list_available_apis()

        if api_name not in available_apis:
            return jsonify({'error': f'API "{api_name}" not available'}), 404

        # Load existing keys
        api_keys = load_api_keys()

        # Delete the key if it exists
        if api_name in api_keys:
            del api_keys[api_name]

            # Save the keys
            if save_api_keys(api_keys):
                # Update the API configuration
                api_manager.configure_api(api_name, {'api_key': None})

                return jsonify({
                    'success': True,
                    'message': f'API key for "{api_name}" deleted'
                })
            else:
                return jsonify({'error': 'Could not save changes'}), 500
        else:
            return jsonify({'error': f'No API key found for "{api_name}"'}), 404
    except Exception as e:
        logger.error(f"Error deleting API key for {api_name}: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500
