"""
**AI APIs Configuration Module**
Provides centralized access to the configurations of various AI APIs.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

def load_config() -> Dict:
    """Loads the configuration from the JSON file"""
    config_file = Path("ai_api_config.json")
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Default configuration
        default_config = {
            "default_api": "gemini",
            "apis": {
                "gemini": {
                    "api_key": None,
                    "api_url": None
                },
                "claude": {
                    "api_key": None,
                    "api_url": None
                }
            }
        }
        # Save default configuration
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        return default_config

def get_api_config(api_name: Optional[str] = None) -> Dict:
    """
    Gets the configuration for a specific API
    
    Args:
        api_name: Name of the API (gemini, claude, etc.) or None for the full configuration
    
    Returns:
        Dict: API configuration
    """
    config = load_config()
    
    if api_name is None:
        # Return the full configuration with environment keys
        result_config = config.copy()
        
        # Check environment variables for API keys
        for api in result_config['apis']:
            env_var = f"{api.upper()}_API_KEY"
            env_key = os.getenv(env_var)
            if env_key:
                result_config['apis'][api]['api_key'] = env_key
        
        # Add the main artificial intelligence API GOOGLE GEMINI 2.0 FLASH key if available
        if os.getenv('GEMINI_API_KEY'):
            result_config['gemini_api_key'] = os.getenv('GEMINI_API_KEY')
        
        return result_config
    else:
        if api_name in config['apis']:
            api_config = config['apis'][api_name].copy()
            
            # Check environment variable
            env_var = f"{api_name.upper()}_API_KEY"
            env_key = os.getenv(env_var)
            if env_key:
                api_config['api_key'] = env_key
            
            return api_config
        else:
            return {}

def update_api_config(api_name: str, api_key: str, api_url: Optional[str] = None):
    """
    Updates the configuration of an API
    
    Args:
        api_name: Name of the API
        api_key: API key
        api_url: API URL (optional)
    """
    config = load_config()
    
    if api_name not in config['apis']:
        config['apis'][api_name] = {}
    
    config['apis'][api_name]['api_key'] = api_key
    if api_url:
        config['apis'][api_name]['api_url'] = api_url
    
    config_file = Path("ai_api_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def get_default_api() -> str:
    """Gets the name of the default API"""
    config = load_config()
    return config.get('default_api', 'gemini')

def set_default_api(api_name: str):
    """Sets the default API"""
    config = load_config()
    config['default_api'] = api_name
    
    config_file = Path("ai_api_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def has_api_key(api_name: str) -> bool:
    """Checks if an API key is available"""
    config = get_api_config(api_name)
    return bool(config.get('api_key'))

def get_available_apis() -> list:
    """Returns the list of APIs with available keys"""
    config = get_api_config()
    available = []
    
    for api_name in config['apis']:
        if has_api_key(api_name):
            available.append(api_name)
    
    return available

# Compatibility functions for older scripts
def get_gemini_api_key() -> Optional[str]:
    """Gets the artificial intelligence API GOOGLE GEMINI 2.0 FLASH API key"""
    config = get_api_config('gemini')
    return config.get('api_key') or os.getenv('GEMINI_API_KEY')

def get_claude_api_key() -> Optional[str]:
    """Gets the Claude API key"""
    config = get_api_config('claude')
    return config.get('api_key') or os.getenv('CLAUDE_API_KEY')
