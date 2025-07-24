"""
Module de Configuration des APIs AI
Fournit l'accès centralisé aux configurations des différentes APIs AI
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

def load_config() -> Dict:
    """Charge la configuration depuis le fichier JSON"""
    config_file = Path("ai_api_config.json")
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Configuration par défaut
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
        # Sauvegarder la configuration par défaut
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=4)
        return default_config

def get_api_config(api_name: Optional[str] = None) -> Dict:
    """
    Obtient la configuration d'une API spécifique
    
    Args:
        api_name: Nom de l'API (gemini, claude, etc.) ou None pour la configuration complète
    
    Returns:
        Dict: Configuration de l'API
    """
    config = load_config()
    
    if api_name is None:
        # Retourner la configuration complète avec les clés d'environnement
        result_config = config.copy()
        
        # Vérifier les variables d'environnement pour les clés API
        for api in result_config['apis']:
            env_var = f"{api.upper()}_API_KEY"
            env_key = os.getenv(env_var)
            if env_key:
                result_config['apis'][api]['api_key'] = env_key
        
        # Ajouter la clé Gemini principale si disponible
        if os.getenv('GEMINI_API_KEY'):
            result_config['gemini_api_key'] = os.getenv('GEMINI_API_KEY')
        
        return result_config
    else:
        if api_name in config['apis']:
            api_config = config['apis'][api_name].copy()
            
            # Vérifier la variable d'environnement
            env_var = f"{api_name.upper()}_API_KEY"
            env_key = os.getenv(env_var)
            if env_key:
                api_config['api_key'] = env_key
            
            return api_config
        else:
            return {}

def update_api_config(api_name: str, api_key: str, api_url: Optional[str] = None):
    """
    Met à jour la configuration d'une API
    
    Args:
        api_name: Nom de l'API
        api_key: Clé API
        api_url: URL de l'API (optionnel)
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
    """Obtient le nom de l'API par défaut"""
    config = load_config()
    return config.get('default_api', 'gemini')

def set_default_api(api_name: str):
    """Définit l'API par défaut"""
    config = load_config()
    config['default_api'] = api_name
    
    config_file = Path("ai_api_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

def has_api_key(api_name: str) -> bool:
    """Vérifie si une clé API est disponible"""
    config = get_api_config(api_name)
    return bool(config.get('api_key'))

def get_available_apis() -> list:
    """Retourne la liste des APIs avec des clés disponibles"""
    config = get_api_config()
    available = []
    
    for api_name in config['apis']:
        if has_api_key(api_name):
            available.append(api_name)
    
    return available

# Fonctions de compatibilité pour les anciens scripts
def get_gemini_api_key() -> Optional[str]:
    """Obtient la clé API Gemini"""
    config = get_api_config('gemini')
    return config.get('api_key') or os.getenv('GEMINI_API_KEY')

def get_claude_api_key() -> Optional[str]:
    """Obtient la clé API Claude"""
    config = get_api_config('claude')
    return config.get('api_key') or os.getenv('CLAUDE_API_KEY')
