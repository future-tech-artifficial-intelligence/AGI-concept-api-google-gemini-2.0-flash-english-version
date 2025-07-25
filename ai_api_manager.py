"""
**Artificial Intelligence API Manager.**
This module allows loading and managing different AI API implementations.
"""
import os
import json
import logging
from typing import Dict, Any, Optional, List, Type

from ai_api_interface import AIApiInterface
from gemini_api_adapter import GeminiAPI
from claude_api_adapter import ClaudeAPI
from custom_llm_adapter import CustomLLMAPI

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the API configuration file
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_api_config.json')

class AIApiManager:
    """
    AI API manager that allows loading and using different implementations.
    """
    # Dictionary of available APIs (api_name -> implementation class)
    _available_apis = {
        'artificial intelligence API GOOGLE GEMINI 2.0 FLASH': GeminiAPI,
        'claude': ClaudeAPI,
        'custom_llm': CustomLLMAPI
    }
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Implements the Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(AIApiManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initializes the API manager."""
        self.default_api_name = 'artificial intelligence API GOOGLE GEMINI 2.0 FLASH'
        self.active_api: Optional[AIApiInterface] = None
        self.config: Dict[str, Dict[str, Any]] = {}
        
        # Load configuration
        self._load_config()
        
        # Initialize default API
        self._set_active_api(self.default_api_name)
        
        logger.info(f"API Manager initialized with default API: {self.default_api_name}")
    
    def _load_config(self):
        """Loads API configuration from the configuration file."""
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                logger.info(f"API configuration loaded from {CONFIG_PATH}")
                
                # Set default API if specified in configuration
                if 'default_api' in self.config:
                    self.default_api_name = self.config['default_api']
            else:
                # Create default configuration
                self.config = {
                    'default_api': 'artificial intelligence API GOOGLE GEMINI 2.0 FLASH',
                    'apis': {
                        'artificial intelligence API GOOGLE GEMINI 2.0 FLASH': {
                            'api_key': None,  # Use default key
                            'api_url': None   # Use default URL
                        },                        'claude': {
                            'api_key': None,  # To be configured by the user
                            'api_url': None   # Use default URL
                        },
                        'custom_llm': {
                            'api_key': None,  # To be configured by the user
                            'api_url': None   # To be configured by the user
                        }
                    }
                }
                self._save_config()
                logger.info("Default configuration created")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def _save_config(self):
        """Saves API configuration to the configuration file."""
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"API configuration saved to {CONFIG_PATH}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def _set_active_api(self, api_name: str) -> bool:
        """
        Sets the active API by its name.
        
        Args:
            api_name: Name of the API to activate
            
        Returns:
            True if the API was successfully activated, False otherwise
        """
        if api_name not in self._available_apis:
            logger.error(f"API '{api_name}' not available")
            return False
        
        try:
            # Retrieve API configuration
            api_config = {}
            if 'apis' in self.config and api_name in self.config['apis']:
                api_config = self.config['apis'][api_name]
            
            # Create an instance of the API
            api_class = self._available_apis[api_name]
            self.active_api = api_class(**api_config)
            logger.info(f"API '{api_name}' successfully activated")
            return True
        except Exception as e:
            logger.error(f"Error activating API '{api_name}': {str(e)}")
            return False
    
    def add_api_implementation(self, api_name: str, api_class: Type[AIApiInterface]) -> bool:
        """
        Adds a new API implementation.
        
        Args:
            api_name: Name of the API to add
            api_class: API implementation class (must inherit from AIApiInterface)
            
        Returns:
            True if the API was successfully added, False otherwise
        """
        if not issubclass(api_class, AIApiInterface):
            logger.error(f"Class '{api_class.__name__}' does not implement the AIApiInterface")
            return False
        
        self._available_apis[api_name] = api_class
        logger.info(f"API '{api_name}' successfully added")
        return True
    
    def set_api(self, api_name: str) -> bool:
        """
        Changes the active API.
        
        Args:
            api_name: Name of the API to activate
            
        Returns:
            True if the API was successfully activated, False otherwise
        """
        result = self._set_active_api(api_name)
        if result:
            # Update default API in configuration
            self.config['default_api'] = api_name
            self._save_config()
        return result
    
    def list_available_apis(self) -> List[str]:
        """
        Returns the list of available APIs.
        
        Returns:
            List of available API names
        """
        return list(self._available_apis.keys())
    
    def get_current_api_name(self) -> str:
        """
        Returns the name of the active API.
        
        Returns:
            Name of the active API
        """
        return self.default_api_name
    
    def configure_api(self, api_name: str, config: Dict[str, Any]) -> bool:
        """
        Configures a specific API.
        
        Args:
            api_name: Name of the API to configure
            config: API configuration (API key, URL, etc.)
            
        Returns:
            True if the configuration was successfully applied, False otherwise
        """
        if api_name not in self._available_apis:
            logger.error(f"API '{api_name}' not available")
            return False
        
        try:
            # Update configuration
            if 'apis' not in self.config:
                self.config['apis'] = {}
            self.config['apis'][api_name] = config
            self._save_config()
            
            # If it's the active API, reinitialize it with the new configuration
            if self.default_api_name == api_name:
                self._set_active_api(api_name)
                
            logger.info(f"Configuration for API '{api_name}' updated")
            return True
        except Exception as e:
            logger.error(f"Error configuring API '{api_name}': {str(e)}")
            return False
    
    def get_api_config(self, api_name: str) -> Dict[str, Any]:
        """
        Retrieves the configuration for a specific API.
        
        Args:
            api_name: Name of the API whose configuration to retrieve
            
        Returns:
            API configuration or an empty dictionary if the API does not exist
        """
        if api_name not in self._available_apis:
            logger.warning(f"API '{api_name}' not available, cannot retrieve its configuration")
            return {}
        
        try:
            # Retrieve configuration
            if 'apis' in self.config and api_name in self.config['apis']:
                return self.config['apis'][api_name]
            else:
                return {}
        except Exception as e:
            logger.error(f"Error retrieving configuration for API '{api_name}': {str(e)}")
            return {}
    
    def get_api_instance(self) -> Optional[AIApiInterface]:
        """
        Returns the instance of the active API.
        
        Returns:
            Instance of the active API or None if no API is active
        """
        return self.active_api
    
    # Delegation methods for ease of use
    def get_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Delegates the call to the get_response method of the active API.
        
        Args:
            prompt: The request text
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            Dictionary containing the response and metadata
        """
        if self.active_api is None:
            logger.error("No active API")
            return {
                'response': "Error: No artificial intelligence API GOOGLE GEMINI 2.0 FLASH is active.",
                'status': 'error',
                'error': 'No active API'
            }
        
        return self.active_api.get_response(prompt, **kwargs)
    
    def process_memory_request(self, prompt: str, user_id: int, session_id: str) -> Optional[str]:
        """
        Delegates the call to the process_memory_request method of the active API.
        
        Args:
            prompt: The user's question or instruction
            user_id: User ID
            session_id: Current session ID
            
        Returns:
            Enriched context if the request is memory-related, otherwise None
        """
        if self.active_api is None:
            logger.error("No active API")
            return None
        
        return self.active_api.process_memory_request(prompt, user_id, session_id)
    
    def get_conversation_history(self, user_id: int, session_id: str, max_messages: int = 10) -> str:
        """
        Delegates the call to the get_conversation_history method of the active API.
        
        Args:
            user_id: User ID
            session_id: Session ID
            max_messages: Maximum number of messages to include
            
        Returns:
            A summary of the previous conversation
        """
        if self.active_api is None:
            logger.error("No active API")
            return ""
        
        return self.active_api.get_conversation_history(user_id, session_id, max_messages)


# Global function to get the API manager instance
def get_ai_api_manager() -> AIApiManager:
    """
    Returns the singleton instance of the API manager.
    
    Returns:
        API manager instance
    """
    return AIApiManager()
