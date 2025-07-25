"""
**Abstract Interface for Various Artificial Intelligence APIs.**
This module defines a common interface for all supported AI APIs.
"""
import logging
import abc
from typing import Dict, Any, Optional, List, Union

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIApiInterface(abc.ABC):
    """Abstract interface that all AI API implementations must follow."""
    
    @abc.abstractmethod
    def get_response(self, 
                    prompt: str, 
                    image_data: Optional[str] = None,
                    context: Optional[str] = None,
                    emotional_state: Optional[Dict[str, Any]] = None,
                    user_id: int = 1,
                    session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Abstract method to get a response from the AI API.
        
        Args:
            prompt: The request text
            image_data: Base64-encoded image data (optional)
            context: Previous conversation context (optional)
            emotional_state: Current emotional state of the AI (optional)
            user_id: User ID
            session_id: Session ID (optional)
        
        Returns:
            Dictionary containing the response and metadata
        """
        pass
    
    @abc.abstractmethod
    def process_memory_request(self, prompt: str, user_id: int, session_id: str) -> Optional[str]:
        """
        Abstract method to process memory-related requests.
        
        Args:
            prompt: The user's question or instruction
            user_id: User ID
            session_id: Current session ID
            
        Returns:
            Enriched context if the request is memory-related, otherwise None
        """
        pass
    
    @abc.abstractmethod
    def get_conversation_history(self, user_id: int, session_id: str, max_messages: int = 10) -> str:
        """
        Abstract method to retrieve conversation history.
        
        Args:
            user_id: User ID
            session_id: Session ID
            max_messages: Maximum number of messages to include
            
        Returns:
            A summary of the previous conversation
        """
        pass
