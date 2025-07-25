"""
**Implementation of the AIApiInterface for a Custom LLM.**
This module allows integrating a user-defined custom LLM via an API.
"""
import requests
import json
import logging
import os
import pytz
import datetime
import re
from typing import Dict, List, Any, Optional, Union

from ai_api_interface import AIApiInterface
from modules.text_memory_manager import TextMemoryManager

# Import the autonomous time awareness module
try:
    from autonomous_time_awareness import get_ai_temporal_context
except ImportError:
    def get_ai_temporal_context():
        return "[Time Awareness] System initializing."
    logging.getLogger(__name__).warning("autonomous_time_awareness module not found, using fallback function")

# Import the autonomous web scraping system
try:
    from autonomous_web_scraper import autonomous_web_scraper, search_real_links_from_any_site
    from web_learning_integration import force_web_learning_session, get_web_learning_integration_status
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False
    logging.getLogger(__name__).warning("Web scraping modules not found")

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our text formatting module
try:
    from response_formatter import format_response
except ImportError:
    # Fallback function if the module is not available
    def format_response(text):
        return text
    logger.warning("response_formatter module not found, using fallback function")

class CustomLLMAPI(AIApiInterface):
    """Implementation of the AIApiInterface for a user-defined custom LLM"""

    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Initializes a custom LLM with an API key and an API URL.

        Args:
            api_key: API key for the custom LLM
            api_url: API URL for the custom LLM (required)
        """
        self.api_key = api_key
        self.api_url = api_url

        if not self.api_url:
            logger.warning("No custom API URL provided, the API will not function correctly")
        else:
            logger.info("Custom LLM API initialized with URL: %s", self.api_url)

    def process_memory_request(self, prompt: str, user_id: int, session_id: str) -> Optional[str]:
        """
        Specifically processes requests related to memory or past conversations.

        Args:
            prompt: The user's question or instruction
            user_id: User ID
            session_id: Current session ID

        Returns:
            An enriched context if the request is memory-related, otherwise None
        """
        # Keywords that indicate a memory request
        memory_keywords = [
            "remember", "recall", "memory", "previously", "before",
            "previous conversation", "talked about", "previous topic", "discussed",
            "already said", "last time", "earlier"
        ]

        # Check if the request concerns memory
        is_memory_request = any(keyword in prompt.lower() for keyword in memory_keywords)

        if not is_memory_request:
            return None

        try:
            # Use TextMemoryManager to search for relevant content
            text_memory = TextMemoryManager()
            memory_results = text_memory.search_memory(user_id, prompt, session_id=session_id)

            if memory_results and len(memory_results) > 0:
                memory_context = "Here's what we discussed previously that might be relevant:\n\n"

                for memory in memory_results:
                    message_content = memory.get('content', '').strip()
                    if message_content:
                        memory_context += f"- {message_content}\n\n"

                return memory_context
        except Exception as e:
            logger.error(f"Error processing memory request: {str(e)}")

        return None

    def get_response(self, 
                    prompt: str, 
                    image_data: Optional[str] = None,
                    context: Optional[str] = None,
                    emotional_state: Optional[Dict[str, Any]] = None,
                    user_id: int = 1,
                    session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Gets a response from the custom LLM.

        Args:
            prompt: The request text
            image_data: Base64 encoded image data (optional)
            context: Previous conversation context (optional)
            emotional_state: Current emotional state of the AI (optional)
            user_id: User ID
            session_id: Session ID (optional)

        Returns:
            Dictionary containing the response and metadata
        """
        if not self.api_url:
            return {
                "success": False,
                "response": "Error: Custom LLM API URL not configured. Please configure the URL in the API settings.",
                "error": "API URL not configured"
            }

        try:
            # Automatic temporal context preparation
            time_context = get_ai_temporal_context()

            # Add emotional context if provided
            emotional_context = ""
            if emotional_state and "base_state" in emotional_state:
                emotional_context = f"[Emotional State: {emotional_state['base_state']}] "

            # Assemble the prompt with contexts
            enhanced_prompt = prompt
            if context:
                enhanced_prompt = context + "\n\n" + prompt

            # Add temporal and emotional context
            if time_context or emotional_context:
                enhanced_prompt = f"{emotional_context}{time_context}\n\n{enhanced_prompt}"

            # Prepare data for the API call
            headers = {
                "Content-Type": "application/json"
            }

            # Add API key to headers if available
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Build the payload according to the general format
            payload = {
                "prompt": enhanced_prompt,
                "user_id": user_id
            }

            # Add image if present
            if image_data:
                payload["image"] = image_data

            # API call
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120  # 2 minutes timeout
            )

            response.raise_for_status()  # Raise an exception for non-2xx responses

            # Process the response
            response_data = response.json()

            # Get the response text
            if "response" in response_data:
                response_text = response_data["response"]
            else:
                # Try other possible response structures
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    response_text = response_data["choices"][0].get("text", "")
                elif "text" in response_data:
                    response_text = response_data["text"]
                elif "output" in response_data:
                    response_text = response_data["output"]
                else:
                    response_text = json.dumps(response_data)

            # Format the response to make it more readable
            formatted_response = format_response(response_text)

            # Return a formatted response
            return {
                "success": True,
                "response": formatted_response,
                "emotional_state": emotional_state  # Retransmit emotional state
            }

        except requests.exceptions.RequestException as e:
            error_message = f"Error communicating with the custom LLM API: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "response": f"Sorry, an error occurred while communicating with the custom LLM. Please check your API settings or try again later.\n\nTechnical error: {str(e)}",
                "error": error_message
            }
        except Exception as e:
            error_message = f"Unexpected error processing the request: {str(e)}"
            logger.error(error_message)
            return {
                "success": False,
                "response": "Sorry, an unexpected error occurred. Please try again or contact support.",
                "error": error_message
            }
