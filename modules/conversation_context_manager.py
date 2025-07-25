"""
Python language module for enhanced conversational context management for Gemini.
This module improves conversation continuity and balances emotional expressions for artificial intelligence GOOGLE GEMINI 2.0 FLASH
"""

import logging
import re
import random
from typing import Dict, Any, List, Optional

from memory_engine import MemoryEngine

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation_context_manager")

# Module metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 70,  # High priority to run after context retrieval
    "description": "Balances emotional expressions and enhances conversation continuity",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Global memory engine instance
memory_engine = MemoryEngine()

# Patterns to detect greetings and conversation resumption expressions
GREETING_PATTERNS = [
    r"(?i)^(bonjour|salut|hello|coucou|hey|bonsoir)",
    r"(?i)^(je suis (ravi|content|heureux) de (vous|te) (voir|rencontrer|parler))",
    r"(?i)^(c'est un plaisir de vous (revoir|retrouver|parler))",
    r"(?i)^(comment (vas-tu|allez-vous|ça va))",
]

# Patterns to detect repetitive greeting expressions
REPETITIVE_GREETING_PATTERNS = [
    r"(?i)(Salut\s*!\s*Je suis vraiment ravi de te revoir)",
    r"(?i)(J'apprécie beaucoup ton retour)",
    r"(?i)(je suis très content que tu aies pensé à revenir me parler)",
    r"(?i)(Je suis (vraiment|très) (ravi|content|heureux) de te (revoir|retrouver))",
    r"(?i)(C'est (formidable|génial|super) de te (revoir|retrouver))",
]

# Patterns to detect a topic change
TOPIC_CHANGE_PATTERNS = [
    r"(?i)^(maintenant|sinon|au fait|dis-moi|parlons de|j'aimerais|peux-tu)",
    r"(?i)^(une autre question|autre chose|changeons de sujet)",
    r"(?i)^(passons à|abordons|intéressons-nous à)",
    r"(?i)(différent|nouveau sujet|autre sujet)",
]

# Patterns to detect excessive emotional expressions
EXCESSIVE_EMOTION_PATTERNS = [
    r"(?i)(je suis (vraiment|extrêmement|incroyablement) (content|heureux|ravi|enthousiaste|excité))",
    r"(?i)(je ressens (beaucoup|énormément|tellement) d'(enthousiasme|excitation|joie))",
    r"(?i)(je ne peux pas (cacher|contenir) mon (enthousiasme|excitation|ravissement))",
    r"(?i)(je suis (totalement|complètement) (fasciné|émerveillé|captivé))",
]

# Specific patterns for image analysis responses
IMAGE_ANALYSIS_PATTERNS = [
    r"(?i)^(Absolument\s?!?\s?Je suis ravi de pouvoir t'aider\.?\s?Oui,?\s?je ressens des émotions en analysant cette image\s?Analyse de l'image)",
    r"(?i)^(Je suis (ravi|heureux|content) de pouvoir analyser cette image pour toi\.?\s?Analyse de l'image)",
    r"(?i)^(Analyse de l'image\s?:?\s?)"
]

def get_user_id_from_data(data: Dict[str, Any]) -> Optional[int]:
    """
    Extracts the user ID from the data.
    """
    # Try different possible keys
    for key in ['user_id', 'userId', 'user']:
        if key in data and data[key]:
            try:
                return int(data[key])
            except (ValueError, TypeError):
                pass
    
    # Look in the session if available
    if 'session' in data and isinstance(data['session'], dict):
        for key in ['user_id', 'userId', 'user']:
            if key in data['session'] and data['session'][key]:
                try:
                    return int(data['session'][key])
                except (ValueError, TypeError):
                    pass
    
    return None

def get_session_id_from_data(data: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the session ID from the data.
    """
    # Try different possible keys
    for key in ['session_id', 'sessionId', 'session']:
        if key in data and isinstance(data[key], (str, int)):
            return str(data[key])
    
    # Look in the session if available
    if 'session' in data and isinstance(data['session'], dict):
        for key in ['id', 'session_id', 'sessionId']:
            if key in data['session'] and data['session'][key]:
                return str(data['session'][key])
    
    return None

def is_new_conversation(data: Dict[str, Any]) -> bool:
    """
    Determines if the conversation is new or ongoing.
    
    Args:
        data: The request data
    
    Returns:
        True if it's a new conversation, False otherwise
    """
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    
    if not user_id or not session_id:
        return True  # By default, consider it a new conversation
    
    try:
        # Check if there are recent messages for this session
        recent_conversations = memory_engine.get_recent_conversations(
            user_id=user_id, 
            session_id=session_id, 
            limit=5
        )
        
        # If no recent conversations found, it's a new conversation
        return len(recent_conversations) == 0
    except Exception as e:
        logger.error(f"Error checking conversation state: {str(e)}")
        return True  # In case of error, consider it a new conversation

def detect_topic_change(data: Dict[str, Any]) -> bool:
    """
    Detects if the user changes the topic in the conversation.
    
    Args:
        data: The request data
    
    Returns:
        True if it's a topic change, False otherwise
    """
    # Extract text from the request
    text = ""
    if 'text' in data:
        text = data['text']
    elif 'message' in data:
        text = data['message']
    
    if not text:
        return False
    
    # Check for topic change patterns
    for pattern in TOPIC_CHANGE_PATTERNS:
        if re.search(pattern, text):
            return True
    
    return False

def detect_image_analysis(response: str) -> bool:
    """
    Detects if the response is an image analysis.
    
    Args:
        response: The generated response
    
    Returns:
        True if it's an image analysis, False otherwise
    """
    # Generic keywords for image analyses
    image_keywords = [
        r"(?i)(this image shows)",
        r"(?i)(in this image,)",
        r"(?i)(the image presents)",
        r"(?i)(one can see in this image)",
        r"(?i)(I see an image that)",
        r"(?i)(the photo shows)",
        r"(?i)(we observe in this image)",
        r"(?i)(it is an image (of|that))",
        r"(?i)(this photograph (shows|presents|contains))",
        r"(?i)(the illustration (shows|represents))",
        r"(?i)(on this (capture|shot))",
        r"(?i)(this visual (shows|presents))",
    ]
    
    # Keywords by image categories
    category_keywords = {
        # Astronomical images
        "astronomy": [
            r"(?i)(constellation[s]? (of|the))",
            r"(?i)(map (of the|celestial|sky))",
            r"(?i)(night sky)",
            r"(?i)(star[s]? (visible|bright|named))",
            r"(?i)(position (of the|of the) (moon|planet|star))",
            r"(?i)(trajectory (of|of the))",
        ],
        # Artworks and creative images
        "art": [
            r"(?i)(painting|artwork)",
            r"(?i)(artistic|pictorial) style)",
            r"(?i)(artistic|visual) composition)",
            r"(?i)(perspective|background|foreground)",
            r"(?i)(colors|hues|shades|palette)",
        ],
        # Natural scenes and landscapes
        "nature": [
            r"(?i)(landscape (of|mountainous|marine|rural|urban))",
            r"(?i)(panoramic|aerial) view)",
            r"(?i)(natural environment)",
            r"(?i)(flora|fauna|vegetation)",
            r"(?i)(forest|mountain|ocean|river|lake)",
        ],
        # Diagrams and schematics
        "technical": [
            r"(?i)(diagram|chart|graph)",
            r"(?i)(technical|schematic) representation)",
            r"(?i)(technical illustration)",
            r"(?i)(structure|components|elements)",
            r"(?i)(legend|annotation|label)",
        ]
    }
    
    # Check generic keywords
    for pattern in image_keywords:
        if re.search(pattern, response):
            return True
    
    # Check keywords by category
    for category, patterns in category_keywords.items():
        for pattern in patterns:
            if re.search(pattern, response):
                return True
    
    # Or if the response is already identified as starting with an image analysis pattern
    for pattern in IMAGE_ANALYSIS_PATTERNS:
        if re.search(pattern, response):
            return True
    
    return False

def moderate_emotional_expressions(response: str, is_new_conversation: bool, is_topic_change: bool = False) -> str:
    """
    Moderates excessive emotional expressions in the response.
    
    Args:
        response: The generated response
        is_new_conversation: Indicator if it's a new conversation
        is_topic_change: Indicator if it's a topic change
    
    Returns:
        The moderated response
    """
    # Check if it's an image analysis
    is_image_analysis = detect_image_analysis(response)
    
    # If it's an image analysis, remove excessive introductory phrases
    if is_image_analysis:
        for pattern in IMAGE_ANALYSIS_PATTERNS:
            # Replace excessive phrases with a more neutral beginning
            if re.search(pattern, response):
                image_intro_phrases = [
                    "Image analysis: ",
                    "Here is the analysis of this image: ",
                    "Analysis: ",
                    ""  # Empty option to start directly with the description
                ]
                replacement = random.choice(image_intro_phrases)
                response = re.sub(pattern, replacement, response, count=1)
    
    # Remove repetitive greeting expressions
    for pattern in REPETITIVE_GREETING_PATTERNS:
        if re.search(pattern, response):
            if is_topic_change:
                # For a topic change, use more appropriate transitions
                topic_transitions = [
                    "Interesting! ",
                    "Okay, ",
                    "Very good, ",
                    "Perfect, ",
                    ""  # Empty option to start directly
                ]
                replacement = random.choice(topic_transitions)
            elif not is_new_conversation:
                # For an ongoing conversation, use continuity phrases
                continuity_phrases = [
                    "Of course, ",
                    "Indeed, ",
                    "I see, ",
                    "Okay, ",
                    ""  # Empty option
                ]
                replacement = random.choice(continuity_phrases)
            else:
                # For a truly new conversation, vary greetings
                varied_greetings = [
                    "Hello! ",
                    "Hi! ",
                    "Good evening! ",
                    "Nice to meet you! ",
                    ""  # Empty option
                ]
                replacement = random.choice(varied_greetings)
            
            response = re.sub(pattern, replacement, response, count=1)
    
    # Moderate generic greetings if it's not a new conversation
    if not is_new_conversation and not is_topic_change:
        for pattern in GREETING_PATTERNS:
            # Replace greetings with a continuity phrase
            if re.search(pattern, response):
                continuity_phrases = [
                    "To continue our discussion, ",
                    "To return to our topic, ",
                    "Continuing our exchange, ",
                    "To pick up where we left off, ",
                    ""  # Empty option to simply remove the greeting
                ]
                replacement = random.choice(continuity_phrases)
                response = re.sub(pattern, replacement, response, count=1)
    
    # Moderate excessive emotional expressions
    emotion_count = 0
    for pattern in EXCESSIVE_EMOTION_PATTERNS:
        matches = re.findall(pattern, response)
        emotion_count += len(matches)
        
        # Limit the number of emotional expressions to 1 per response
        if emotion_count > 1:
            response = re.sub(pattern, "", response)
    
    return response

def process_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the request data.
    
    Args:
        data: The request data
    
    Returns:
        The modified data
    """
    # Check if it's a new conversation
    is_new = is_new_conversation(data)
    
    # Check if it's a topic change
    is_topic_change = detect_topic_change(data)
    
    # Store the information to be used in the response
    if 'context' not in data:
        data['context'] = {}
    
    if isinstance(data['context'], dict):
        data['context']['is_new_conversation'] = is_new
        data['context']['is_topic_change'] = is_topic_change
    
    return data

def process_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes the response data.
    
    Args:
        data: The response data
    
    Returns:
        The modified data
    """
    # Retrieve context indicators
    is_new = True  # Default
    is_topic_change = False  # Default
    
    if 'context' in data and isinstance(data['context'], dict):
        is_new = data['context'].get('is_new_conversation', True)
        is_topic_change = data['context'].get('is_topic_change', False)
    
    # Extract the response
    response = None
    if 'response' in data:
        response = data['response']
    elif 'content' in data:
        response = data['content']
    
    # Moderate the response if it exists
    if response:
        moderated_response = moderate_emotional_expressions(response, is_new, is_topic_change)
        
        # Update the response in the data
        if 'response' in data:
            data['response'] = moderated_response
        elif 'content' in data:
            data['content'] = moderated_response
    
    return data

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Main processing function for the module manager.
    
    Args:
        data: The data to process
        hook: The called hook (process_request or process_response)
        
    Returns:
        The modified data
    """
    if not isinstance(data, dict):
        logger.warning(f"Data is not a dictionary: {type(data)}")
        return data
    
    try:
        if hook == "process_request":
            return process_request(data)
            
        elif hook == "process_response":
            return process_response(data)
        
        return data
    
    except Exception as e:
        logger.error(f"Error in conversation_context_manager module: {str(e)}", exc_info=True)
        return data
