"""
Conversation Memory Enhancement Module for Artificial Intelligence API GOOGLE GEMINI 2.0 FLASH.
This module enhances the ability of Artificial Intelligence API GOOGLE GEMINI 2.0 FLASH to remember previous conversations
by using an advanced temporal memory system.
"""

import logging
import datetime
import json
from typing import Dict, Any, List, Optional

from time_engine import should_remember_conversation, timestamp_to_readable_time_diff
from memory_engine import MemoryEngine

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("conversation_memory_enhancer")

# Module metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 60,  # High priority to be executed among the first
    "description": "Enhances AI conversation memory",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Global instance of the memory engine
memory_engine = MemoryEngine()

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
    
    logger.warning("Unable to determine user ID in data")
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
    
    logger.warning("Unable to determine session ID in data")
    return None

def extract_message_content(data: Dict[str, Any]) -> Optional[str]:
    """
    Extracts the message content from the data.
    """
    # For a user request
    if 'text' in data:
        return data['text']
    if 'message' in data:
        if isinstance(data['message'], str):
            return data['message']
        elif isinstance(data['message'], dict) and 'content' in data['message']:
            return data['message']['content']
    
    # For a Gemini response
    if 'response' in data:
        return data['response']
    if 'content' in data:
        return data['content']
    
    logger.warning("Unable to extract message content from data")
    return None

def add_conversation_context(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds context from previous conversations to the request data.
    """
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    
    if not user_id:
        logger.warning("Cannot add conversation context without user ID")
        return data
    
    # Check if there are recent messages in this session
    recent_conversations = memory_engine.get_recent_conversations(
        user_id=user_id,
        session_id=session_id,
        limit=3,
        include_time_context=True
    )
    
    # Determine if it's an ongoing or new conversation
    is_ongoing_conversation = len(recent_conversations) > 0
    
    # Generate memory context
    memory_context = memory_engine.get_memory_context(
        user_id=user_id,
        session_id=session_id,
        max_conversations=5,
        max_long_term=3,
        format_as_text=True
    )
    
    # Add context to the request
    if 'context' not in data:
        data['context'] = {}
    
    if isinstance(data['context'], dict):
        data['context']['conversation_memory'] = memory_context
        data['context']['is_ongoing_conversation'] = is_ongoing_conversation
        
        # If it's an ongoing conversation, add an explicit instruction
        # to prevent the AI from starting with greetings again
        if is_ongoing_conversation:
            last_conversation_time = recent_conversations[0].get('time_ago', 'recently')
            
            if 'instructions' not in data['context']:
                data['context']['instructions'] = []
            
            if isinstance(data['context']['instructions'], list):
                data['context']['instructions'].append(
                    f"This conversation is ongoing. You have already interacted with this user {last_conversation_time}. "
                    f"Avoid introducing yourself again or repeating greetings like 'I'm glad to meet you'. "
                    f"Simply continue the conversation naturally."
                )
    else:
        # If context is a string or other type, convert it to a dictionary
        current_context = data['context']
        data['context'] = {
            'previous': current_context,
            'conversation_memory': memory_context,
            'is_ongoing_conversation': is_ongoing_conversation
        }
    
    logger.info(f"Conversation context added for user {user_id}")
    return data

def store_conversation_entry(data: Dict[str, Any], is_response: bool = False) -> Dict[str, Any]:
    """
    Stores the conversation entry in memory.
    """
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    content = extract_message_content(data)
    
    if not user_id or not session_id or not content:
        logger.warning("Insufficient data to store conversation entry")
        return data
    
    # Retrieve user's timezone if available
    user_timezone = None
    if 'user_timezone' in data:
        user_timezone = data['user_timezone']
    elif 'context' in data and isinstance(data['context'], dict) and 'user_timezone' in data['context']:
        user_timezone = data['context']['user_timezone']
    
    # Prepare metadata
    metadata = {
        'is_response': is_response,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    # Add additional metadata if available
    if 'emotional_state' in data:
        metadata['emotional_state'] = data['emotional_state']
    
    # Determine memory level (short, medium, or long term)
    memory_level = "MEDIUM_TERM"  # By default, remember for approximately 1 hour
    
    # If marked as important or an emotional response, store for longer
    if is_response and 'emotional_state' in data:
        emotional_state = data['emotional_state']
        if isinstance(emotional_state, dict) and 'intensity' in emotional_state:
            intensity = emotional_state.get('intensity', 0)
            if intensity > 7:  # High emotional intensity
                memory_level = "LONG_TERM"
    
    # Store in memory with user timezone
    memory_id = memory_engine.store_conversation(
        session_id=session_id,
        user_id=user_id,
        content=content,
        memory_level=memory_level,
        metadata=metadata,
        user_timezone=user_timezone
    )
    
    logger.info(f"Conversation stored with ID {memory_id}, level {memory_level}")
    
    # Add memory ID to data for reference
    if 'memory' not in data:
        data['memory'] = {}
    
    data['memory']['memory_id'] = memory_id
    data['memory']['memory_level'] = memory_level
    
    return data

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Main processing function for the module handler.
    
    Args:
        data: The data to process
        hook: The hook called (process_request or process_response)
        
    Returns:
        The modified data
    """
    if not isinstance(data, dict):
        logger.warning(f"Data is not a dictionary: {type(data)}")
        return data
    
    try:
        if hook == "process_request":
            # 1. Store the user request in memory
            data = store_conversation_entry(data, is_response=False)
            
            # 2. Add context from previous conversations
            data = add_conversation_context(data)
            
            return data
            
        elif hook == "process_response":
            # Store Gemini's response in memory
            data = store_conversation_entry(data, is_response=True)
            
            return data
        
        return data
    
    except Exception as e:
        logger.error(f"Error in conversation_memory_enhancer module: {str(e)}", exc_info=True)
        return data
