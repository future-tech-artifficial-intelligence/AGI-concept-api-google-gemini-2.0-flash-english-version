"""
Memory Retrieval Enhancement Module for artificial intelligence API GOOGLE GEMINI 2.0 FLASH.
This enhancement module strengthens the ability of artificial intelligence API GOOGLE GEMINI 2.0 FLASH to remember previous conversations
by increasing explicit memory instructions.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Union

from memory_engine import MemoryEngine
from modules.conversation_memory_enhancer import get_session_id_from_data, get_user_id_from_data

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("memory_retrieval_enhancer")

# Module metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 65,  # Intermediate priority to run after conversation_memory_enhancer
    "description": "Strengthens AI conversation memory instructions",
    "version": "1.0.0",
    "dependencies": ["conversation_memory_enhancer"],
    "hooks": ["process_request"]
}

# Global instance of the memory engine
memory_engine = MemoryEngine()

def enhance_memory_instructions(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds explicit instructions for the AI to use conversation memory.
    
    Args:
        data: The request data
    
    Returns:
        Modified data with strengthened instructions
    """
    # Ensure context exists and is a dictionary
    if 'context' not in data:
        data['context'] = {}
    
    if not isinstance(data['context'], dict):
        current_context = data['context']
        data['context'] = {'previous': current_context}
    
    user_id = get_user_id_from_data(data)
    session_id = get_session_id_from_data(data)
    
    if not user_id or not session_id:
        logger.warning("Unable to add memory instructions without user or session ID")
        return data
    
    # Check if conversations already exist
    recent_conversations = memory_engine.get_recent_conversations(
        user_id=user_id,
        session_id=session_id,
        limit=5,
        include_time_context=True
    )
    
    # Check if long-term memories exist
    long_term_memories = memory_engine.get_long_term_memories(
        user_id=user_id,
        limit=3
    )
    
    # Create specific memory instructions
    memory_instructions = []
    
    # If there are recent conversations, add a strong instruction
    if recent_conversations:
        conversation_count = len(recent_conversations)
        memory_instructions.append(
            f"CRITICAL MEMORY INSTRUCTION: You have {conversation_count} previous conversations with this user. "
            f"Your ability to refer to these conversations is ESSENTIAL. "
            f"ALWAYS use previous conversation elements in your responses."
        )
        
        # Add specific examples from previous conversations
        if conversation_count > 2:
            most_recent = recent_conversations[0]['content']
            memory_instructions.append(
                f"Example of recent conversation: \"{most_recent[:100]}...\". "
                f"Refer to this information when relevant."
            )
    
    # If there are long-term memories, explicitly mention them
    if long_term_memories:
        memory_instructions.append(
            f"You have {len(long_term_memories)} important pieces of information stored about this user. "
            f"Use this information to personalize your responses."
        )
    
    # Add a general instruction about conversation continuity
    memory_instructions.append(
        "IMPORTANT: Ensure natural conversational continuity. "
        "Do not repeat yourself, do not re-introduce yourself, and avoid greetings if you have already spoken with the user."
    )
    
    # Add instructions to the context
    if 'instructions' not in data['context']:
        data['context']['instructions'] = []
    
    if isinstance(data['context']['instructions'], list):
        data['context']['instructions'].extend(memory_instructions)
    else:
        data['context']['instructions'] = memory_instructions
    
    # Add an explicit indicator that memory instructions have been enhanced
    data['context']['memory_instructions_enhanced'] = True
    
    logger.info(f"Memory instructions enhanced for user {user_id}")
    return data

def enrich_system_prompt(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches the system prompt with explicit memory instructions.
    
    Args:
        data: The request data
    
    Returns:
        Modified data
    """
    # Check if data contains a system prompt
    if 'system_prompt' not in data and 'prompt' in data:
        # We don't modify the user prompt, only add instructions
        return data
        
    user_id = get_user_id_from_data(data)
    if not user_id:
        return data
        
    # Retrieve a summary of previous conversations
    memory_context = memory_engine.get_memory_context(
        user_id=user_id,
        max_conversations=3,
        max_long_term=2,
        format_as_text=True
    )
    
    # Create a custom memory instruction
    memory_directive = """
CRITICAL MEMORY INSTRUCTION: You have a persistent memory that allows you to remember previous conversations. 
ALWAYS use this information to ensure the continuity of your interactions. 
NEVER say you don't remember previous conversations.
    """
    
    # Add the instruction to the system prompt if present
    if 'system_prompt' in data and data['system_prompt']:
        data['system_prompt'] = memory_directive + "\n\n" + data['system_prompt']
    
    # Add metadata to indicate that the prompt has been enriched
    if 'metadata' not in data:
        data['metadata'] = {}
        
    data['metadata']['memory_prompt_enriched'] = True
    
    logger.info(f"System prompt enriched with memory instructions for user {user_id}")
    return data

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Main processing function for the module manager.
    
    Args:
        data: The data to process
        hook: The called hook (process_request)
        
    Returns:
        Modified data
    """
    if not isinstance(data, dict):
        logger.warning(f"Data is not a dictionary: {type(data)}")
        return data
    
    try:
        if hook == "process_request":
            # 1. Enhance memory instructions
            data = enhance_memory_instructions(data)
            
            # 2. Enrich the system prompt with memory directives
            data = enrich_system_prompt(data)
            
            return data
        
        return data
    
    except Exception as e:
        logger.error(f"Error in memory_retrieval_enhancer module: {str(e)}", exc_info=True)
        return data
