"""
**Conversation Continuity Diagnostic Script.**
This script tests the ability of artificial intelligence API GOOGLE GEMINI 2.0 FLASH to maintain conversation continuity and balance its emotional responses.
"""

import sys
import logging
import os
import time
from memory_engine import MemoryEngine
from modules.conversation_context_manager import is_new_conversation, moderate_emotional_expressions

# Logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("conversation_continuity_diagnosis")

def print_diagnostic_header(title):
    """
    Displays a formatted diagnostic header.
    """
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def run_continuity_diagnostic():
    """
    Runs a conversation continuity diagnostic.
    """
    print_diagnostic_header("CONVERSATION CONTINUITY DIAGNOSTIC")
    print("This script checks the AI's ability to maintain conversation continuity")
    print("and balance its emotional responses.")
    
    # Check if the MemoryEngine module is correctly configured
    memory_engine = MemoryEngine()
    
    try:
        # Test new vs. ongoing conversation detection
        print("\n1. Conversation State Detection Test")
        
        # Example data for a new conversation
        new_conversation_data = {
            'user_id': 1,
            'session_id': f"test_session_{int(time.time())}",  # Unique session ID
            'text': "Hello, how are you?"
        }
        
        is_new = is_new_conversation(new_conversation_data)
        print(f"- New conversation detected: {is_new} (expected: True)")
        
        # Example of emotional expression moderation
        print("\n2. Emotional Expression Moderation Test")
        
        test_responses = [
            "Hello! I am truly delighted to meet you. How can I help you today?",
            "I feel a lot of enthusiasm for this question! It's a fascinating topic that greatly interests me!",
            "I am incredibly excited to be able to work with you on this project!"
        ]
        
        print("Responses before and after moderation:")
        for i, response in enumerate(test_responses):
            moderated = moderate_emotional_expressions(response, is_new=False)
            print(f"\nOriginal [{i+1}]: {response}")
            print(f"Moderated  [{i+1}]: {moderated}")
        
        print("\nDiagnostic completed successfully!")
        print("\nFor more thorough verification, please test the full application")
        print("and observe the AI's behavior in real conversations.")
    
    except Exception as e:
        logger.error(f"Error during diagnostic: {str(e)}", exc_info=True)
        print(f"\nERROR: {str(e)}")
        print("The diagnostic could not be completed.")

if __name__ == "__main__":
    run_continuity_diagnostic()
