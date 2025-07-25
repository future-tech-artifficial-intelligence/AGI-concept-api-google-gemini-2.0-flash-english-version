#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the memory system of artificial intelligence API GOOGLE GEMINI 2.0 FLASH
This script tests access to previous conversation files and the process_memory_request function.
"""

import os
import logging
import uuid
import sys
from datetime import datetime

# Configure logger to display in console
logger = logging.getLogger('memory_test')
logger.setLevel(logging.INFO)

# Create a formatter with colors for better readability
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add a console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Import necessary modules
from modules.text_memory_manager import TextMemoryManager
from gemini_api import process_memory_request, get_conversation_history

def create_test_conversation(user_id=1):
    """
    Creates a test conversation to verify the memory system.
    
    Args:
        user_id: User ID for the test
        
    Returns:
        ID of the created session
    """
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    logger.info(f"Creating a test conversation with session_id: {session_id}")
    
    # Messages for the test conversation
    test_messages = [
        ("user", "Hello, I'd like to discuss artificial intelligence."),
        ("assistant", "Hello! I'd be happy to discuss artificial intelligence with you. What would you like to know in particular?"),
        ("user", "Can you explain what deep learning is?"),
        ("assistant", "Deep learning is a branch of machine learning that uses multi-layered neural networks to learn from data. These networks can automatically extract high-level features from raw data, making them particularly effective for tasks such as computer vision and natural language processing."),
        ("user", "Thanks for the explanation. What are the main deep learning frameworks?"),
        ("assistant", "The main deep learning frameworks include TensorFlow (developed by Google), PyTorch (developed by Facebook/Meta), Keras (now integrated into TensorFlow), JAX (also from Google), and MXNet (used by Amazon). TensorFlow and PyTorch are currently the most popular, each with its own strengths. TensorFlow is often used in production, while PyTorch is highly regarded for research due to its flexibility.")
    ]
    
    # Save messages to the conversation file
    for message_type, content in test_messages:
        TextMemoryManager.save_message(
            user_id=user_id,
            session_id=session_id,
            message_type=message_type,
            content=content
        )
        # Add a dummy delay to simulate time passing
        logger.info(f"'{message_type}' type message added to conversation")
    
    logger.info(f"Test conversation created with {len(test_messages)} messages")
    return session_id

def test_memory_system():
    """Tests the memory system"""
    user_id = 1
    print("\n" + "="*80)
    print("\033[1;36mMEMORY SYSTEM TESTS\033[0m")
    print("="*80)
    
    # Create a test conversation
    session_id = create_test_conversation(user_id)
    
    # Test 1: Retrieve conversation history
    print("\n" + "-"*50)
    print("\033[1;33mTEST 1: Conversation History Retrieval\033[0m")
    print("-"*50)
    conversation_history = get_conversation_history(user_id, session_id)
    if conversation_history:
        print("[✓] SUCCESS: History retrieved ({} characters)".format(len(conversation_history)))
        print("Excerpt: {}...".format(conversation_history[:100].replace("\n", " ")))
        logger.info(f"Test 1 successful: History retrieved ({len(conversation_history)} characters)")
    else:
        print("[✗] FAIL: Unable to retrieve history")
        logger.error("Test 1 failed: Unable to retrieve history")
    
    # Test 2: Process a specific memory request
    print("\n" + "-"*50)
    print("\033[1;33mTEST 2: Processing a Memory-Related Request\033[0m")
    print("-"*50)
    memory_request = "Can you remind me what we talked about regarding deep learning?"
    print(f"Request: \"{memory_request}\"")
    memory_context = process_memory_request(memory_request, user_id, session_id)
    
    if memory_context:
        print("[✓] SUCCESS: The memory request generated context")
        print(f"Generated context: {memory_context[:100].replace('\n', ' ')}...")
        logger.info(f"Test 2 successful: The memory request generated context")
    else:
        print("[✗] FAIL: No context generated for the memory request")
        logger.error("Test 2 failed: No context generated for the memory request")
    
    # Test 3: Verify that memory keywords are correctly detected
    print("\n" + "-"*50)
    print("\033[1;33mTEST 3: Memory Keyword Detection\033[0m")
    print("-"*50)
    memory_keywords_tests = [
        ("Do you remember our discussion about AI?", True),
        ("Remind me what you said about TensorFlow", True),
        ("What is machine learning?", False),
        ("We previously talked about frameworks", True)
    ]
    
    test3_success = 0
    test3_total = len(memory_keywords_tests)
    
    print("Verifying memory request detection:")
    for prompt, expected in memory_keywords_tests:
        result = process_memory_request(prompt, user_id, session_id) is not None
        if result == expected:
            mark = "[✓]"
            status = "SUCCESS"
            test3_success += 1
            logger.info(f"Prompt '{prompt}' correctly {'identified' if expected else 'ignored'} as memory request")
        else:
            mark = "[✗]"
            status = "FAIL"
            logger.error(f"Prompt '{prompt}' incorrectly {'ignored' if expected else 'identified'} as memory request")
            print(f"{mark} {status}: \"{prompt}\" - {'Should be' if expected else 'Should not be'} a memory request")
    
    # Test 4: Verify image storage
    print("\n" + "-"*50)
    print("\033[1;33mTEST 4: Image Storage\033[0m")
    print("-"*50)
    # Create a small base64 test image
    test_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQ42mNk+P+/HgAFeAJYgZIL4AAAAABJRU5ErkJggg=="
    print("Attempting to save a base64 test image...")
    
    image_path = TextMemoryManager.save_uploaded_image(user_id, test_image)
    
    if image_path:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), image_path)
        if os.path.exists(full_path):
            print(f"[✓] SUCCESS: Image saved correctly to {image_path}")
            logger.info(f"Test 4 successful: Image saved correctly to {image_path}")
        else:
            print(f"[✗] FAIL: Invalid file path {full_path}")
            logger.error(f"Test 4 failed: Invalid file path {full_path}")
    else:
        print("[✗] FAIL: Failed to save image")
        logger.error("Test 4 failed: Failed to save image")
    
    # Test Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    # Calculate final test result
    test1_success = conversation_history is not None
    test2_success = memory_context is not None
    test4_success = image_path is not None and os.path.exists(full_path) if image_path else False
    
    total_tests = 4  # 4 total tests
    successful_tests = sum([
        1 if test1_success else 0,
        1 if test2_success else 0,
        test3_success,  # Already a count
        1 if test4_success else 0
    ])
    
    print(f"Tests successful: {successful_tests}/{total_tests}")
    print(f"Test 1 (History Retrieval): {'✓' if test1_success else '✗'}")
    print(f"Test 2 (Memory Request Processing): {'✓' if test2_success else '✗'}")
    print(f"Test 3 (Keyword Detection): {test3_success}/{test3_total}")
    print(f"Test 4 (Image Storage): {'✓' if test4_success else '✗'}")
    
    print("\n" + "="*50)
    print("Tests completed")
    print("="*50)
    
    return session_id

if __name__ == "__main__":
    try:
        print("\nStarting memory system tests...")
        session_id = test_memory_system()
        print(f"\nTest session ID: {session_id}")
        print("\nYou can now start the application and try asking the AI:")
        print("\"Can you remind me what we talked about regarding deep learning?\"")
    except Exception as e:
        print(f"\n[!] ERROR during test execution: {str(e)}")
        import traceback
        print("\nError details:")
        traceback.print_exc()
