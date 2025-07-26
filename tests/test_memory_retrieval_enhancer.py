"""
Test for the memory retrieval enhancement module for artificial intelligence API GOOGLE GEMINI 2.0 FLASH
"""

import unittest
import sys
import os
import json
from typing import Dict, Any
import datetime

# Add parent directory to path for module import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.memory_retrieval_enhancer import enhance_memory_instructions, enrich_system_prompt, process
from memory_engine import MemoryEngine

class MockMemoryEngine:
    def get_recent_conversations(self, **kwargs):
        return [
            {
                'content': 'This is a test conversation',
                'time_ago': '5 minutes ago'
            }
        ]
    
    def get_long_term_memories(self, **kwargs):
        return [
            {
                'category': 'preference',
                'content': 'The user prefers blue'
            }
        ]
    
    def get_memory_context(self, **kwargs):
        return "Recent conversations memory:\n1. 5 minutes ago: This is a test conversation"

class TestMemoryRetrievalEnhancer(unittest.TestCase):
    
    def setUp(self):
        # Temporarily replace the MemoryEngine instance with our mock
        import modules.memory_retrieval_enhancer as module
        self.original_memory_engine = module.memory_engine
        module.memory_engine = MockMemoryEngine()
        
    def tearDown(self):
        # Restore the original instance
        import modules.memory_retrieval_enhancer as module
        module.memory_engine = self.original_memory_engine
    
    def test_enhance_memory_instructions(self):
        # Prepare test data
        data = {
            'user_id': 1,
            'session_id': 'test_session',
            'context': {}
        }
        
        # Call the function to test
        result = enhance_memory_instructions(data)
        
        # Verify results
        self.assertTrue('instructions' in result['context'])
        self.assertTrue(isinstance(result['context']['instructions'], list))
        self.assertTrue(len(result['context']['instructions']) > 0)
        self.assertTrue('memory_instructions_enhanced' in result['context'])
        self.assertTrue(result['context']['memory_instructions_enhanced'])
        
        # Verify instruction content
        found_memory_instruction = False
        for instruction in result['context']['instructions']:
            if "CRITICAL MEMORY INSTRUCTION" in instruction:
                found_memory_instruction = True
                break
                
        self.assertTrue(found_memory_instruction)
        
    def test_enrich_system_prompt(self):
        # Prepare test data
        data = {
            'user_id': 1,
            'system_prompt': 'Original system instruction'
        }
        
        # Call the function to test
        result = enrich_system_prompt(data)
        
        # Verify results
        self.assertTrue('system_prompt' in result)
        self.assertIn("CRITICAL MEMORY INSTRUCTION", result['system_prompt'])
        self.assertTrue(result['metadata']['memory_prompt_enriched'])
        
    def test_process_function(self):
        # Prepare test data
        data = {
            'user_id': 1,
            'session_id': 'test_session',
            'context': {},
            'system_prompt': 'Original system instruction'
        }
        
        # Call the function to test
        result = process(data, 'process_request')
        
        # Verify results
        self.assertTrue('instructions' in result['context'])
        self.assertIn("CRITICAL MEMORY INSTRUCTION", result['system_prompt'])
        self.assertTrue(result['context']['memory_instructions_enhanced'])
        self.assertTrue(result['metadata']['memory_prompt_enriched'])
        
if __name__ == '__main__':
    unittest.main()
