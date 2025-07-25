"""
Logging module to identify issues within the module system.
This module logs incoming and outgoing data types for each module.
"""

import logging
import inspect
from typing import Any, Dict

# Module metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 10,  # Very high priority to run before other modules
    "description": "Data type logging module",
    "version": "0.1.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Specific logger configuration
logger = logging.getLogger('module_debugger')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    file_handler = logging.FileHandler('module_debug.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Main function that logs data types.
    """
    # Log the type of input data
    logger.debug(f"[{hook}] Input: Type={type(data)}")
    
    # For dictionaries, log the type of each value
    if isinstance(data, dict):
        for key, value in data.items():
            logger.debug(f"[{hook}] Key={key}, Type={type(value)}")
            
            # If the value is a dictionary, log its types as well
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    logger.debug(f"[{hook}] Sub-key={key}.{sub_key}, Type={sub_value}")
    
    # Get the name of the calling module (which will be executed after this one)
    caller_frame = inspect.currentframe().f_back
    if caller_frame:
        caller_module = caller_frame.f_globals.get('__name__')
        logger.debug(f"[{hook}] Next module: {caller_module}")
    
    # Do not modify the data, just log it
    return data
