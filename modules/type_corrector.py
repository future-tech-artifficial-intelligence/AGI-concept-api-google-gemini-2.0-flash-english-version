"""
Security module that intercepts and corrects incorrect data types.
Completely rewritten to provide maximum protection against type errors.
"""

import logging
import traceback
from typing import Any, Dict, Union

# Module Metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 1,  # MAXIMUM priority to run before all other modules
    "description": "Security module that protects against type errors",
    "version": "0.2.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Specific logger configuration with more details
logger = logging.getLogger('type_security')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    # Add a file handler with more details
    file_handler = logging.FileHandler('type_security.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Add a console handler for critical errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

def safe_copy(obj: Any) -> Any:
    """
    Creates a safe copy of an object based on its type.
    """
    try:
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: safe_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_copy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(safe_copy(item) for item in obj)
        else:
            # For unsupported types, attempt to convert them to string
            return str(obj)
    except Exception as e:
        logger.error(f"Error during copy: {str(e)}")
        return None

def process(data: Any, hook: str) -> Dict[str, Any]:
    """
    Security function that ensures data is always in a valid format.
    """
    try:
        logger.debug(f"[{hook}] Input: type={type(data)}")
        
        # 1. Protection against non-dictionary data
        if not isinstance(data, dict):
            logger.warning(f"[{hook}] Non-dictionary data detected: {type(data)}")
            
            if isinstance(data, str):
                logger.info(f"[{hook}] Converting a string to a dictionary")
                return {"text": data, "_secured": True}
            else:
                try:
                    # Attempt to convert to dictionary if possible
                    dict_data = dict(data) if hasattr(data, "__iter__") else {"value": str(data)}
                    dict_data["_secured"] = True
                    return dict_data
                except:
                    # If unsuccessful, create a new dictionary
                    return {"value": str(data) if data is not None else "", "_secured": True}
        
        # 2. Create a secure copy of the dictionary
        result = {}
        for key, value in data.items():
            # Protection against non-string keys
            safe_key = str(key) if not isinstance(key, str) else key
            
            # Special handling for the "text" key
            if safe_key == "text":
                if not isinstance(value, str):
                    logger.warning(f"[{hook}] 'text' key contains an incorrect type: {type(value)}")
                    result["text"] = str(value) if value is not None else ""
                    result["_text_corrected"] = True
                else:
                    result["text"] = value
            else:
                # Secure copy of other values
                result[safe_key] = safe_copy(value)
        
        # 3. Check for the presence of the "text" key if necessary in the response hook
        if hook == "process_response" and "text" not in result:
            logger.warning(f"[{hook}] 'text' key missing in response data")
            result["text"] = data.get("value", "") if isinstance(data, dict) else ""
            result["_text_added"] = True
        
        # 4. Add a security marker
        result["_secured"] = True
        
        return result
        
    except Exception as e:
        # In case of a critical error, return a minimal dictionary
        logger.critical(f"Critical error in the security module: {str(e)}")
        logger.critical(traceback.format_exc())
        return {"text": "Sorry, an error occurred.", "_error": str(e), "_secured": True}
