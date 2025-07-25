"""
Temporal Awareness Module for Gemini
This module enables artificial intelligence GOOGLE GEMINI 2.0 FLASH API to intelligently respond to time-related queries,
without introducing temporal references into every response.
"""

import logging
import re
from typing import Dict, Any, Optional

from time_engine import get_current_datetime, format_datetime, is_time_request

# Module Metadata
MODULE_METADATA = {
    "enabled": True,
    "priority": 40,  # Medium priority
    "description": "Manages temporal awareness and time/date references",
    "version": "1.0.0",
    "dependencies": [],
    "hooks": ["process_request", "process_response"]
}

# Logger configuration
logger = logging.getLogger(__name__)

def generate_time_response(timezone: str = "Europe/Paris") -> str:
    """
    Generates a response concerning the current time and date
    
    Args:
        timezone: The timezone to use
        
    Returns:
        A response containing the time and date
    """
    current_time = get_current_datetime(timezone)
    time_str = format_datetime(current_time, "time") # Changed "heure" to "time"
    date_str = format_datetime(current_time, "full") # Changed "complet" to "full"
    
    return f"It is currently {time_str} on {date_str} in the {timezone} timezone." # Adjusted sentence structure

def extract_timezone(text: str, default_timezone: str = "Europe/Paris") -> str:
    """
    Extracts the timezone mentioned in the text, if present
    
    Args:
        text: The text to analyze
        default_timezone: The default timezone
        
    Returns:
        The extracted timezone or the default one
    """
    # List of common timezones and their synonyms
    timezone_patterns = {
        "Europe/Paris": ["france", "paris", "french"], # Adjusted "français/française" to "french"
        "America/New_York": ["new york", "united states", "usa", "america"], # Adjusted "états-unis/etats-unis" to "united states"
        "Asia/Tokyo": ["japan", "tokyo"], # Adjusted "japon" to "japan"
        "Europe/London": ["london", "england", "united kingdom", "uk"], # Adjusted "londres", "angleterre", "royaume-uni"
        # Add other timezones as needed
    }
    
    text_lower = text.lower()
    
    # Look for timezone mentions
    for timezone, patterns in timezone_patterns.items():
        if any(pattern in text_lower for pattern in patterns):
            return timezone
    
    return default_timezone

def process(data: Dict[str, Any], hook: str) -> Dict[str, Any]:
    """
    Processes requests and responses to manage temporal references
    
    Args:
        data: The data to process
        hook: The hook called (process_request or process_response)
        
    Returns:
        The modified data
    """
    try:
        # Request processing
        if hook == "process_request" and "text" in data:
            user_input = data["text"]
            
            # Check if it's a time request (and only an explicit request)
            if is_time_request(user_input):
                # Mark this request as a time request
                data["is_time_request"] = True
                
                # Extract the timezone if mentioned
                timezone = extract_timezone(user_input)
                data["requested_timezone"] = timezone
                
                logger.info(f"Time request detected, timezone: {timezone}")
            else:
                # Ensure this is not treated as a time request
                data["is_time_request"] = False
            
            return data
        
        # Response processing
        elif hook == "process_response":
            # Only if it was an explicit time request, generate an appropriate response
            if data.get("is_time_request", False) and "text" in data:
                timezone = data.get("requested_timezone", "Europe/Paris")
                time_response = generate_time_response(timezone)
                
                # Completely replace the response for direct time/date requests
                data["text"] = time_response
                logger.info(f"Temporal response generated for timezone: {timezone}")
            
            return data
        
        return data
    
    except Exception as e:
        logger.error(f"Error in temporal awareness module: {str(e)}")
        return data
