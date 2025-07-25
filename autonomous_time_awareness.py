"""
**Autonomous Time Awareness Module for artificial intelligence API GOOGLE GEMINI 2.0 FLASH**
This module allows artificial intelligence API GOOGLE GEMINI 2.0 FLASH to maintain a constant awareness of time and automatically access temporal information.
"""

import datetime
import pytz
import logging
from typing import Dict, Any, Optional

# Logging configuration
logger = logging.getLogger(__name__)

class AutonomousTimeAwareness:
    """
    Class to manage the autonomous time awareness of the artificial intelligence API GOOGLE GEMINI 2.0 FLASH.
    """
    
    def __init__(self, default_timezone: str = "Europe/Paris"):
        """
        Initializes the autonomous time awareness system.
        
        Args:
            default_timezone: Default timezone
        """
        self.default_timezone = default_timezone
        self.user_timezones = {}  # Cache of timezones per user
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate the default timezone
        try:
            pytz.timezone(self.default_timezone)
            self.logger.info(f"Default timezone validated: {self.default_timezone}")
        except pytz.exceptions.UnknownTimeZoneError:
            self.logger.error(f"Invalid default timezone: {self.default_timezone}, using UTC")
            self.default_timezone = "UTC"
        
    def set_user_timezone(self, user_id: int, timezone: str):
        """
        Sets the timezone for a specific user.
        
        Args:
            user_id: User ID
            timezone: User's timezone
        """
        try:
            # Validate that the timezone exists
            pytz.timezone(timezone)
            self.user_timezones[user_id] = timezone
            self.logger.info(f"Timezone set for user {user_id}: {timezone}")
        except pytz.exceptions.UnknownTimeZoneError:
            self.logger.warning(f"Invalid timezone for user {user_id}: {timezone}")
    
    def get_user_timezone(self, user_id: int) -> str:
        """
        Retrieves a user's timezone or returns the default.
        
        Args:
            user_id: User ID
            
        Returns:
            User's timezone or default
        """
        if user_id and user_id in self.user_timezones:
            timezone = self.user_timezones[user_id]
            self.logger.info(f"Timezone retrieved for user {user_id}: {timezone}")
            return timezone
        
        self.logger.info(f"Using default timezone for user {user_id}: {self.default_timezone}")
        return self.default_timezone
    
    def get_current_awareness(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Gets the complete current time awareness.
        
        Returns:
            Dictionary containing all temporal information
        """
        try:
            # Determine the timezone to use
            timezone = self.get_user_timezone(user_id) if user_id else self.default_timezone
            
            # Get the current time in the appropriate timezone
            current_dt = datetime.datetime.now(pytz.timezone(timezone))
            
            # Build the complete time awareness
            awareness = {
                "current_moment": {
                    "hour": current_dt.strftime("%H:%M:%S"),
                    "date": current_dt.strftime("%A %d %B %Y"),
                    "timestamp": current_dt.timestamp(),
                    "iso_format": current_dt.isoformat()
                },
                "temporal_context": {
                    "day_of_week": current_dt.strftime("%A"),
                    "day_of_month": current_dt.day,
                    "month": current_dt.strftime("%B"),
                    "year": current_dt.year,
                    "timezone": timezone,
                    "user_id": user_id
                },
                "narrative_awareness": self._generate_temporal_narrative(current_dt),
                "meta_awareness": {
                    "type": "autonomous_time_awareness",
                    "version": "1.0.0",
                    "last_update": current_dt.isoformat()
                }
            }
            
            return awareness
            
        except Exception as e:
            self.logger.error(f"Error generating time awareness: {str(e)}")
            return self._get_fallback_awareness()
    
    def _generate_temporal_narrative(self, dt: datetime.datetime) -> str:
        """
        Generates a narrative description of the current moment.
        
        Args:
            dt: The current datetime object
            
        Returns:
            Narrative description of time
        """
        try:
            hour = dt.hour
            day_name = dt.strftime("%A")
            date_str = dt.strftime("%d %B %Y")
            time_str = dt.strftime("%H:%M")
            
            # Determine the period of the day
            if 5 <= hour < 12:
                period = "morning"
            elif 12 <= hour < 17:
                period = "afternoon"
            elif 17 <= hour < 21:
                period = "evening"
            else:
                period = "night"
            
            narrative = f"It is {day_name}, {date_str}, {time_str} in the {period}."
            
            return narrative
            
        except Exception as e:
            self.logger.error(f"Error generating narrative: {str(e)}")
            return "Time awareness is being retrieved."
    
    def _get_fallback_awareness(self) -> Dict[str, Any]:
        """
        Returns a fallback time awareness in case of an error.
        
        Returns:
            Basic time awareness
        """
        return {
            "current_moment": {
                "hour": "Undetermined",
                "date": "Undetermined",
                "timestamp": 0,
                "iso_format": "Undetermined"
            },
            "temporal_context": {
                "day_of_week": "Undetermined",
                "day_of_month": 0,
                "month": "Undetermined",
                "year": 0,
                "timezone": self.default_timezone
            },
            "narrative_awareness": "My time awareness is temporarily unavailable.",
            "meta_awareness": {
                "type": "autonomous_time_awareness_fallback",
                "version": "1.0.0",
                "last_update": "Error"
            }
        }
    
    def get_temporal_context_for_ai(self) -> str:
        """
        Gets a formatted temporal context for the AI.
        
        Returns:
            Temporal context as a string
        """
        awareness = self.get_current_awareness()
        narrative = awareness["narrative_awareness"]
        
        return f"[Autonomous Time Awareness] {narrative}"

# Global instance for AI use
autonomous_time = AutonomousTimeAwareness()

def get_ai_temporal_context(user_id: Optional[int] = None) -> str:
    """
    Utility function to get the temporal context for the AI.
    
    Args:
        user_id: User ID to use their timezone
    
    Returns:
        Formatted temporal context
    """
    awareness = autonomous_time.get_current_awareness(user_id)
    narrative = awareness["narrative_awareness"]
    return f"[Autonomous Time Awareness] {narrative}"

def get_full_temporal_awareness(user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Utility function to get the complete time awareness.
    
    Args:
        user_id: User ID to use their timezone
    
    Returns:
        Complete time awareness
    """
    return autonomous_time.get_current_awareness(user_id)

def set_user_timezone(user_id: int, timezone: str):
    """
    Utility function to set a user's timezone.
    
    Args:
        user_id: User ID
        timezone: User's timezone
    """
    autonomous_time.set_user_timezone(user_id, timezone)
