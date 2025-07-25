"""
Time management module for artificial intelligence API GOOGLE GEMINI 2.0 FLASH
This module allows the artificial intelligence API GOOGLE GEMINI 2.0 FLASH to access real-time hour and date,
and to understand temporal concepts.
"""

import datetime
import pytz
import re
from typing import Dict, Any, Optional, List, Tuple

# Default configuration
DEFAULT_TIMEZONE = 'Europe/Paris'
AVAILABLE_TIMEZONES = pytz.common_timezones

# Constants for memory retention
MEMORY_RETENTION = {
    'SHORT_TERM': datetime.timedelta(minutes=30),
    'MEDIUM_TERM': datetime.timedelta(days=1),
    'LONG_TERM': datetime.timedelta(days=7)
}

# Cache for frequently used timezones
timezone_cache = {}

def get_current_datetime(timezone_str: str = DEFAULT_TIMEZONE) -> datetime.datetime:
    """
    Gets the current hour and date in the specified timezone

    Args:
        timezone_str: The timezone (default: Europe/Paris)

    Returns:
        The current datetime object in the specified timezone
    """
    import logging
    logger = logging.getLogger(__name__)

    original_timezone = timezone_str

    # Use cache if available
    if timezone_str in timezone_cache:
        tz = timezone_cache[timezone_str]
        logger.debug(f"Timezone retrieved from cache: {timezone_str}")
    else:
        # Check if the timezone is valid
        if timezone_str not in AVAILABLE_TIMEZONES:
            logger.warning(f"Invalid timezone detected: {timezone_str}, using default: {DEFAULT_TIMEZONE}")
            timezone_str = DEFAULT_TIMEZONE

        try:
            # Get the timezone object and cache it
            tz = pytz.timezone(timezone_str)
            timezone_cache[timezone_str] = tz
            logger.info(f"Timezone configured and cached: {timezone_str}")
        except pytz.exceptions.UnknownTimeZoneError:
            logger.error(f"Error configuring timezone {timezone_str}, using UTC")
            tz = pytz.timezone('UTC')
            timezone_cache[timezone_str] = tz

    # Get the current datetime with the timezone
    current_dt = datetime.datetime.now(tz)

    if original_timezone != timezone_str:
        logger.warning(f"Timezone changed from {original_timezone} to {timezone_str}")

    logger.debug(f"Current time in {timezone_str}: {current_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    return current_dt

def format_datetime(dt: datetime.datetime, format_str: str = "full") -> str:
    """
    Formats a datetime object into a readable string

    Args:
        dt: The datetime object to format
        format_str: The desired format ("full", "date", "time", "short")

    Returns:
        The formatted string
    """
    if format_str == "full":
        return dt.strftime("%A %d %B %Y, %H:%M:%S")
    elif format_str == "date":
        return dt.strftime("%d/%m/%Y")
    elif format_str == "time":
        return dt.strftime("%H:%M")
    elif format_str == "short":
        return dt.strftime("%d/%m/%Y %H:%M")
    else:
        return dt.strftime(format_str)

def is_time_request(text: str) -> bool:
    """
    Detects if a text contains an explicit request for time or date

    Args:
        text: The text to analyze

    Returns:
        True if the text asks for time or date, False otherwise
    """
    text_lower = text.lower()

    # Patterns to detect time and date requests
    time_patterns = [
        r"what\s+time(?:\s+is\s+it)?",
        r"the\s+current\s+time",
        r"time\s+(?:is\s+it|current)",
        r"time\s+please"
    ]

    date_patterns = [
        r"what\s+(?:date|day)(?:\s+is\s+it)?",
        r"(?:date|day)\s+(?:are\s+we|is\s+it)",
        r"(?:date|day)\s+of\s+today",
        r"what\s+day\s+(?:are\s+we|is\s+it)"
    ]

    # Combine all patterns
    all_patterns = time_patterns + date_patterns

    # Check if any of the patterns match
    for pattern in all_patterns:
        if re.search(pattern, text_lower):
            return True

    return False

def should_remember_conversation(creation_time: datetime.datetime, current_time: Optional[datetime.datetime] = None) -> bool:
    """
    Determines if a conversation should be memorized based on its age

    Args:
        creation_time: The time the conversation was created
        current_time: The current time (default: now)

    Returns:
        True if the conversation should be memorized, False otherwise
    """
    if current_time is None:
        current_time = datetime.datetime.now(creation_time.tzinfo)

    # Calculate time difference
    time_diff = current_time - creation_time

    # If less than 7 days, memorize
    return time_diff.days < 7

def timestamp_to_readable_time_diff(timestamp: str) -> str:
    """
    Converts a timestamp to a readable time difference

    Args:
        timestamp: The timestamp in ISO format

    Returns:
        A string describing the time difference (e.g., "2 days ago")
    """
    try:
        # Convert timestamp to datetime object
        dt = datetime.datetime.fromisoformat(timestamp)

        # Get current time with the same timezone
        now = datetime.datetime.now(dt.tzinfo)

        # Calculate the difference
        diff = now - dt

        # Convert to readable phrasing
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds >= 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds >= 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "just now"
    except (ValueError, TypeError):
        return "at an unknown time"
